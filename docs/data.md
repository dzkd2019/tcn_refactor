# 数据集格式说明

本文说明当前项目的合成场景数据、NPZ 文件格式、训练窗口构造方式，以及窗口
对齐和随机偏移的含义。数据管线的目标是：训练阶段使用与部署阶段相同的因果
特征和 15 秒响应窗口，评估阶段仍然保留事件真值，用于事后计算检测延迟和
浓度误差。

## 文件格式

`main.py generate` 生成压缩的 NumPy NPZ 文件：

```bash
uv run python main.py generate \
  --output data/train_v5.npz --count 1600 --seed 7026
```

训练、验证和测试文件应使用不同的随机种子和独立文件。NPZ 只保存数值数组，
读取时使用 `allow_pickle=False`，不依赖 Python 对象反序列化。

```python
import numpy as np

with np.load("data/train_v5.npz", allow_pickle=False) as data:
    print(data.files)
```

## 文件顶层结构

假设有 `N` 个场景、每个场景有 `T` 个采样点、最多 `E` 个事件，则顶层键为：

| 键 | 形状 | dtype | 含义 |
|---|---:|---|---|
| `voltage` | `(N, T)` | `float32` | 带基线、响应和噪声的原始电压 |
| `temperature` | `(N, T)` | `float32` | 传感器观测温度，单位 ℃ |
| `humidity` | `(N, T)` | `float32` | 传感器观测相对湿度，单位 %RH |
| `baseline` | `(N, T)` | `float32` | 合成器内部的真实基线，仅用于诊断，不用于训练特征 |
| `concentration` | `(N, T)` | `float32` | 到达传感器表面的逐点真实浓度 |
| `event_start` | `(N, E)` | `int32` | 阀门命令事件起始索引；无事件位置为 `-1` |
| `event_end` | `(N, E)` | `int32` | 阀门命令事件结束索引；无事件位置为 `-1` |
| `event_ppm` | `(N, E)` | `float64` | 事件设定浓度标签；无事件位置为 `NaN` |
| `seeds` | `(N,)` | `int64` | 每个场景的独立随机种子 |
| `schema_version` | `()` | `int32` | 文件格式版本，当前为 `2` |
| `generator_version` | `()` | string | 数据生成器版本，当前为 `calibrated_stateful_v5` |
| `config_json` | `()` | string | `SystemConfig` 的 JSON 序列化结果 |

`T` 由场景时长和采样率决定。默认配置为 100 Hz、180 秒，因此 `T=18000`；
模型窗口为 15 秒，即 `L=1500` 个采样点。`E` 是同一文件中所有场景的最大
事件数，事件不足的场景使用填充值。

## 单个场景的语义

`load_scenarios()` 将第 `i` 行数组恢复为一个 `Scenario`：

```python
scenario.voltage             # (T,)
scenario.temperature_c       # (T,)
scenario.humidity_rh         # (T,)
scenario.true_baseline       # (T,)
scenario.concentration_ppm   # (T,)
scenario.events              # tuple[GasEvent, ...]
scenario.seed                # int
```

每个 `GasEvent(start, end, concentration_ppm)` 表示阀门命令的起止和设定浓度。
它不是传感器立刻感受到气体的时间：命令还要经过管路纯延迟和气室混合，
`scenario.concentration_ppm` 才表示每个采样点真正到达传感器表面的浓度。因此：

```text
阀门事件起点 ── 管路延迟/气室混合 ── 传感器响应开始
     event.start                         concentration 上升
```

事件真值用于训练窗口索引和回放后的指标匹配；模型输入不能读取
`true_baseline`、逐点真实浓度或事件起止。

## 合成场景的生成流程

每个场景使用独立随机种子，生成流程由 `synthetic.py:generate_scenarios()` 完成。

### 1. 温湿度环境

温度由随机初值、慢变化和随机游走组成；湿度与温度相关，再叠加通风扰动，并
裁剪到 `[0, 100] %RH`。最终保存的是带观测噪声的 `temperature` 和 `humidity`。

### 2. 阀门命令和事件

首次事件在清洁空气标定结束后随机出现，后续事件之间的空气间隔也随机。事件
持续时间分为短脉冲、常规暴露和长暴露三类，避免所有样本都具有相同的时间长度。
设定浓度从 `(100, 200, 400, 600, 800, 1000, 1200)` ppm 附近随机取值。

### 3. 气路和气室动态

阀门命令先经过约 0.3～0.8 秒的纯延迟，再通过一阶气室混合得到逐点表面浓度。
因此事件起点、表面浓度起点和检测器触发点通常不同。

### 4. 传感器响应、基线和噪声

传感器响应使用带平方根浓度关系的饱和模型，并由快、慢两个状态共同形成；
温湿度会改变响应幅度和时间常数。基线包含器件偏置、温湿度项、随机游走、
老化斜率和暴露诱导漂移。电压噪声包含相关低频噪声、异方差白噪声、尖峰和
ADC 量化。

## 从场景到训练样本

`StreamWindowDataset` 不把完整场景直接交给模型，而是先构造一个
`WindowRecord`，再在 `__getitem__` 中截取长度为 `L` 的窗口：

```python
WindowRecord(
    scenario_index, event_start, event_end,
    concentration_ppm, baseline,
)
```

训练样本最终只返回：

```python
x, y = dataset[index]
# x: (8, L) float32
# y: () float32，事件浓度（ppm）
```

基线不使用 NPZ 中的 `baseline` 真值。事件对齐模式使用事件前
`calibration_s` 秒的观测电压中位数；检测对齐模式使用部署侧
`BaselineEstimator` 在线估计出的基线。

## 窗口对齐方式

数据集支持两种 `alignment`：`event` 和 `detection`。

### `event`：事件真值对齐

```python
dataset = StreamWindowDataset(path, config=config, alignment="event")
```

每个窗口从 `GasEvent.start` 开始：

```text
event.start
    ↓
[────────────── L 个采样点 ──────────────]
```

这是一种理想、可重复的窗口，适合分析“从事件真值起点开始，模型随时间如何
收敛”。`window-eval` 默认使用该模式，因此它不模拟响应检测延迟。

如果事件剩余长度不足 `L`，该事件不会进入数据集。

### `detection`：检测器触发对齐

```python
dataset = StreamWindowDataset(path, config=config, alignment="detection")
```

数据集会先逐点运行与部署相同的 `ResponseDetector` 和基线估计器。当检测器从
未响应变为响应时，使用该时刻作为窗口起点：

```text
event.start ───── 检测延迟 ───── detector trigger
                                      ↓
                              [────── L ──────]
```

训练和独立验证目前使用检测对齐：

```python
StreamWindowDataset(..., alignment="detection")
```

这样训练窗口与 `StreamSession` 实际部署时的窗口语义一致，模型不会只适应
理想的事件起点。检测器无法在某个事件内触发，或触发后场景剩余长度不足 `L`，
该事件不会生成检测对齐记录。

## 随机偏移

在任一对齐起点之后，还可以启用随机偏移。`max_offset_s` 表示允许向后移动的
最大秒数；实际偏移按整数采样点均匀抽取：

```python
max_offset = min(
    round(max_offset_s * config.fs_hz),
    event_end - event_start - config.window_samples,
)
offset = randint(0, max_offset)
window_start = event_start + offset
window_end = window_start + config.window_samples
```

例如 100 Hz 采样、最大偏移 4 秒时，窗口可能在对齐点之后 0～400 个采样点
之间开始：

```text
对齐起点       +0 s                 +4 s
                ├─────────────────────┤
                └── 随机选择窗口起点 ──┘
```

偏移用于模拟检测延迟或窗口起点抖动，避免模型把固定的“事件开始后第几秒”当作
浓度线索。它不是打乱样本顺序，也不是改变浓度标签；窗口仍来自同一个事件，
标签仍为该事件的 `event_ppm`。

偏移具有可复现性。第 `epoch` 轮、第 `index` 个样本使用：

```python
np.random.default_rng(seed + 1_000_003 * epoch + index)
```

因此同一事件在不同 epoch 可以看到不同窗口，而使用相同种子仍能复现实验。
偏移还会受事件剩余长度限制，不能截到事件范围或场景末尾之外。

### 当前项目的实际设置

| 使用位置 | 对齐方式 | `max_offset_s` | 含义 |
|---|---|---:|---|
| `window-eval` | `event` | `0` | 从事件真值起点截取理想测试窗口 |
| `train_stream_model` 训练集 | `detection` | `0` | 从部署检测器触发点截取窗口 |
| `train_stream_model` 验证集 | `detection` | `0` | 用真实部署起点选择 checkpoint |
| `replay` | 不使用 Dataset 窗口 | 不适用 | 逐点运行 `StreamSession`，由检测器实时决定起点 |

`max_offset_s` 仍保留在 `StreamWindowDataset` 和训练函数 API 中，实验需要时可以
显式传入非零值；当前命令行训练参数未单独暴露该选项，所以默认训练不会额外
随机移动窗口。

## 八通道特征

`build_window()` 和在线的 `FeatureBuilder.build_one()` 使用完全相同的公式，
输出形状为 `(8, L)`：

| 通道 | 名称 | 含义 |
|---:|---|---|
| 0 | normalized response voltage | `(voltage - baseline) / voltage_ref` |
| 1 | raw derivative | 相邻电压差乘采样率，再除以 `dvdt_ref` |
| 2 | running mean | 从窗口开始到当前时刻的响应累计均值 |
| 3 | temperature | `(temperature - 50) / 30` |
| 4 | humidity | `(humidity - 40) / 40` |
| 5 | elapsed response time | `min(sample_count / L, 1)` |
| 6 | smoothed slope | 最近约 2 秒因果平滑响应的斜率 |
| 7 | steady-state proxy | `y_f + tau * dy_f/dt` 的弱动态先验 |

所有统计量都是因果的。第 `t` 个特征只使用当前及此前样本，不读取未来数据。
第 7 通道不执行浓度曲线反演，只作为网络可以忽略或修正的弱先验。

## Dataset 和 DataLoader 形状

```python
dataset = StreamWindowDataset("data/train_v5.npz", alignment="detection")
x, y = dataset[0]
print(x.shape, y.shape)  # torch.Size([8, 1500]), torch.Size([])
```

经过 `DataLoader(batch_size=B)` 后：

```text
x: (B, 8, 1500)
y: (B,)
```

模型输出为 `(B, 1500)`，训练标签沿时间维复制进行监督；部署时只取窗口最后
一个时间步，并在 `StreamSession` 中锁定浓度读数。

## 完整数据流

```text
generate_scenarios()
    环境 + 阀门命令 + 气路混合 + 器件响应 + 基线漂移 + 噪声
        │
        ▼
save_scenarios() → data/*.npz（二维数值数组，含事件真值）
        │
        ├── alignment="event"
        │       从 event.start 截取窗口
        │
        └── alignment="detection"
                先运行 BaselineEstimator + ResponseDetector
                从实际触发点截取窗口
                        │
                        ▼
                可选随机偏移
                        │
                        ▼
                build_window() → (8, L)
                        │
                        ▼
                DataLoader → (B, 8, L)
                        │
                        ▼
                StreamingTCN → (B, L)
```

端到端 `replay` 不经过上述离线窗口切分，而是将原始场景逐点送入
`StreamSession.push()`；事件真值只在事后用于计算检测延迟、漏检数和最终浓度
误差。这也是对齐窗口指标和完整部署指标必须分别报告的原因。

## 直接读取场景和事件真值

```python
from tcn_refactor.synthetic import load_scenarios

scenarios = load_scenarios("data/test_v5.npz")
scenario = scenarios[0]
event = scenario.events[0]

print(scenario.voltage.shape)          # (T,)
print(event.start, event.end)          # 阀门命令索引
print(event.concentration_ppm)         # 事件浓度标签
print(scenario.concentration_ppm.shape)  # (T,)
```

调试时可以比较 `event.start`、检测器触发点和 `concentration_ppm` 开始上升的
位置，但训练和部署特征均不得使用真实基线或逐点真实浓度。
