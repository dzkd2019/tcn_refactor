# tcn-refactor

面向部署的气体传感器响应加速与温湿度/基线漂移补偿系统。

模型结构、训练优化过程、实验演进和指标边界见
[`docs/model_optimization.md`](docs/model_optimization.md)。

## ROCm 环境

项目固定使用 PyTorch ROCm 7.2：`torch` 仅从南京大学 ROCm 7.2 镜像解析，
其余依赖从清华大学镜像解析。安装并验证环境：

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```

输出中的版本应带有 `+rocm7.2`，`torch.version.hip` 不应为空，设备名应为
AMD GPU。ROCm 为兼容 PyTorch API 仍使用 `cuda` 作为设备名，因此下列命令中的
`--device cuda` 是正确写法，并不表示使用 NVIDIA CUDA。

项目只保留逐样本（stream）推理路径：`StreamSession.push(sample)` 接收一条
传感器读数，完成基线标定、响应检测、因果特征构造和浓度输出。合成场景生成器
保留完整事件真值，用于训练、回放及端到端验证。

```bash
uv run python main.py generate --output data/train_v5.npz --count 1600 --seed 7026
uv run python main.py generate --output data/val_v5.npz --count 320 --seed 7027
uv run python main.py generate --output data/test_v5.npz --count 320 --seed 7028
uv run python main.py train --data data/train_v5.npz --val-data data/val_v5.npz \
  --checkpoint checkpoints/stream_v5_langmuir_rocm72.pt --epochs 50 --lr 1e-4 \
  --warmup-epochs 2 --min-lr 2e-6 --device cuda
uv run python main.py train --data data/train_v5.npz --val-data data/val_v5.npz \
  --checkpoint checkpoints/stream_v5_deployment_rocm72.pt --epochs 30 --lr 3e-5 \
  --warmup-epochs 1 --min-lr 1e-6 --device cuda \
  --resume checkpoints/stream_v5_langmuir_rocm72.pt
uv run python main.py replay --data data/test_v5.npz \
  --checkpoint checkpoints/stream_v5_deployment_rocm72.pt --device cuda
```

训练与部署共享 `FeatureBuilder`；模型没有跨时间归一化层，保证任意时刻的输出
不会依赖未来样本。

## 数据流

训练数据以一个 `Scenario` 为单位保存：每个场景同时包含逐点电压、温度、湿度、
真实基线、真实浓度和每一次注入的起止索引。训练集复用部署侧基线估计器与响应
检测器，从实际检测时刻截取窗口；短脉冲结束后的恢复段也参与训练。特征形状为
`(8, L)`，DataLoader 拼为 `(B, 8, L)` 后送入模型，模型输出 `(B, L)`。

8 个输入通道依次为：归一化响应电压、原始电压导数、响应累计均值、温度、湿度、
响应持续时间、2 秒因果平滑斜率、弱稳态响应外推。最后一个通道使用
`y_f + tau * dy_f/dt`，但不执行浓度标定曲线的逆变换，也不直接作为模型输出；
它只是网络可以忽略或修正的弱先验。这样既利用了传感器响应通常具有惯性的事实，
又不会强制假设 PdCu 合金传感器严格服从单一一阶方程。

部署时不使用事件真值。`StreamSession.push(Sample)` 依次完成基线标定、响应检测、
共享特征构造和模型推理；当累积满 15 秒响应窗口时，读取模型最后一个时间步的预测
并锁定，直到检测器判断响应结束。

## 合成数据模型

第 5 版生成器先构造随机阀门命令，再模拟管路延迟和气室混合，得到传感器表面的
连续浓度。每个场景抽取一组固定的虚拟器件参数，使用带饱和的平方根浓度关系和
快慢双状态响应；多次暴露共享同一状态，能够自然形成不完全恢复和历史依赖。
温湿度具有环境相关性，基线包含器件差异、自然扩散漂移和暴露诱导偏移；测量噪声
包含相关低频分量、异方差白噪声、尖峰和 ADC 量化。事件起点、持续时间及空气间隔
均随机，并包含一部分不足 15 秒的短脉冲。

这些机制来自 `docs/data.md` 的评审建议。v5 面向同一支已标定传感器和同一气路的
重复实验，器件级参数只保留短期重复性；跨器件泛化需要额外提供设备标定数据。
生成器仍未使用真实 PdCu 器件参数，因此不是经过实验验证的数字孪生。

模型使用深层 TCN 动态支路和浅层校准支路，通过可微 Langmuir 输出层回归连续
浓度；评估和部署时再投影到实验使用的七个标准气体等级（100–1200 ppm）。训练
采用对数均方损失、AdamW、梯度裁剪以及 warmup+cosine 调度。

使用 `seed=7026/7027/7028` 生成的 1600/320/320 个 v5 场景，短脉冲微调后的
最佳验证 MAPE 为 8.15%。开发测试集完整回放检出 320/320 个事件，平均检测延迟
3.103 秒，最终 MAE 为 66.929 ppm，MAPE 为 9.680%。该测试结果曾用于定位短脉冲
训练缺口，因此固定模型后又使用 `seed=8028` 生成 320 场景确认集；确认回放检出
320/320 个事件，平均检测延迟 3.140 秒，MAE 为 67.467 ppm，MAPE 为 9.872%。
从事件真值起点截取的开发测试窗口在 5/10/15 秒的 MAPE 分别为
132.521%/42.840%/14.191%，说明当前模型只在
“检测后等待 15 秒并输出标准等级”的部署路径上达到 10% 以下。

对应权重为 `checkpoints/stream_v5_deployment_rocm72.pt`。9.872% 不能外推为任意
连续浓度或真实器件精度；接入实测数据后必须重新标定 Langmuir 系数和浓度等级，
并同时报告连续回归与等级投影指标。

## 旧生成器历史结果

以下结果由旧版事件波形叠加生成器产生，仅用于重构前后的历史对照。第 2 版数据分布
与 v5 数据分布不兼容，必须重新生成训练/验证/测试数据并重新训练，不能沿用这些数值或权重。

模型是 8 通道直接回归 TCN。它不再以一阶方程的浓度反演值为输出基线，而是根据
完整因果序列直接预测浓度。每个残差块的两层因果卷积均使用
`torch.nn.utils.parametrizations.weight_norm`，将卷积核的方向和大小解耦优化。
训练使用相对 Huber 损失：误差大于 5% 时近似 MAPE，小误差区间使用平滑二次项。
优化器为 AdamW；前 2 轮从峰值学习率的 10% 线性 warmup，随后余弦退火到
`1e-5`。独立验证集最佳 MAPE 为 7.205%。

独立测试集结果：

- 对齐窗口：5 秒 MAPE 30.40%，10 秒 15.18%，15 秒 9.07%。这是更困难的
  “从真实事件起点计时”指标，包含随机多动力学造成的不可辨识性。
- 逐点端到端回放：320/320 事件检出，平均检测延迟 2.515 秒。
- 部署路径最终 MAE 46.988 ppm，MAPE 8.000%。部署计时从检测器实际触发点开始，
  与 `StreamSession` 锁定读数的真实使用方式一致。

旧版参考权重为 `checkpoints/stream_rocm72_warmup_cosine.pt`。数据和权重均被
`.gitignore` 排除；该权重不应作为第 2 版生成器的评估模型。

上述结果仅证明方法能覆盖生成器中的四类动力学，不能证明它已完整描述真实 PdCu
器件。接入实测数据后，应分别拟合和比较单指数、双指数、拉伸指数等响应，再按传感器
批次重新训练或校准；若真实动力学超出合成分布，必须扩充生成器，而不能把 8.000%
直接当作真实器件精度。
