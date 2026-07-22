# tcn-refactor

面向部署的气体传感器响应加速与温湿度/基线漂移补偿系统。

项目只保留逐样本（stream）推理路径：`StreamSession.push(sample)` 接收一条
传感器读数，完成基线标定、响应检测、因果特征构造和浓度输出。合成场景生成器
保留完整事件真值，用于训练、回放及端到端验证。

```bash
uv run python main.py generate --output data/scenarios.npz
uv run python main.py train --data data/train.npz --val-data data/val.npz \
  --checkpoint checkpoints/stream.pt --device cuda
uv run python main.py window-eval --data data/test.npz \
  --checkpoint checkpoints/stream.pt --device cuda
uv run python main.py replay --data data/test.npz \
  --checkpoint checkpoints/stream.pt --device cuda
```

训练与部署共享 `FeatureBuilder`；模型没有跨时间归一化层，保证任意时刻的输出
不会依赖未来样本。

## 数据流

训练数据以一个 `Scenario` 为单位保存：每个场景同时包含逐点电压、温度、湿度、
真实基线、真实浓度和每一次注入的起止索引。训练集从事件起点截取窗口，构造形状
`(7, L)` 的特征；DataLoader 把它们拼为 `(B, 7, L)` 后送入模型，模型输出 `(B, L)`。

部署时不使用事件真值。`StreamSession.push(Sample)` 依次完成基线标定、响应检测、
共享特征构造和模型推理；当累积满 15 秒响应窗口时，读取模型最后一个时间步的预测
并锁定，直到检测器判断响应结束。

## 当前实验结果

正式数据采用“一场景一次独立注入”，避免传感器尚未恢复时上一浓度残留与下一
随机标签混合。训练/验证/测试分别使用 800/160/160 个独立随机场景和不同种子。

模型是 7 通道物理残差 TCN：前 6 通道为电压、导数、累计均值、温度、湿度和
响应时间，第 7 通道是一阶响应模型给出的因果浓度粗估；TCN 只学习噪声、漂移、
检测延迟和参数误差造成的残差。

独立测试集结果：

- 对齐窗口：5 秒 MAPE 17.85%，10 秒 6.80%，15 秒 4.24%。
- 逐点端到端回放：160/160 事件检出，平均检测延迟约 2.29 秒。
- 端到端最终 MAE 约 30.5 ppm，MAPE 约 4.68%。

当前推荐部署权重为 `checkpoints/stream_tcn_robust.pt`。数据和权重均被
`.gitignore` 排除，需要在本机生成或训练。
