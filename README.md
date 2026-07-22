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
`(8, L)` 的特征；DataLoader 把它们拼为 `(B, 8, L)` 后送入模型，模型输出 `(B, L)`。

8 个输入通道依次为：归一化响应电压、原始电压导数、响应累计均值、温度、湿度、
响应持续时间、0.5 秒因果平滑斜率、弱稳态响应外推。最后一个通道使用
`y_f + tau * dy_f/dt`，但不执行浓度标定曲线的逆变换，也不直接作为模型输出；
它只是网络可以忽略或修正的弱先验。这样既利用了传感器响应通常具有惯性的事实，
又不会强制假设 PdCu 合金传感器严格服从单一一阶方程。

部署时不使用事件真值。`StreamSession.push(Sample)` 依次完成基线标定、响应检测、
共享特征构造和模型推理；当累积满 15 秒响应窗口时，读取模型最后一个时间步的预测
并锁定，直到检测器判断响应结束。

## 当前实验结果

正式数据采用“一场景一次独立注入”，避免传感器尚未恢复时上一浓度残留与下一
随机标签混合。训练/验证/测试分别使用 1600/320/320 个独立随机场景和不同种子。
每个事件随机选择单指数、双指数、拉伸指数或带纯延迟的响应；时间常数允许随浓度
变化，恢复过程另用独立时间常数，从数据层面消除“所有器件必为单一一阶系统”的
错误先验。

模型是 8 通道直接回归 TCN。它不再以一阶方程的浓度反演值为输出基线，而是根据
完整因果序列直接预测浓度。每个残差块的两层因果卷积均使用
`torch.nn.utils.parametrizations.weight_norm`，将卷积核的方向和大小解耦优化。
训练使用相对 Huber 损失：误差大于 5% 时近似 MAPE，小误差区间使用平滑二次项。

独立测试集结果：

- 对齐窗口：5 秒 MAPE 29.90%，10 秒 15.33%，15 秒 10.67%。这是更困难的
  “从真实事件起点计时”指标，包含随机多动力学造成的不可辨识性。
- 逐点端到端回放：320/320 事件检出，平均检测延迟 2.131 秒。
- 部署路径最终 MAE 53.646 ppm，MAPE 9.587%。部署计时从检测器实际触发点开始，
  与 `StreamSession` 锁定读数的真实使用方式一致。

当前推荐部署权重为 `checkpoints/stream_tcn_weak_prior_huber.pt`。数据和权重均被
`.gitignore` 排除，需要在本机生成或训练。

上述结果仅证明方法能覆盖生成器中的四类动力学，不能证明它已完整描述真实 PdCu
器件。接入实测数据后，应分别拟合和比较单指数、双指数、拉伸指数等响应，再按传感器
批次重新训练或校准；若真实动力学超出合成分布，必须扩充生成器，而不能把 9.587%
直接当作真实器件精度。
