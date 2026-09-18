`SessionState` 描述的是一次流式部署会话的运行阶段，定义在 [domain.py](/home/mrawa/code/graduate/tcn_refactor/tcn_refactor/domain.py:9)，状态机主要实现于 [session.py](/home/mrawa/code/graduate/tcn_refactor/tcn_refactor/session.py:36)。

### 1. `CALIBRATING`：初始基线标定

设备刚启动或调用 `session.reset()` 后进入该状态。

此时系统：

- 收集清洁空气样本；
- 使用约 `calibration_s` 秒的数据估计初始基线；
- 不启动响应检测；
- 不构造模型特征；
- 不输出浓度。

默认配置下，采样率为 100 Hz、标定时间为 20 秒，因此需要先收集约 2000 个样本。

标定期间返回：

```python
Prediction(
    state=SessionState.CALIBRATING,
    concentration_ppm=None,
    response_samples=0,
    detected_event_start=None,
)
```

### 2. `IDLE`：已标定、当前没有气体响应

基线标定完成后，如果检测器没有发现响应，状态进入 `IDLE`。

此时系统：

- 持续监测电压是否偏离基线；
- 在空气期缓慢更新基线 EMA；
- 不积累响应窗口；
- 不运行 TCN；
- 不输出浓度。

这是正常的待机状态。

### 3. `RESPONDING`：检测到响应，正在积累窗口

当 `ResponseDetector` 判定出现气体响应时，状态从 `IDLE` 进入 `RESPONDING`。

进入该状态时会：

- 记录检测器触发的样本索引；
- 冻结当前基线；
- 清空旧的特征缓冲区；
- 重置 `FeatureBuilder`；
- 开始逐点构造 8 通道特征。

此时模型还不会立即输出浓度，而是先收集完整的 15 秒响应窗口。

如果响应在窗口收满之前结束：

```text
RESPONDING → IDLE
```

窗口会被清空，本次事件不会产生最终浓度读数。

### 4. `LATCHED`：窗口完成，浓度已锁定

当 `RESPONDING` 状态下积累满完整窗口，例如 1500 个样本后：

1. 将特征堆叠为 `(1, 8, 1500)`；
2. 执行一次 TCN 推理；
3. 取最后一个时间步的预测；
4. 保存到 `_latched`；
5. 状态切换为 `LATCHED`。

之后在同一响应持续期间，即使继续输入样本，也会重复返回同一个锁定值，而不会反复改变浓度结果。

这对应部署逻辑：

> 检测到响应后等待 15 秒，输出一次最终浓度，并保持该读数。

### 状态转移图

```text
                    reset()
                       │
                       ▼
                CALIBRATING
                 │         │
     标定完成且无响应       │ 检测器触发
                 ▼         │
                IDLE ───────┘
                 │
                 │ 检测器触发
                 ▼
             RESPONDING
              │       │
   响应提前结束       │ 收满 15 秒窗口
              ▼       ▼
             IDLE   LATCHED
                       │
                       │ 检测器确认响应结束
                       ▼
                      IDLE
```

其中检测器不是单点触发，而是带有持续确认和释放延迟：

- 偏离基线达到约 `8 mV`，并持续约 `0.4 秒`，才进入响应；
- 偏离降到约 `5 mV` 以下，并持续约 `4 秒`，才释放响应。

因此实际转移为：

```text
IDLE → RESPONDING
```

表示检测器确认了一个新响应，而不是单个噪声尖峰。

```text
RESPONDING/LATCHED → IDLE
```

表示响应已经结束，系统清空本次事件状态并准备检测下一次响应。

另外，进入 `RESPONDING` 后基线会被冻结，直到回到 `IDLE`；这样气体响应不会被错误地吸收到基线中。
