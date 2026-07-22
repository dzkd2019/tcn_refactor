"""部署状态机使用的在线基线估计器。

模型输入包含 ``voltage - frozen_baseline``，因此气体响应不能污染该次
预测使用的基线。估计器先显式标定，再仅在检测器判定为空气期时更新 EMA。
"""

from __future__ import annotations

from .config import SystemConfig


class BaselineEstimator:
    """从逐点电压样本估计清洁空气基线。

    参数 ``time_constant_s`` 是完成标定后的 EMA 时间常数。标定期间
    ``update`` 返回 ``None``，调用方不得在此之前启动气体检测。
    """

    def __init__(self, config: SystemConfig, time_constant_s: float = 30.0) -> None:
        self.config = config
        self.alpha = 1.0 / (time_constant_s * config.fs_hz + 1.0)
        self.reset()

    def reset(self) -> None:
        """清除已有状态，例如设备重新上电时调用。"""
        self._count = 0
        self._sum = 0.0
        self.value: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.value is not None

    def update(self, voltage: float, *, idle: bool) -> float | None:
        """输入一个电压样本，并在空气期选择性更新 EMA。

        标定阶段所有样本参与初始均值；标定完成后，只有 ``idle=True`` 的
        样本才会更新基线。
        """
        if self.value is None:
            self._sum += voltage
            self._count += 1
            if self._count == self.config.calibration_samples:
                self.value = self._sum / self._count
            return self.value
        if idle:
            self.value += self.alpha * (voltage - self.value)
        return self.value
