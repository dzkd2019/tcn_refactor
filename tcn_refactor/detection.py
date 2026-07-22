"""用于决定何时开始、何时结束响应窗口的小型因果检测器。"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import SystemConfig


class ResponseDetector:
    """带平滑、持续确认和延迟释放的阈值检测器。

    它不依赖神经网络；因此误报、漏检和检测延迟可以独立测试，而无需加载
    PyTorch 模型。
    """

    def __init__(self, config: SystemConfig, trigger_voltage: float = 0.008,
                 release_voltage: float = 0.005, persist_s: float = 0.4,
                 release_s: float = 4.0) -> None:
        self.config = config
        self.trigger_voltage = trigger_voltage
        self.release_voltage = release_voltage
        self.persist_samples = max(1, round(persist_s * config.fs_hz))
        self.release_samples = max(1, round(release_s * config.fs_hz))
        self._smooth: deque[float] = deque(maxlen=max(1, round(0.8 * config.fs_hz)))
        self.reset()

    def reset(self) -> None:
        self.responding = False
        self._trigger_count = 0
        self._release_count = 0
        self._smooth.clear()

    def update(self, voltage: float, baseline: float) -> bool:
        """处理一个样本并返回当前是否处于响应状态。"""
        self._smooth.append(voltage)
        deviation = float(np.mean(self._smooth)) - baseline
        if not self.responding:
            self._trigger_count = self._trigger_count + 1 if abs(deviation) >= self.trigger_voltage else 0
            if self._trigger_count >= self.persist_samples:
                self.responding = True
                self._release_count = 0
        else:
            self._release_count = self._release_count + 1 if abs(deviation) <= self.release_voltage else 0
            if self._release_count >= self.release_samples:
                self.responding = False
                self._trigger_count = 0
        return self.responding
