from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    """数据生成、模型训练和部署推理共用的可序列化配置。"""

    fs_hz: int = 100  # 采样率，单位 Hz（每秒样本数）
    response_window_s: float = 15.0
    calibration_s: float = 20.0
    voltage_ref: float = 0.5
    dvdt_ref: float = 0.05
    temperature_ref: float = 50.0
    temperature_scale: float = 30.0
    humidity_ref: float = 40.0
    humidity_scale: float = 40.0
    concentration_scale: float = 1000.0

    @property
    def window_samples(self) -> int:
        return round(self.fs_hz * self.response_window_s)

    @property
    def calibration_samples(self) -> int:
        return round(self.fs_hz * self.calibration_s)
