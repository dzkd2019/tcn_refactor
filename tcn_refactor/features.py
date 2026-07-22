from __future__ import annotations

import numpy as np

from .config import SystemConfig
from .domain import Sample


class FeatureBuilder:
    """训练数据集和在线部署共用的唯一特征构造器。

    每次输入一条 :class:`Sample`，输出 ``(7,)`` 的 ``float32`` 数组：
    ``[电压, 导数, 累计平均电压, 温度, 湿度, 已响应时间, 物理粗估]``。
    推理不会各自复制一套归一化公式而发生分布不一致。
    """

    def __init__(self, config: SystemConfig = SystemConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """清除上一时刻电压，使下一条样本的导数为 0。"""
        self._previous_voltage = None
        self._delta_sum = 0.0
        self._count = 0

    def build_one(self, sample: Sample, baseline: float) -> np.ndarray:
        """把一条原始样本转为形状 ``(7,)`` 的模型输入特征。"""
        derivative = 0.0 if self._previous_voltage is None else (
            (sample.voltage - self._previous_voltage) * self.config.fs_hz / self.config.dvdt_ref
        )
        self._previous_voltage = sample.voltage
        voltage_feature = (sample.voltage - baseline) / self.config.voltage_ref
        self._delta_sum += voltage_feature
        self._count += 1
        running_mean = self._delta_sum / self._count
        elapsed = min(self._count / self.config.window_samples, 1.0)
        proxy = _physics_proxy(
            running_mean, sample.temperature_c, sample.humidity_rh,
            self._count / self.config.fs_hz, self.config,
        )
        return np.asarray([
            voltage_feature,
            derivative,
            running_mean,
            (sample.temperature_c - self.config.temperature_ref) / self.config.temperature_scale,
            (sample.humidity_rh - self.config.humidity_ref) / self.config.humidity_scale,
            elapsed,
            proxy,
        ], dtype=np.float32)


def build_window(voltage: np.ndarray, temperature: np.ndarray, humidity: np.ndarray,
                 baseline: float, config: SystemConfig = SystemConfig()) -> np.ndarray:
    """向量化构造窗口特征，返回形状 ``(7, 时间长度)`` 的数组。

    这里没有逐点 Python 循环：NumPy 一次完成整段窗口的减法、差分和缩放，
    可显著降低 DataLoader 的 CPU 开销。六个公式与 ``build_one`` 完全相同。
    """
    voltage = np.asarray(voltage, dtype=np.float32)
    temperature = np.asarray(temperature, dtype=np.float32)
    humidity = np.asarray(humidity, dtype=np.float32)
    voltage_feature = (voltage - baseline) / config.voltage_ref
    derivative = np.empty_like(voltage)
    derivative[0] = 0.0
    derivative[1:] = np.diff(voltage) * config.fs_hz / config.dvdt_ref
    temperature_feature = (
        temperature - config.temperature_ref
    ) / config.temperature_scale
    humidity_feature = (humidity - config.humidity_ref) / config.humidity_scale
    sample_count = np.arange(1, len(voltage) + 1, dtype=np.float32)
    running_mean = np.cumsum(voltage_feature, dtype=np.float32) / sample_count
    elapsed = np.minimum(sample_count / config.window_samples, 1.0)
    proxy = _physics_proxy(running_mean, temperature, humidity,
                           sample_count / config.fs_hz, config)
    return np.stack(
        [voltage_feature, derivative, running_mean, temperature_feature,
         humidity_feature, elapsed, proxy], axis=0
    ).astype(np.float32, copy=False)


def _physics_proxy(running_mean_normalized: np.ndarray | float,
                   temperature_c: np.ndarray | float,
                   humidity_rh: np.ndarray | float,
                   elapsed_s: np.ndarray | float,
                   config: SystemConfig) -> np.ndarray | float:
    """根据一阶响应方程计算无量纲粗浓度估计。

    对 ``V(t)=A*(1-exp(-t/tau))`` 在 ``[0,t]`` 上求平均，可得平均响应因子
    ``1 - tau/t*(1-exp(-t/tau))``。由累计平均电压反推出稳态幅值 A，再用
    温湿度修正后的灵敏度反推浓度。结果裁剪到 0~1.5 倍量程，TCN 负责学习
    检测延迟、噪声、漂移以及物理参数误差的修正。
    """
    elapsed = np.maximum(np.asarray(elapsed_s, dtype=np.float32), 1e-3)
    temperature = np.asarray(temperature_c, dtype=np.float32)
    humidity = np.asarray(humidity_rh, dtype=np.float32)
    tau = 30.0 * (1.0 + 0.020 * humidity) / np.exp(0.012 * (temperature - 20.0))
    mean_factor = 1.0 - tau / elapsed * (1.0 - np.exp(-elapsed / tau))
    mean_factor = np.maximum(mean_factor, 1e-4)
    amplitude = np.maximum(
        np.asarray(running_mean_normalized) * config.voltage_ref / mean_factor, 0.0
    )
    sensitivity = 0.0015 * (1.0 + 0.005 * (temperature - 20.0)) * (
        1.0 - 0.006 * humidity
    )
    concentration = np.power(amplitude / np.maximum(sensitivity, 1e-6), 1.0 / 0.85)
    normalized = np.clip(concentration / 1200.0, 0.0, 1.5)
    return float(normalized) if normalized.ndim == 0 else normalized.astype(np.float32)
