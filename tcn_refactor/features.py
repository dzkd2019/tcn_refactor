from __future__ import annotations

from collections import deque

import numpy as np

from .config import SystemConfig
from .domain import Sample


class FeatureBuilder:
    """训练数据集和在线部署共用的唯一特征构造器。

    每次输入一条 :class:`Sample`，输出 ``(8,)`` 的 ``float32`` 数组：
    ``[电压, 原始导数, 累计平均电压, 温度, 湿度, 已响应时间, 平滑斜率,
    弱稳态响应外推]``。最后一项只作为可被网络忽略的弱先验，不反演浓度。
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
        self._smooth_samples = max(1, round(2.0 * self.config.fs_hz))
        self._smooth_window: deque[float] = deque(maxlen=self._smooth_samples)
        self._smooth_history: deque[float] = deque(
            maxlen=self._smooth_samples + 1
        )

    def build_one(self, sample: Sample, baseline: float) -> np.ndarray:
        """把一条原始样本转为形状 ``(8,)`` 的模型输入特征。"""
        derivative = (
            0.0
            if self._previous_voltage is None
            else (
                (sample.voltage - self._previous_voltage)
                * self.config.fs_hz
                / self.config.dvdt_ref
            )
        )
        self._previous_voltage = sample.voltage
        voltage_feature = (sample.voltage - baseline) / self.config.voltage_ref
        self._delta_sum += voltage_feature
        self._count += 1
        running_mean = self._delta_sum / self._count
        elapsed = min(self._count / self.config.window_samples, 1.0)
        # 先对最近 2 秒响应做因果移动平均，再计算相隔最多 2 秒的斜率。
        # 相比逐点差分，它对电压噪声更鲁棒，又不要求响应符合任何方程。
        self._smooth_window.append(voltage_feature)
        smoothed = float(np.mean(self._smooth_window))
        self._smooth_history.append(smoothed)
        lag = len(self._smooth_history) - 1
        smooth_slope = (
            0.0
            if lag == 0
            else (
                (smoothed - self._smooth_history[0])
                * self.config.voltage_ref
                * self.config.fs_hz
                / lag
                / self.config.dvdt_ref
            )
        )
        steady_proxy = _steady_response_proxy(
            smoothed,
            smooth_slope,
            sample.temperature_c,
            sample.humidity_rh,
            self.config,
        )
        return np.asarray(
            [
                voltage_feature,
                derivative,
                running_mean,
                (sample.temperature_c - self.config.temperature_ref)
                / self.config.temperature_scale,
                (sample.humidity_rh - self.config.humidity_ref)
                / self.config.humidity_scale,
                elapsed,
                smooth_slope,
                steady_proxy,
            ],
            dtype=np.float32,
        )


def build_window(
    voltage: np.ndarray,
    temperature: np.ndarray,
    humidity: np.ndarray,
    baseline: float,
    config: SystemConfig = SystemConfig(),
) -> np.ndarray:
    """向量化构造窗口特征，返回形状 ``(8, 时间长度)`` 的数组。
    原始电压归一化，温度和湿度归一化，电压导数归一化，累计平均电压归一化，已响应时间归一化，
    平滑斜率归一化，弱稳态响应外推归一化。

    这里没有逐点 Python 循环：NumPy 一次完成整段窗口的减法、差分和缩放，
    可显著降低 DataLoader 的 CPU 开销。八个公式与 ``build_one`` 完全相同。
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
    smoothing_samples = max(1, round(2.0 * config.fs_hz))
    # 因果平均后的归一化电压
    smoothed = _causal_moving_average(voltage_feature, window=smoothing_samples)
    lag = smoothing_samples
    smooth_slope = np.zeros_like(smoothed)
    # 前 2 秒使用“当前平滑值 - 第一个平滑值”除以实际间隔；之后固定
    # 使用 2 秒间隔。这与 FeatureBuilder 的在线 deque 实现一致。
    if len(smoothed) > 1:
        early_index = np.arange(1, min(lag, len(smoothed)), dtype=np.float32)
        smooth_slope[1 : min(lag, len(smoothed))] = (
            (smoothed[1 : min(lag, len(smoothed))] - smoothed[0])
            * config.voltage_ref
            * config.fs_hz
            / early_index
            / config.dvdt_ref
        )
    if len(smoothed) > lag:
        smooth_slope[lag:] = (
            (smoothed[lag:] - smoothed[:-lag])
            * config.voltage_ref
            * config.fs_hz
            / lag
            / config.dvdt_ref
        )
    steady_proxy = _steady_response_proxy(
        smoothed, smooth_slope, temperature, humidity, config
    )
    return np.stack(
        [
            voltage_feature,
            derivative,
            running_mean,
            temperature_feature,
            humidity_feature,
            elapsed,
            smooth_slope,
            steady_proxy,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """向量化计算因果移动平均，第 t 点只使用不晚于 t 的数据。"""
    cumulative = np.cumsum(values, dtype=np.float64)
    result = cumulative.copy()
    result[window:] = cumulative[window:] - cumulative[:-window]
    denominator = np.minimum(np.arange(1, len(values) + 1), window)
    return (result / denominator).astype(np.float32)


def _steady_response_proxy(
    smoothed_response: np.ndarray | float,
    smooth_slope: np.ndarray | float,
    temperature_c: np.ndarray | float,
    humidity_rh: np.ndarray | float,
    config: SystemConfig,
) -> np.ndarray | float:
    """计算 ``y_f + tau*dy_f/dt`` 弱稳态响应先验。

    该特征不使用浓度标定曲线，也不执行 ``F^{-1}``，因此不会把“一阶反演
    浓度”强加给模型。若 PdCu 呈双指数、拉伸指数或延迟响应，TCN 可根据其他
    七个通道学习降低该先验的权重。裁剪仅用于限制异常导数的数值范围。
    """
    temperature = np.asarray(temperature_c, dtype=np.float32)
    humidity = np.asarray(humidity_rh, dtype=np.float32)
    # 该任务使用同一支已标定传感器；生成器中的环境修正相对于暴露前环境，
    # 而不是温湿度绝对值。窗口内缺少这个参考点，因此使用标定得到的 30 秒
    # 名义时间常数比基于绝对温湿度进行错误放大更稳健。
    nominal_tau = 30.0 + np.zeros_like(temperature + humidity)
    # smooth_slope 的归一化单位为 (V/s)/DVDT_REF，需要换回归一化响应每秒。
    normalized_slope = (
        np.asarray(smooth_slope) * config.dvdt_ref / config.voltage_ref
    )
    estimate = np.clip(
        np.asarray(smoothed_response) + nominal_tau * normalized_slope,
        -0.5,
        2.0,
    )
    return (
        float(estimate) if estimate.ndim == 0 else estimate.astype(np.float32)
    )
