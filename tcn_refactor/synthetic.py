from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import SystemConfig
from .domain import GasEvent, Scenario

SCHEMA_VERSION = 2
GENERATOR_VERSION = "calibrated_stateful_v5"


def generate_scenarios(
    count: int, *, config: SystemConfig = SystemConfig(), duration_s: float = 180.0,
    seed: int = 2026, events_per_scenario: int = 1,
) -> list[Scenario]:
    """生成带气路输运、器件差异、状态记忆和相关噪声的传感器场景。

    ``GasEvent`` 记录阀门命令的起止与设定浓度，逐点浓度则表示经过管路延迟
    和气室混合后实际到达传感器表面的浓度。所有响应由同一个连续器件状态
    演化，不再把每次暴露的独立波形直接相加。
    """
    rng = np.random.default_rng(seed)
    length = round(duration_s * config.fs_hz)
    scenarios: list[Scenario] = []
    dt = 1.0 / config.fs_hz
    for _ in range(count):
        scenario_seed = int(rng.integers(0, 2**31 - 1))
        local = np.random.default_rng(scenario_seed)
        time = np.arange(length) / config.fs_hz

        # 温度由缓慢环境变化和相关扰动组成；湿度通过近似恒定绝对含水量与
        # 温度反向耦合，再叠加独立的通风变化，避免两个无关正弦信号。
        t0, rh0 = local.uniform(15.0, 65.0), local.uniform(10.0, 85.0)
        temperature_true = t0 + local.uniform(0.5, 2.5) * np.sin(
            2 * np.pi * time / local.uniform(0.8, 1.8) / duration_s + local.uniform(0, 2 * np.pi)
        )
        temperature_true += np.cumsum(local.normal(0.0, 0.002 * np.sqrt(dt), length))
        humidity_true = rh0 * np.exp(-0.035 * (temperature_true - t0))
        humidity_true += local.uniform(0.5, 3.0) * np.sin(
            2 * np.pi * time / local.uniform(0.5, 1.3) / duration_s + local.uniform(0, 2 * np.pi)
        )
        humidity_true = np.clip(humidity_true, 0.0, 100.0)

        # 先构造阀门命令。首次暴露在标定结束后的可用区间内随机出现，后续
        # 暴露之间的空气间隔也随机，从而消除固定 25 秒位置特征。
        command = np.zeros(length, dtype=np.float64)
        events: list[GasEvent] = []
        first_min = config.calibration_samples + round(2.0 * config.fs_hz)
        first_max = max(first_min + 1, length - config.window_samples - round(10.0 * config.fs_hz))
        cursor = int(local.integers(first_min, first_max)) if first_min < length else length
        for _ in range(events_per_scenario):
            duration_class = local.choice(("short", "normal", "long"), p=(0.12, 0.68, 0.20))
            if duration_class == "short":
                on_s = local.uniform(0.5 * config.response_window_s, 0.95 * config.response_window_s)
            elif duration_class == "normal":
                on_s = local.uniform(max(20.0, config.response_window_s), 55.0)
            else:
                on_s = local.uniform(55.0, 90.0)
            start, end = cursor, min(cursor + round(on_s * config.fs_hz), length)
            if start >= length or end <= start:
                break
            calibration_levels = np.asarray((100, 200, 400, 600, 800, 1000, 1200), dtype=np.float64)
            ppm = float(local.choice(calibration_levels) * local.uniform(0.97, 1.03))
            command[start:end] = ppm
            events.append(GasEvent(start, end, ppm))
            cursor = end + round(local.uniform(5.0, 45.0) * config.fs_hz)

        # 阀门命令先经过管路纯延迟和气室混合，得到传感器表面真实浓度。
        delay_samples = round(local.uniform(0.3, 0.8) * config.fs_hz)
        delayed_command = np.zeros_like(command)
        if delay_samples < length:
            delayed_command[delay_samples:] = command[:length - delay_samples]
        chamber_tau = local.uniform(0.8, 1.4)
        concentration = np.zeros(length, dtype=np.float64)
        chamber_alpha = 1.0 - np.exp(-dt / chamber_tau)
        for i in range(1, length):
            concentration[i] = concentration[i - 1] + chamber_alpha * (
                delayed_command[i] - concentration[i - 1]
            )

        # 以下参数在整个场景内固定，代表一个虚拟器件；事件之间只通过连续
        # 状态和环境变化产生差异。平方根 Langmuir 形式提供有限饱和值。
        # 默认任务针对同一已标定器件批次，场景间只保留重复性误差。跨器件
        # 泛化需要显式设备标定信息，不能把灵敏度作为不可观测随机变量。
        max_response = 0.52 * np.exp(local.normal(0.0, 0.003))
        langmuir_k = 0.045 * np.exp(local.normal(0.0, 0.003))
        sensitivity_temp = local.normal(0.004, 0.0002)
        sensitivity_rh = local.normal(-0.004, 0.0002)
        base_tau = 30.0 * np.exp(local.normal(0.0, 0.01))
        tau_temp = local.normal(-0.012, 0.0004)
        tau_rh = local.normal(0.012, 0.0005)
        concentration_exponent = local.normal(-0.06, 0.005)
        slow_mix = float(np.clip(local.normal(0.30, 0.01), 0.25, 0.35))
        slow_ratio = float(np.clip(local.normal(3.0, 0.05), 2.8, 3.2))
        recovery_ratio = float(np.clip(local.normal(1.5, 0.05), 1.3, 1.7))
        fast_state = slow_state = 0.0
        response = np.zeros(length, dtype=np.float64)
        for i in range(1, length):
            root_concentration = np.sqrt(max(concentration[i], 0.0))
            saturation = langmuir_k * root_concentration / (1.0 + langmuir_k * root_concentration)
            environment_scale = np.clip(
                1.0 + sensitivity_temp * (temperature_true[i] - t0)
                + sensitivity_rh * (humidity_true[i] - rh0),
                0.35, 1.65,
            )
            target = max_response * environment_scale * saturation
            tau = base_tau * np.exp(tau_temp * (temperature_true[i] - t0)) * (
                1.0 + tau_rh * max(0.0, humidity_true[i] - rh0)
            )
            if concentration[i] > 1.0:
                tau *= (concentration[i] / 600.0) ** concentration_exponent
            else:
                tau *= recovery_ratio
            tau = float(np.clip(tau, 3.0, 180.0))
            fast_state += (1.0 - np.exp(-dt / tau)) * (target - fast_state)
            slow_state += (1.0 - np.exp(-dt / (tau * slow_ratio))) * (target - slow_state)
            response[i] = (1.0 - slow_mix) * fast_state + slow_mix * slow_state

        # 器件级温湿度系数、交互项和自然扩散漂移不再被强制归一到固定峰峰值。
        baseline_offset = local.uniform(0.90, 1.10)
        baseline_temp = local.normal(0.0010, 0.00005)
        baseline_rh = local.normal(-0.0008, 0.00004)
        baseline_interaction = local.normal(0.0, 2e-6)
        drift_sigma = local.uniform(1e-5, 4e-5)
        drift = np.cumsum(local.normal(0.0, drift_sigma * np.sqrt(dt), length))
        aging_slope = local.normal(0.0, 3e-6)
        exposure_shift = local.normal(0.0, 1.5e-6) * np.cumsum(concentration / 1000.0) * dt
        baseline = (
            baseline_offset
            + baseline_temp * (temperature_true - t0)
            + baseline_rh * (humidity_true - rh0)
            + baseline_interaction * (temperature_true - t0) * (humidity_true - rh0)
            + drift + aging_slope * time + exposure_shift
        )

        # AR 低频噪声、异方差白噪声、偶发尖峰和 ADC 量化共同构成电压噪声。
        ar_rho = local.uniform(0.92, 0.995)
        ar_sigma = local.uniform(0.0002, 0.0008)
        correlated_noise = np.zeros(length, dtype=np.float64)
        innovations = local.normal(0.0, ar_sigma * np.sqrt(1.0 - ar_rho**2), length)
        for i in range(1, length):
            correlated_noise[i] = ar_rho * correlated_noise[i - 1] + innovations[i]
        white_sigma = local.uniform(0.0010, 0.0025) * (1.0 + 0.8 * response)
        noise = correlated_noise + local.normal(0.0, white_sigma)
        spike_mask = local.random(length) < local.uniform(2e-5, 2e-4)
        noise[spike_mask] += local.normal(0.0, 0.020, int(spike_mask.sum()))
        adc_step = local.choice((0.0002, 0.0005, 0.0010))
        voltage = np.round((baseline + response + noise) / adc_step) * adc_step

        temperature = temperature_true + local.normal(0.0, local.uniform(0.03, 0.15), length)
        humidity = np.clip(
            humidity_true + local.normal(0.0, local.uniform(0.10, 0.50), length), 0.0, 100.0
        )
        scenarios.append(Scenario(
            voltage.astype(np.float32), temperature.astype(np.float32), humidity.astype(np.float32),
            baseline.astype(np.float32), concentration.astype(np.float32), tuple(events), scenario_seed,
        ))
    return scenarios


def save_scenarios(path: str | Path, scenarios: list[Scenario], config: SystemConfig) -> None:
    """仅保存数值数组；读取时不需要、也不允许使用 pickle。"""
    if not scenarios:
        raise ValueError("至少需要一个场景才能保存数据集")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = len(scenarios)
    max_events = max(len(s.events) for s in scenarios)
    event_start = np.full((n, max_events), -1, dtype=np.int32)
    event_end = np.full_like(event_start, -1)
    # 事件标签属于评估真值，使用 float64 保存，避免写入/读取后浓度发生变化。
    event_ppm = np.full((n, max_events), np.nan, dtype=np.float64)
    for i, scenario in enumerate(scenarios):
        for j, event in enumerate(scenario.events):
            event_start[i, j], event_end[i, j], event_ppm[i, j] = event.start, event.end, event.concentration_ppm
    np.savez_compressed(target, voltage=np.stack([s.voltage for s in scenarios]),
                        temperature=np.stack([s.temperature_c for s in scenarios]),
                        humidity=np.stack([s.humidity_rh for s in scenarios]),
                        baseline=np.stack([s.true_baseline for s in scenarios]),
                        concentration=np.stack([s.concentration_ppm for s in scenarios]),
                        event_start=event_start, event_end=event_end, event_ppm=event_ppm,
                        seeds=np.asarray([s.seed for s in scenarios]),
                        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
                        generator_version=np.asarray(GENERATOR_VERSION),
                        config_json=np.asarray(json.dumps(asdict(config), ensure_ascii=False)))


def load_scenarios(path: str | Path) -> list[Scenario]:
    """从 NPZ 读取场景，并保证每个压缩数组只解压一次。

    ``np.load`` 返回的 NPZ 对象按键惰性解压。如果在场景循环中反复写
    ``data["voltage"]``，就会反复解压整块二维数组，数据量大时极其缓慢。
    因此先把各字段缓存到局部变量，再按第一维切分场景。
    """
    with np.load(path, allow_pickle=False) as data:
        voltage = data["voltage"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        baseline = data["baseline"]
        concentration = data["concentration"]
        event_start = data["event_start"]
        event_end = data["event_end"]
        event_ppm = data["event_ppm"]
        seeds = data["seeds"]
        sequence_shape = voltage.shape
        sequence_arrays = (temperature, humidity, baseline, concentration)
        if voltage.ndim != 2 or any(array.shape != sequence_shape for array in sequence_arrays):
            raise ValueError("场景数组必须具有一致的二维形状")
        if (
            len(seeds) != len(voltage)
            or event_start.shape != event_end.shape
            or event_start.shape != event_ppm.shape
            or event_start.shape[0] != len(voltage)
        ):
            raise ValueError("场景、事件或随机种子数量不一致")
        if any(not np.isfinite(array).all() for array in (voltage, *sequence_arrays)):
            raise ValueError("场景数组包含 NaN 或 Inf")
        result: list[Scenario] = []
        for i in range(len(voltage)):
            for start, end, ppm in zip(event_start[i], event_end[i], event_ppm[i]):
                if start >= 0 and not (start < end <= sequence_shape[1] and np.isfinite(ppm) and ppm > 0):
                    raise ValueError("事件标签的范围或浓度非法")
            events = tuple(GasEvent(int(start), int(end), float(ppm)) for start, end, ppm in zip(
                event_start[i], event_end[i], event_ppm[i]) if start >= 0)
            result.append(Scenario(voltage[i], temperature[i], humidity[i], baseline[i],
                                   concentration[i], events, int(seeds[i])))
    return result
