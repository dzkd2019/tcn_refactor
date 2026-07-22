from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import SystemConfig
from .domain import GasEvent, Scenario


def generate_scenarios(
    count: int, *, config: SystemConfig = SystemConfig(), duration_s: float = 180.0,
    seed: int = 2026, events_per_scenario: int = 1,
) -> list[Scenario]:
    """生成可复现、带完整标签的传感器场景。

    每个场景中的温度、湿度、基线漂移、响应时间及灵敏度均会变化；每段
    气体暴露的起止位置和浓度都显式记录在 :class:`GasEvent` 中，避免旧版
    数据“有注入起点、却没有可靠结束位置”的问题。
    """
    rng = np.random.default_rng(seed)
    length = round(duration_s * config.fs_hz)
    scenarios: list[Scenario] = []
    for index in range(count):
        scenario_seed = int(rng.integers(0, 2**31 - 1))
        local = np.random.default_rng(scenario_seed)
        time = np.arange(length) / config.fs_hz
        t0, rh0 = local.uniform(20, 80), local.uniform(0, 80)
        temperature = t0 + 1.5 * np.sin(2 * np.pi * time / duration_s)
        humidity = rh0 + 2.0 * np.sin(2 * np.pi * time / (duration_s * 0.7) + 0.5)

        base_nominal = 1.0 + 0.001 * (temperature - 20) - 0.0008 * humidity
        drift = np.cumsum(local.normal(0.0, 2e-5, length))
        drift -= drift.mean()
        drift *= 0.01 / max(float(np.ptp(drift)), 1e-8)
        baseline = base_nominal + drift

        target = baseline.copy()
        concentration = np.zeros(length, dtype=np.float32)
        events: list[GasEvent] = []
        cursor = max(config.calibration_samples + 100, round(25 * config.fs_hz))
        for _ in range(events_per_scenario):
            on = int(local.integers(35, 56) * config.fs_hz)
            off = int(local.integers(20, 36) * config.fs_hz)
            start, end = cursor, min(cursor + on, length)
            if end - start < config.window_samples or end >= length:
                break
            ppm = float(local.uniform(150, 1100))
            sensitivity = 0.0015 * (1 + 0.005 * (temperature[start] - 20)) * (
                1 - 0.006 * humidity[start]
            )
            amplitude = sensitivity * ppm**0.85
            target[start:end] = baseline[start:end] + amplitude
            concentration[start:end] = ppm
            events.append(GasEvent(start, end, ppm))
            cursor = end + off

        voltage = np.empty(length, dtype=np.float32)
        voltage[0] = baseline[0]
        for i in range(1, length):
            tau = 30.0 * (1 + 0.020 * humidity[i]) / np.exp(0.012 * (temperature[i] - 20))
            voltage[i] = voltage[i - 1] + (target[i - 1] - voltage[i - 1]) / (tau * config.fs_hz)
        voltage += local.normal(0.0, 0.004, length).astype(np.float32)
        scenarios.append(Scenario(voltage, temperature.astype(np.float32), humidity.astype(np.float32),
                                  baseline.astype(np.float32), concentration, tuple(events), scenario_seed))
    return scenarios


def save_scenarios(path: str | Path, scenarios: list[Scenario], config: SystemConfig) -> None:
    """仅保存数值数组；读取时不需要、也不允许使用 pickle。"""
    if not scenarios:
        raise ValueError("至少需要一个场景才能保存数据集")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n, length = len(scenarios), len(scenarios[0].voltage)
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
                        seeds=np.asarray([s.seed for s in scenarios]), config=np.asarray([asdict(config)], dtype=str))


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
        result: list[Scenario] = []
        for i in range(len(voltage)):
            events = tuple(GasEvent(int(start), int(end), float(ppm)) for start, end, ppm in zip(
                event_start[i], event_end[i], event_ppm[i]) if start >= 0)
            result.append(Scenario(voltage[i], temperature[i], humidity[i], baseline[i],
                                   concentration[i], events, int(seeds[i])))
    return result
