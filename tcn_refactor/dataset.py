from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .baseline import BaselineEstimator
from .config import SystemConfig
from .detection import ResponseDetector
from .features import build_window
from .synthetic import load_scenarios


@dataclass(frozen=True)
class WindowRecord:
    scenario_index: int
    event_start: int
    event_end: int
    concentration_ppm: float
    baseline: float


class StreamWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """以真实气体事件对齐的流式训练窗口数据集。

    ``__getitem__`` 返回两个张量：特征 ``x`` 的形状为 ``(8, L)``，其中
    ``L=config.window_samples``；标签 ``y`` 是形状 ``()`` 的浓度标量。
    每个 epoch 的随机偏移由 ``epoch + index`` 决定：它既可复现，又会在
    不同 epoch 模拟不同的检测延迟。
    """

    def __init__(self, path: str, *, config: SystemConfig = SystemConfig(),
                 max_offset_s: float = 0.0, seed: int = 0,
                 alignment: str = "event") -> None:
        self.scenarios = load_scenarios(path)
        self.config, self.max_offset_samples, self.seed, self.epoch = config, round(max_offset_s * config.fs_hz), seed, 0
        if alignment == "event":
            self.records = []
            for i, scenario in enumerate(self.scenarios):
                for event in scenario.events:
                    if event.end - event.start < config.window_samples:
                        continue
                    baseline_start = max(0, event.start - config.calibration_samples)
                    baseline = float(np.median(scenario.voltage[baseline_start:event.start]))
                    self.records.append(WindowRecord(
                        i, event.start, event.end, event.concentration_ppm, baseline,
                    ))
        elif alignment == "detection":
            self.records = self._detection_aligned_records()
        else:
            raise ValueError("alignment 必须是 event 或 detection")
        if not self.records:
            raise ValueError("no event is long enough for a response window")

    def _detection_aligned_records(self) -> list[WindowRecord]:
        """复用部署检测器，以因果检测时刻构造训练窗口。"""
        records: list[WindowRecord] = []
        for scenario_index, scenario in enumerate(self.scenarios):
            baseline_estimator = BaselineEstimator(self.config)
            detector = ResponseDetector(self.config)
            previous_responding = False
            for index, voltage in enumerate(scenario.voltage):
                if not baseline_estimator.calibrated:
                    baseline_estimator.update(float(voltage), idle=True)
                    continue
                baseline = baseline_estimator.value
                assert baseline is not None
                responding = detector.update(float(voltage), baseline)
                baseline_estimator.update(float(voltage), idle=not responding)
                if responding and not previous_responding:
                    event = next(
                        (candidate for candidate in scenario.events
                         if candidate.start <= index < candidate.end),
                        None,
                    )
                    if event is not None and index + self.config.window_samples <= len(scenario.voltage):
                        records.append(WindowRecord(
                            scenario_index, index, event.end,
                            event.concentration_ppm, baseline,
                        ))
                previous_responding = responding
        return records

    def set_epoch(self, epoch: int) -> None:
        """由训练循环在每轮开始时调用，以重采样窗口偏移。"""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """构造第 ``index`` 个 ``(x, y)`` 训练样本。"""
        record = self.records[index]
        scenario = self.scenarios[record.scenario_index]
        max_offset = max(0, min(
            self.max_offset_samples,
            record.event_end - record.event_start - self.config.window_samples,
        ))
        offset = int(np.random.default_rng(self.seed + 1_000_003 * self.epoch + index).integers(max_offset + 1)) if max_offset else 0
        start, end = record.event_start + offset, record.event_start + offset + self.config.window_samples
        # baseline 来自事件前观测中位数或部署侧在线估计器，绝不读取生成器保存
        # 的 ``true_baseline`` oracle。
        x = build_window(scenario.voltage[start:end], scenario.temperature_c[start:end],
                         scenario.humidity_rh[start:end], record.baseline, self.config)
        return torch.from_numpy(x), torch.tensor(record.concentration_ppm, dtype=torch.float32)
