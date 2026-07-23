"""公开部署 API：输入一条原始样本，输出一条结构化预测结果。"""

from __future__ import annotations

from collections import deque

import torch

from .baseline import BaselineEstimator
from .config import SystemConfig
from .detection import ResponseDetector
from .domain import Prediction, Sample, SessionState
from .features import FeatureBuilder
from .model import StreamingTCN


class StreamSession:
    """围绕 :class:`StreamingTCN` 的有状态、仅流式推理封装。

    每次 ``push`` 的数据流为：
    ``Sample -> 基线 -> 检测器 -> FeatureBuilder -> 特征缓冲 -> 模型``。
    缓冲区内每个特征为 ``(8,)``，堆叠后为 ``(T, 8)``；在送入 PyTorch
    前转置并增加 batch 维，成为 ``(1, 8, T)``。模型输出形状为 ``(1, T)``。
    """

    def __init__(self, model: StreamingTCN, *, config: SystemConfig = SystemConfig(),
                 device: torch.device | str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.config, self.device = config, torch.device(device)
        self.baseline = BaselineEstimator(config)
        self.detector = ResponseDetector(config)
        self.features = FeatureBuilder(config)
        self._buffer: deque[torch.Tensor] = deque(maxlen=config.window_samples)
        self.reset()

    def reset(self) -> None:
        """重置全部部署状态，并重新开始清洁空气标定。"""
        self.baseline.reset()
        self.detector.reset()
        self.features.reset()
        self._buffer.clear()
        self.state = SessionState.CALIBRATING
        self._frozen_baseline: float | None = None
        self._event_start: int | None = None
        self._sample_index = 0
        self._latched: float | None = None

    def push(self, sample: Sample) -> Prediction:
        """输入一条原始样本，返回当前部署状态和可选浓度。

        只有收集满完整响应窗口才产生浓度；随后读数被锁定，直至检测器释放
        当前响应。这样模型始终工作在训练过的 15 秒窗口分布内。
        """
        index = self._sample_index
        self._sample_index += 1
        if not self.baseline.calibrated:
            self.baseline.update(sample.voltage, idle=True)
            return Prediction(SessionState.CALIBRATING, None, 0, None)

        # 检测器使用当前的清洁空气基线；随后仅在空气期更新基线估计器。
        baseline = self.baseline.value
        assert baseline is not None
        responding = self.detector.update(sample.voltage, baseline)
        self.baseline.update(sample.voltage, idle=not responding)

        if responding and self.state in (SessionState.IDLE, SessionState.CALIBRATING):
            self.state = SessionState.RESPONDING
            self._event_start, self._frozen_baseline = index, baseline
            self._latched = None
            self._buffer.clear()
            self.features.reset()
        elif not responding and self.state in (SessionState.RESPONDING, SessionState.LATCHED):
            self.state = SessionState.IDLE
            self._buffer.clear()
            self.features.reset()
            self._event_start, self._frozen_baseline, self._latched = None, None, None

        if self.state == SessionState.IDLE:
            return Prediction(self.state, None, 0, None)
        if self.state == SessionState.CALIBRATING:
            self.state = SessionState.IDLE
            return Prediction(self.state, None, 0, None)

        assert self._frozen_baseline is not None
        self._buffer.append(torch.from_numpy(self.features.build_one(sample, self._frozen_baseline)))
        response_samples = len(self._buffer)
        if response_samples < self.config.window_samples:
            return Prediction(self.state, None, response_samples, self._event_start)

        if self._latched is None:
            # stack 后为 (T, 8)，转置、增加 batch 维后为模型输入 (1, 8, T)。
            # 右端元素是当前时刻的因果预测。
            x = torch.stack(tuple(self._buffer)).transpose(0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self._latched = float(self.model(x)[0, -1].cpu())
            self.state = SessionState.LATCHED
        return Prediction(self.state, self._latched, response_samples, self._event_start)
