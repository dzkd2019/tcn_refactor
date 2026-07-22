"""核心回归测试。

测试不追求模型精度，而是守住重构最重要的契约：数据无需 pickle、训练与
部署的特征完全一致、模型输出没有未来信息泄漏，以及部署状态机能锁定读数。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parametrize

from tcn_refactor.config import SystemConfig
from tcn_refactor.domain import Sample, SessionState
from tcn_refactor.features import FeatureBuilder, build_window
from tcn_refactor.model import StreamingTCN
from tcn_refactor.session import StreamSession
from tcn_refactor.synthetic import generate_scenarios, load_scenarios, save_scenarios


class CoreContractTests(unittest.TestCase):
    def test_scenario_round_trip_without_pickle(self) -> None:
        """保存后再读取应保留波形、事件起止及浓度标签。"""
        config = SystemConfig(fs_hz=10, response_window_s=2, calibration_s=2)
        scenarios = generate_scenarios(2, config=config, duration_s=100, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.npz"
            save_scenarios(path, scenarios, config)
            loaded = load_scenarios(path)
        self.assertEqual(len(loaded), len(scenarios))
        self.assertTrue(np.array_equal(loaded[0].voltage, scenarios[0].voltage))
        self.assertEqual(loaded[0].events, scenarios[0].events)

    def test_window_and_stream_features_are_identical(self) -> None:
        """批量窗口与逐点 FeatureBuilder 必须产生相同的 (8, L) 特征。"""
        config = SystemConfig(fs_hz=10, response_window_s=1, calibration_s=1)
        voltage = np.asarray([1.0, 1.02, 1.04], dtype=np.float32)
        temperature = np.asarray([25.0, 25.0, 25.0], dtype=np.float32)
        humidity = np.asarray([40.0, 40.0, 40.0], dtype=np.float32)
        batch = build_window(voltage, temperature, humidity, baseline=1.0, config=config)
        builder = FeatureBuilder(config)
        streamed = np.stack([builder.build_one(Sample(float(v), float(t), float(h)), 1.0)
                             for v, t, h in zip(voltage, temperature, humidity)], axis=1)
        np.testing.assert_allclose(batch, streamed, rtol=1e-6, atol=1e-7)

    def test_model_is_causal(self) -> None:
        """修改第 11 个样本之后的数据，前 11 个预测必须严格不变。"""
        torch.manual_seed(3)
        model = StreamingTCN(channels=(8, 8)).eval()
        original = torch.randn(1, 8, 20)
        changed = original.clone()
        changed[:, :, 11:] += 100.0
        with torch.no_grad():
            output_original, output_changed = model(original), model(changed)
        torch.testing.assert_close(output_original[:, :11], output_changed[:, :11], rtol=0, atol=0)

    def test_causal_convolutions_use_weight_norm(self) -> None:
        """每个 TCN 残差块的两层因果卷积都必须使用 weight_norm。"""
        model = StreamingTCN(channels=(8, 8))
        for block in model.backbone:
            self.assertTrue(parametrize.is_parametrized(block.conv1.conv, "weight"))
            self.assertTrue(parametrize.is_parametrized(block.conv2.conv, "weight"))

    def test_session_latches_after_full_response_window(self) -> None:
        """完成标定、检测到响应并积累满窗口后，部署会话应产生锁定读数。"""
        config = SystemConfig(fs_hz=10, response_window_s=1, calibration_s=1)
        session = StreamSession(StreamingTCN(channels=(4,)).eval(), config=config)
        for _ in range(config.calibration_samples + 1):
            session.push(Sample(1.0, 25.0, 40.0))
        results = [session.push(Sample(1.1, 25.0, 40.0)) for _ in range(30)]
        self.assertTrue(any(result.state == SessionState.LATCHED for result in results))
        self.assertTrue(any(result.concentration_ppm is not None for result in results))
