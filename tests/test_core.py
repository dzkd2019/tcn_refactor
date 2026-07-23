"""核心回归测试。

测试不追求模型精度，而是守住重构最重要的契约：数据无需 pickle、训练与
部署的特征完全一致、模型输出没有未来信息泄漏，以及部署状态机能锁定读数。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parametrize

from tcn_refactor.config import SystemConfig
from tcn_refactor.dataset import StreamWindowDataset
from tcn_refactor.domain import GasEvent, Sample, Scenario, SessionState
from tcn_refactor.features import FeatureBuilder, build_window
from tcn_refactor.model import StreamingTCN
from tcn_refactor.session import StreamSession
from tcn_refactor.synthetic import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    generate_scenarios,
    load_scenarios,
    save_scenarios,
)
from tcn_refactor.training import (
    configure_cuda_for_causal_tcn,
    cosine_learning_rate,
    weighted_stream_loss,
)


class CoreContractTests(unittest.TestCase):
    def test_synthetic_scenarios_use_random_event_times_and_chamber_dynamics(self) -> None:
        """事件位置应随机，表面浓度应有气路延迟且湿度必须保持物理边界。"""
        config = SystemConfig(fs_hz=10, response_window_s=2, calibration_s=2)
        scenarios = generate_scenarios(12, config=config, duration_s=100, seed=17)
        starts = [scenario.events[0].start for scenario in scenarios]
        self.assertGreater(len(set(starts)), 8)
        for scenario in scenarios:
            event = scenario.events[0]
            self.assertGreaterEqual(float(scenario.humidity_rh.min()), 0.0)
            self.assertLessEqual(float(scenario.humidity_rh.max()), 100.0)
            self.assertLess(float(scenario.concentration_ppm[event.start]), 1.0)
            transition = scenario.concentration_ppm[event.start:]
            self.assertTrue(np.any((transition > 1.0) & (transition < 0.9 * event.concentration_ppm)))

    def test_training_baseline_is_estimated_from_observed_voltage(self) -> None:
        """训练特征不得读取合成数据中的真实基线 oracle。"""
        config = SystemConfig(fs_hz=10, response_window_s=1, calibration_s=1)
        voltage = np.full(50, 1.2, dtype=np.float32)
        voltage[20:] = 1.3
        scenario = Scenario(
            voltage=voltage,
            temperature_c=np.full(50, 25.0, dtype=np.float32),
            humidity_rh=np.full(50, 40.0, dtype=np.float32),
            true_baseline=np.full(50, 0.5, dtype=np.float32),
            concentration_ppm=np.zeros(50, dtype=np.float32),
            events=(GasEvent(20, 40, 400.0),),
            seed=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.npz"
            save_scenarios(path, [scenario], config)
            features, _ = StreamWindowDataset(str(path), config=config)[0]
        self.assertAlmostEqual(float(features[0, 0]), 0.2, places=5)

    def test_warmup_cosine_learning_rate_reaches_boundaries(self) -> None:
        """学习率应从 10% 峰值预热，并在最后一轮降到指定下限。"""
        values = [cosine_learning_rate(epoch, total_epochs=40, peak_lr=5e-4,
                                       min_lr=1e-5, warmup_epochs=2)
                  for epoch in range(40)]
        self.assertAlmostEqual(values[0], 5e-5)
        self.assertAlmostEqual(values[1], 5e-4)
        self.assertAlmostEqual(values[-1], 1e-5)
        self.assertTrue(all(left >= right for left, right in zip(values[1:], values[2:])))

    def test_cuda_device_rejects_unavailable_gpu(self) -> None:
        """请求 CUDA 时，未发现可用 NVIDIA GPU 必须明确失败。"""
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "需要可用的 NVIDIA CUDA GPU"):
                configure_cuda_for_causal_tcn(torch.device("cuda"))

    def test_scenario_round_trip_without_pickle(self) -> None:
        """保存后再读取应保留波形、事件起止及浓度标签。"""
        config = SystemConfig(fs_hz=10, response_window_s=2, calibration_s=2)
        scenarios = generate_scenarios(2, config=config, duration_s=100, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.npz"
            save_scenarios(path, scenarios, config)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(int(data["schema_version"]), SCHEMA_VERSION)
                self.assertEqual(str(data["generator_version"]), GENERATOR_VERSION)
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

    def test_model_langmuir_output_is_positive(self) -> None:
        """Langmuir 标定输出应保持为正，并允许表达高浓度。"""
        model = StreamingTCN(channels=(8,)).eval()
        with torch.no_grad():
            prediction = model(torch.zeros(2, 8, 20))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())
        levels = torch.tensor((100, 200, 400, 600, 800, 1000, 1200))
        self.assertTrue(torch.isin(prediction, levels).all())

    def test_log_loss_has_finite_gradient(self) -> None:
        """数量级预测错误时，对数损失仍应提供有限且非零的梯度。"""
        prediction = torch.full((2, 20), 10.0, requires_grad=True)
        target = torch.tensor([100.0, 1000.0])
        loss = weighted_stream_loss(prediction, target, torch.ones(20), 1000.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

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
