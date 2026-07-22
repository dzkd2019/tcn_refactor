"""流式 TCN 的训练、验证和 checkpoint 读写。

本模块只训练 ``StreamingTCN``：模型对输入 ``(B, 8, L)`` 输出 ``(B, L)``，
监督信号是同一浓度标签沿时间维复制后的 ``(B, L)``。时间越靠后，损失权重
越大，从而让模型更重视 15 秒窗口末端的最终预测。
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import SystemConfig
from .dataset import StreamWindowDataset
from .model import StreamingTCN


def configure_cuda_for_causal_tcn(device: torch.device) -> None:
    """检查 ROCm 设备并为因果 TCN 配置 GPU 后端。

    PyTorch 的 ROCm 构建沿用 ``cuda`` 设备名和 API。选择 GPU 时必须确认当前
    安装确实是 ROCm 构建，防止误用 CUDA 轮子或静默退回 CPU。
    """
    if device.type == "cuda":
        if torch.version.hip is None:
            build = f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU"
            raise RuntimeError(f"--device cuda 需要 PyTorch ROCm 构建，当前为 {build} 构建")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch ROCm 已安装，但未检测到可用 AMD GPU")


def time_weights(length: int, *, ramp_samples: int, device: torch.device) -> torch.Tensor:
    """生成形状 ``(L,)`` 的时间权重，前期从 0.05 线性增加到 1。"""
    index = torch.arange(length, dtype=torch.float32, device=device)
    return torch.clamp(index / max(1, ramp_samples), min=0.05, max=1.0)


def cosine_learning_rate(epoch: int, *, total_epochs: int, peak_lr: float,
                         min_lr: float, warmup_epochs: int) -> float:
    """返回指定轮次的线性 warmup 加余弦退火学习率。"""
    if not 0 <= epoch < total_epochs:
        raise ValueError("epoch 必须位于训练轮次范围内")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs 必须小于总训练轮次")
    if not 0 <= min_lr <= peak_lr:
        raise ValueError("min_lr 必须位于 0 和峰值学习率之间")
    if epoch < warmup_epochs:
        progress = epoch / max(1, warmup_epochs - 1)
        return peak_lr * (0.1 + 0.9 * progress)
    cosine_steps = total_epochs - warmup_epochs
    progress = (epoch - max(0, warmup_epochs - 1)) / max(1, cosine_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def weighted_stream_loss(prediction: torch.Tensor, target_ppm: torch.Tensor,
                         weights: torch.Tensor, concentration_scale: float) -> torch.Tensor:
    """计算逐时刻对数浓度均方损失，并强化窗口末端预测。

    参数形状：``prediction=(B, L)``，``target_ppm=(B,)``，``weights=(L,)``。
    ``log(prediction)-log(target)`` 在小误差时近似相对误差；平方项让远离
    中位数的低/高浓度样本获得更强梯度，又不会直接用 ``1/target`` 放大
    低浓度噪声。末端损失额外乘 2。

    ``weights`` 和 ``concentration_scale`` 参数为兼容旧调用保留，不再参与计算。
    """
    del weights, concentration_scale
    target = target_ppm[:, None].expand_as(prediction)
    log_error = torch.log(prediction.clamp_min(1.0)) - torch.log(target.clamp_min(1.0))
    squared_log_error = log_error.square()
    # 当前部署在 15 秒时锁定读数，因此主要监督最后 3 秒。相比把整个窗口
    # 都纳入损失，这不会强迫模型在气体响应尚不可辨识时猜出浓度。
    tail_length = min(300, prediction.shape[1])
    temporal_loss = squared_log_error[:, -tail_length:].mean()
    final_loss = squared_log_error[:, -1].mean()
    return temporal_loss + 2.0 * final_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """只使用每个窗口最后一个时间步，返回 MAE、RMSE 与 MAPE。"""
    model.eval()
    predictions, targets = [], []
    for features, target in loader:
        # features: (B, 8, L)，model(features): (B, L)，取 [:, -1] 得到 (B,)
        predictions.append(model(features.to(device))[:, -1].cpu())
        targets.append(target)
    p, y = torch.cat(predictions), torch.cat(targets)
    error = p - y
    return {"mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt()),
            "mape": float((error.abs() / y).mean() * 100)}


@torch.no_grad()
def evaluate_time_points(model: nn.Module, loader: DataLoader, device: torch.device,
                         config: SystemConfig) -> dict[str, float]:
    """评估 5、10、15 秒时的 MAPE，并报告末端 MAE/RMSE。

    模型一次前向返回 ``(B, L)`` 的全部流式预测，因此无需为每个时间点重复
    推理。该指标能直观看出预测随响应时间增加而收敛的过程。
    """
    model.eval()
    predictions, targets = [], []
    for features, target in loader:
        predictions.append(model(features.to(device)).cpu())
        targets.append(target)
    p, y = torch.cat(predictions), torch.cat(targets)
    result: dict[str, float] = {}
    for seconds in (5, 10, 15):
        index = min(round(seconds * config.fs_hz) - 1, p.shape[1] - 1)
        result[f"mape_{seconds}s"] = float(((p[:, index] - y).abs() / y).mean() * 100)
    final_error = p[:, -1] - y
    result.update(mae=float(final_error.abs().mean()),
                  rmse=float(final_error.square().mean().sqrt()))
    return result


def _split_by_scenario(dataset: StreamWindowDataset) -> tuple[Subset, Subset]:
    """按场景而非按窗口划分训练/验证集，避免同一轨迹泄漏到两个集合。"""
    scenario_ids = sorted({record.scenario_index for record in dataset.records})
    if len(scenario_ids) < 2:
        raise ValueError("至少需要两个场景，才能划分训练集和验证集")
    cut = max(1, round(len(scenario_ids) * 0.8))
    train_ids, val_ids = set(scenario_ids[:cut]), set(scenario_ids[cut:])
    train = [i for i, record in enumerate(dataset.records) if record.scenario_index in train_ids]
    val = [i for i, record in enumerate(dataset.records) if record.scenario_index in val_ids]
    return Subset(dataset, train), Subset(dataset, val)


def train_stream_model(data_path: str, checkpoint_path: str, *, validation_path: str | None = None,
                       config: SystemConfig = SystemConfig(),
                       epochs: int = 20, batch_size: int = 32, learning_rate: float = 5e-4,
                       warmup_epochs: int = 0, min_learning_rate: float | None = None,
                       max_offset_s: float = 0.0, device: str = "cpu",
                       resume_path: str | None = None) -> dict[str, float]:
    """训练模型并保存带配置元数据的 checkpoint。

    返回最后一次验证的指标字典。为了让初学者容易追踪，本函数不隐藏训练过程：
    每个 epoch 都打印训练损失和验证集指标。
    """
    torch.manual_seed(0)
    run_device = torch.device(device)
    configure_cuda_for_causal_tcn(run_device)
    print(f"[训练] 设备={run_device}，开始读取训练与验证场景……", flush=True)
    dataset = StreamWindowDataset(
        data_path, config=config, max_offset_s=max_offset_s, alignment="detection",
    )
    # 正式实验使用独立验证文件；缺省时才退回按场景的 80/20 划分，便于快速试验。
    if validation_path is None:
        train_set, val_set = _split_by_scenario(dataset)
    else:
        train_set = dataset
        # 验证集复用部署检测器，避免按理想事件真值选择出不鲁棒模型。
        val_set = StreamWindowDataset(validation_path, config=config,
                                      max_offset_s=max_offset_s, seed=10_000,
                                      alignment="detection")
    print(f"[训练] 训练窗口={len(train_set)}，验证窗口={len(val_set)}", flush=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model = StreamingTCN().to(run_device)
    if model.receptive_field < config.window_samples:
        raise ValueError("模型感受野小于训练窗口，无法学习完整响应动态")
    # 可从上一次最佳模型继续训练。优化器状态不续接，避免实现复杂度掩盖
    # 教学主线；模型权重和“当前最佳验证指标”会被安全保留。
    previous_best = float("inf")
    if resume_path is not None:
        payload = torch.load(resume_path, map_location=run_device, weights_only=True)
        # 第 4 版特征把旧的“一阶方程浓度反演”改成了弱动态先验，输入也从
        # 7 通道增加到 8 通道。即使张量形状偶然能对上，也不能混用不同语义
        # 的权重，因此在加载 state_dict 前先做显式版本检查。
        if payload.get("feature_version") != "weak_prior_v4":
            raise ValueError("续训 checkpoint 特征版本与当前弱物理先验模型不兼容")
        if payload.get("model_version") != "context_langmuir_logmse_v4":
            raise ValueError("续训 checkpoint 输出头版本与当前 Softplus 模型不兼容")
        model.load_state_dict(payload["model_state"])
        # 写回同一 checkpoint 时继承历史门槛；写入新路径时只是热启动，
        # 新实验应按自己的验证分布重新建立最佳指标。
        if Path(resume_path).resolve() == Path(checkpoint_path).resolve():
            previous_best = float(payload["validation"]["mape"])
            print(f"已加载续训权重：历史最佳验证 MAPE={previous_best:.2f}%")
        else:
            print("已加载热启动权重：将在新的验证分布上重新选择最佳模型")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 前 12 秒的响应在高湿/低浓度条件下信息不足；把主要监督信号放在
    # 窗口末端，可直接优化“15 秒内给出最终读数”的部署目标。
    weights = time_weights(config.window_samples, ramp_samples=round(12 * config.fs_hz), device=run_device)
    best_mape, best_state, best_metrics, latest = previous_best, None, {}, {}

    for epoch in range(epochs):
        if min_learning_rate is not None:
            current_lr = cosine_learning_rate(
                epoch, total_epochs=epochs, peak_lr=learning_rate,
                min_lr=min_learning_rate, warmup_epochs=warmup_epochs,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr
        else:
            current_lr = learning_rate
        dataset.set_epoch(epoch)  # 同一事件在下一轮会获得另一个可复现的窗口偏移。
        model.train()
        total_loss, seen = 0.0, 0
        for batch_index, (features, target) in enumerate(train_loader, start=1):
            features, target = features.to(run_device), target.to(run_device)
            prediction = model(features)  # (B, 8, L) -> (B, L)
            loss = weighted_stream_loss(prediction, target, weights, config.concentration_scale)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # ``loss`` 仍连接着反向传播图；记录日志时应先 detach，避免
            # PyTorch 把带梯度张量隐式转换为 Python 浮点数的警告。
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
            # 长序列 GPU 训练时每 10 批输出一次进度，兼顾可观察性与日志体积。
            if batch_index % 10 == 0:
                print(f"  epoch={epoch + 1:03d} batch={batch_index:03d}/{len(train_loader):03d} "
                      f"loss={float(loss.detach()):.5f}", flush=True)
        latest = evaluate(model, val_loader, run_device)
        print(f"epoch={epoch + 1:03d} lr={current_lr:.2e} loss={total_loss / seen:.5f} "
              f"val_mae={latest['mae']:.1f} val_mape={latest['mape']:.2f}%")
        if latest["mape"] < best_mape:
            best_mape = latest["mape"]
            best_metrics = latest.copy()
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            _save_checkpoint(checkpoint_path, best_state, config, best_metrics)

    # 若本轮没有刷新历史最佳，直接返回 checkpoint 中的历史最佳指标。
    if best_state is None:
        return {"mape": previous_best}
    return best_metrics


def _save_checkpoint(checkpoint_path: str, model_state: dict[str, torch.Tensor], config: SystemConfig,
                     validation: dict[str, float]) -> None:
    """原子性地保存当前最佳模型；每次验证提升立刻调用，防止长任务中断丢失结果。"""
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_state": model_state, "system_config": asdict(config),
                "model": {"input_channels": 8, "channels": (32, 64, 96, 96, 64, 32),
                          "kernel_size": 15, "output_max_ppm": 1200.0,
                          "output_mode": "langmuir", "output_scale_ppm": 200.0,
                          "use_context_head": True, "langmuir_k": 0.045,
                          "calibration_levels_ppm": (
                              100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0,
                          ), "quantize_in_eval": True},
                "model_version": "context_langmuir_logmse_v4",
                "feature_version": "weak_prior_v4",
                "validation": validation}, target)


def load_stream_model(checkpoint_path: str, *, device: str = "cpu") -> tuple[StreamingTCN, SystemConfig]:
    """读取 checkpoint，返回已切换到 ``eval`` 模式的模型和对应系统配置。"""
    run_device = torch.device(device)
    configure_cuda_for_causal_tcn(run_device)
    payload = torch.load(checkpoint_path, map_location=run_device, weights_only=True)
    if payload.get("feature_version") != "weak_prior_v4":
        raise ValueError("checkpoint 特征版本与当前弱物理先验模型不兼容")
    model_config = payload["model"].copy()
    context_versions = {"context_softplus_log_huber_v3", "context_langmuir_logmse_v4"}
    if payload.get("model_version") not in context_versions:
        model_config["use_context_head"] = False
    if payload.get("model_version") != "context_langmuir_logmse_v4":
        model_config["quantize_in_eval"] = False
    if "model_version" not in payload:
        model_config["output_mode"] = "sigmoid"
    model = StreamingTCN(**model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, SystemConfig(**payload["system_config"])
