from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """可由 cuDNN 加速的因果一维卷积。

    ``Conv1d`` 的内置 padding 会在左右两侧补零。卷积后裁掉多出来的右侧
    ``padding`` 个输出，保留的第 t 个输出便只依赖 ``t`` 及其左侧输入。
    相比 ``F.pad(x, (left, 0)) + Conv1d``，此写法在本机 GTX 1650 上可稳定
    使用 cuDNN 加速大膨胀率卷积。
    """

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int,
                 dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        # weight_norm 把卷积核参数化为“方向 + 大小”，优化器可以分别学习两者。
        # 对多层膨胀 TCN，这通常比直接更新完整卷积核更稳定。它只改变权重
        # 参数化方式，不改变输入输出形状，也不会破坏因果性。
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, padding=self.padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        # 输入长度为 L 时，内置 padding 的输出长度为 L + padding。
        # 去掉右侧未来填充对应的输出后，最终长度恢复为 L。
        return output[:, :, :-self.padding] if self.padding else output


class TemporalBlock(nn.Module):
    """因果残差块，不含跨时间或跨 batch 的归一化层。

    移除 BatchNorm 的原因是它在训练时会统计整段时间序列，破坏流式训练
    与部署时“只使用过去样本”的一致性。
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float,
                 kernel_size: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(torch.relu(self.conv1(x)))
        h = self.dropout(torch.relu(self.conv2(h)))
        return torch.relu(h + self.skip(x))


class StreamingTCN(nn.Module):
    """仅用于流式部署的因果 TCN。

    输入张量形状为 ``(B, 8, L)``：B 是 batch 大小，8 是特征通道数，L
    是时间长度；输出形状为 ``(B, L)``，每个时间步给出一个浓度预测。
    """

    def __init__(self, input_channels: int = 8,
                 channels: tuple[int, ...] = (32, 64, 96, 96, 64, 32),
                 kernel_size: int = 15, dropout: float = 0.10,
                 output_max_ppm: float = 1200.0, output_mode: str = "langmuir",
                 output_scale_ppm: float = 200.0,
                 use_context_head: bool = True, langmuir_k: float = 0.045,
                 calibration_levels_ppm: tuple[float, ...] | None = (
                     100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0,
                 ), quantize_in_eval: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_channels
        for index, current in enumerate(channels):
            layers.append(TemporalBlock(previous, current, 2**index, dropout, kernel_size))
            previous = current
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(previous, previous, 1), nn.ReLU(), nn.Conv1d(previous, 1, 1))
        self.context_head = None
        if use_context_head:
            self.context_head = nn.Sequential(
                nn.Conv1d(input_channels, 16, 1), nn.SiLU(), nn.Conv1d(16, 1, 1),
            )
            nn.init.zeros_(self.context_head[-1].weight)
            nn.init.zeros_(self.context_head[-1].bias)
        self.output_max_ppm, self.output_mode = output_max_ppm, output_mode
        self.output_scale_ppm, self.langmuir_k = output_scale_ppm, langmuir_k
        self.quantize_in_eval = quantize_in_eval
        levels = () if calibration_levels_ppm is None else calibration_levels_ppm
        self.register_buffer(
            "_calibration_levels", torch.tensor(levels, dtype=torch.float32),
            persistent=False,
        )
        if output_mode == "softplus":
            initial = output_max_ppm / 2.0 / output_scale_ppm
            self.head[-1].bias.data.fill_(math.log(math.expm1(initial)))
        elif output_mode == "langmuir":
            root_concentration = math.sqrt(output_max_ppm / 2.0)
            occupancy = langmuir_k * root_concentration / (
                1.0 + langmuir_k * root_concentration
            )
            self.head[-1].bias.data.fill_(math.log(occupancy / (1.0 - occupancy)))
        elif output_mode != "sigmoid":
            raise ValueError("output_mode 必须是 langmuir、softplus 或 sigmoid")
        # 每个残差块有两层卷积；6 层、kernel=15 时感受野为 1765 点，
        # 大于 100 Hz 下的 15 秒训练窗口（1500 点）。
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(2**i for i in range(len(channels)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向计算，输入 ``(B, 8, L)``，返回 ``(B, L)``。"""
        if features.ndim != 3:
            raise ValueError("features 的形状必须为 (batch, channels, time)")
        raw = self.head(self.backbone(features))
        if self.context_head is not None:
            raw = raw + self.context_head(features)
        raw = raw.squeeze(1)
        if self.output_mode == "sigmoid":
            prediction = torch.sigmoid(raw) * self.output_max_ppm
        elif self.output_mode == "langmuir":
            occupancy = torch.sigmoid(raw).clamp(max=0.90)
            root_concentration = occupancy / (self.langmuir_k * (1.0 - occupancy))
            prediction = root_concentration.square()
        else:
            prediction = F.softplus(raw) * self.output_scale_ppm
        if not self.training and self.quantize_in_eval and self._calibration_levels.numel():
            distance = (prediction[..., None] - self._calibration_levels).abs()
            prediction = self._calibration_levels[distance.argmin(dim=-1)]
        return prediction
