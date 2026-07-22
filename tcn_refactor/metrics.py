"""端到端回放评估：以逐样本部署路径而非理想对齐窗口衡量系统。"""

from __future__ import annotations

import numpy as np

from .domain import Sample
from .session import StreamSession
from .synthetic import load_scenarios


def replay(data_path: str, session: StreamSession, *, start: int = 0,
           limit: int | None = None) -> dict[str, float]:
    """将合成场景逐点送入部署会话，返回检测和最终预测的汇总指标。

    此函数不读取真实事件来驱动 session；事件真值仅在事后用于计算检测延迟和
    15 秒预测误差，因此它能揭示检测器与模型组合的实际效果。
    """
    delays, absolute_errors, absolute_percentage_errors, missed = [], [], [], 0
    all_scenarios = load_scenarios(data_path)
    scenarios = all_scenarios[start:] if limit is None else all_scenarios[start:start + limit]
    for scenario_index, scenario in enumerate(scenarios, start=1):
        session.reset()
        detections: list[tuple[int, float]] = []
        for i, (v, t, h) in enumerate(zip(scenario.voltage, scenario.temperature_c, scenario.humidity_rh)):
            result = session.push(Sample(float(v), float(t), float(h)))
            if result.concentration_ppm is not None and result.detected_event_start is not None:
                # 锁定状态会连续返回同一个结果；每个检测起点只记录一次。
                if not detections or detections[-1][0] != result.detected_event_start:
                    detections.append((result.detected_event_start, result.concentration_ppm))
        for event in scenario.events:
            match = next(((start, value) for start, value in detections if event.start <= start < event.end), None)
            if match is None:
                missed += 1
            else:
                delays.append((match[0] - event.start) / session.config.fs_hz)
                absolute_errors.append(abs(match[1] - event.concentration_ppm))
                absolute_percentage_errors.append(
                    abs(match[1] - event.concentration_ppm) / event.concentration_ppm * 100
                )
        if scenario_index % 20 == 0:
            print(f"[回放] 已完成 {scenario_index}/{len(scenarios)} 个场景", flush=True)
    return {"events": float(len(delays) + missed), "missed": float(missed),
            "mean_detection_delay_s": float(np.mean(delays)) if delays else float("nan"),
            "final_mae_ppm": float(np.mean(absolute_errors)) if absolute_errors else float("nan"),
            "final_mape": float(np.mean(absolute_percentage_errors))
            if absolute_percentage_errors else float("nan")}
