from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SessionState(str, Enum):
    CALIBRATING = "calibrating"
    IDLE = "idle"
    RESPONDING = "responding"
    LATCHED = "latched"


@dataclass(frozen=True)
class Sample:
    voltage: float
    temperature_c: float
    humidity_rh: float


@dataclass(frozen=True)
class GasEvent:
    start: int
    end: int
    concentration_ppm: float


@dataclass
class Scenario:
    voltage: np.ndarray
    temperature_c: np.ndarray
    humidity_rh: np.ndarray
    true_baseline: np.ndarray
    concentration_ppm: np.ndarray
    events: tuple[GasEvent, ...]
    seed: int


@dataclass(frozen=True)
class Prediction:
    state: SessionState
    concentration_ppm: float | None
    response_samples: int
    detected_event_start: int | None
