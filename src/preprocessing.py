"""Causal rolling-window feature extraction and online/offline scaling parity.

Provides vectorized batch preprocessing for training alongside an O(1) stateful
ring-buffer extractor for real-time cobot control loops, guaranteeing zero
future-data leakage and identical feature representations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_TELEMETRY_COLUMNS: tuple[str, ...] = (
    "height_m",
    "fatigue_index",
    "velocity_variance",
    "separation_distance_m",
    "approach_speed_mps",
)

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "height_m",
    "ergonomic_reach_abs_m",
    "fatigue_index",
    "fatigue_roll_mean",
    "fatigue_roll_std",
    "fatigue_delta",
    "velocity_variance",
    "vel_var_roll_mean",
    "vel_var_roll_max",
    "proximity_hazard",
)


def _compute_proximity_hazard(
    separation_distance_m: np.ndarray | float,
    approach_speed_mps: np.ndarray | float,
) -> np.ndarray | float:
    """Compute dimensionless human-cobot proximity hazard index."""
    closeness = np.clip((1.35 - separation_distance_m) / 1.05, 0.0, 1.5)
    closing_factor = 0.55 + 0.45 * np.clip(approach_speed_mps / 0.80, 0.0, 2.0)
    return closeness * closing_factor


@dataclass
class TelemetryPreprocessor:
    """Causal rolling-window feature transformer with train-only standardization."""

    window_size: int = 10
    nominal_bench_height_m: float = 1.05
    elbow_height_ratio: float = 0.63
    scaler: StandardScaler = field(default_factory=StandardScaler, init=False)
    is_fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract strictly causal rolling features without scaling."""
        if df.empty:
            raise ValueError("Cannot extract features from an empty DataFrame.")
        missing = [col for col in RAW_TELEMETRY_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required telemetry columns: {missing}")

        if "worker_id" in df.columns:
            grouped_fatigue = df.groupby("worker_id", sort=False)["fatigue_index"]
            grouped_vel = df.groupby("worker_id", sort=False)["velocity_variance"]
            fatigue_mean = grouped_fatigue.transform(
                lambda s: s.rolling(self.window_size, min_periods=1).mean()
            )
            fatigue_var = grouped_fatigue.transform(
                lambda s: s.rolling(self.window_size, min_periods=1).var(ddof=0)
            )
            vel_mean = grouped_vel.transform(
                lambda s: s.rolling(self.window_size, min_periods=1).mean()
            )
            vel_max = grouped_vel.transform(
                lambda s: s.rolling(self.window_size, min_periods=1).max()
            )
        else:
            fatigue_mean = df["fatigue_index"].rolling(self.window_size, min_periods=1).mean()
            fatigue_var = df["fatigue_index"].rolling(self.window_size, min_periods=1).var(ddof=0)
            vel_mean = df["velocity_variance"].rolling(self.window_size, min_periods=1).mean()
            vel_max = df["velocity_variance"].rolling(self.window_size, min_periods=1).max()

        fatigue_std = np.sqrt(np.maximum(fatigue_var.to_numpy(dtype=np.float64), 0.0))
        height_arr = df["height_m"].to_numpy(dtype=np.float64)
        fatigue_arr = df["fatigue_index"].to_numpy(dtype=np.float64)
        fatigue_mean_arr = fatigue_mean.to_numpy(dtype=np.float64)
        vel_arr = df["velocity_variance"].to_numpy(dtype=np.float64)
        sep_arr = df["separation_distance_m"].to_numpy(dtype=np.float64)
        speed_arr = df["approach_speed_mps"].to_numpy(dtype=np.float64)

        ergonomic_reach_abs = np.abs(
            self.elbow_height_ratio * height_arr - self.nominal_bench_height_m
        )
        proximity_hazard = _compute_proximity_hazard(sep_arr, speed_arr)

        return pd.DataFrame(
            {
                "height_m": height_arr,
                "ergonomic_reach_abs_m": ergonomic_reach_abs,
                "fatigue_index": fatigue_arr,
                "fatigue_roll_mean": fatigue_mean_arr,
                "fatigue_roll_std": fatigue_std,
                "fatigue_delta": fatigue_arr - fatigue_mean_arr,
                "velocity_variance": vel_arr,
                "vel_var_roll_mean": vel_mean.to_numpy(dtype=np.float64),
                "vel_var_roll_max": vel_max.to_numpy(dtype=np.float64),
                "proximity_hazard": np.asarray(proximity_hazard, dtype=np.float64),
            },
            index=df.index,
        )

    def fit(self, train_df: pd.DataFrame) -> TelemetryPreprocessor:
        """Fit feature standardization strictly on training telemetry."""
        features_df = self.extract_features(train_df)
        self.scaler.fit(features_df.to_numpy(dtype=np.float64))
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Extract causal features and apply fitted standardization."""
        if not self.is_fitted:
            raise RuntimeError("TelemetryPreprocessor must be fitted before calling transform().")
        features_df = self.extract_features(df)
        return np.asarray(self.scaler.transform(features_df.to_numpy(dtype=np.float64)), dtype=np.float64)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit on training DataFrame and return scaled feature matrix."""
        return self.fit(train_df).transform(train_df)


class OnlineWindowBuffer:
    """Stateful O(1) ring buffer matching ``TelemetryPreprocessor`` at runtime."""

    def __init__(self, preprocessor: TelemetryPreprocessor) -> None:
        if not preprocessor.is_fitted:
            raise RuntimeError("OnlineWindowBuffer requires a fitted TelemetryPreprocessor.")
        self._preprocessor = preprocessor
        self._fatigue_buffers: dict[int, deque[float]] = {}
        self._vel_buffers: dict[int, deque[float]] = {}

    def reset(self, worker_id: int | None = None) -> None:
        """Clear accumulated rolling state for one or all operators."""
        if worker_id is None:
            self._fatigue_buffers.clear()
            self._vel_buffers.clear()
        else:
            self._fatigue_buffers.pop(worker_id, None)
            self._vel_buffers.pop(worker_id, None)

    def push_and_transform(self, sample: Mapping[str, float | int]) -> np.ndarray:
        """Ingest one telemetry sample and return a ``(1, n_features)`` scaled vector."""
        worker_id = int(sample.get("worker_id", 0))
        w_len = self._preprocessor.window_size

        f_buf = self._fatigue_buffers.setdefault(worker_id, deque(maxlen=w_len))
        v_buf = self._vel_buffers.setdefault(worker_id, deque(maxlen=w_len))

        height_m = float(sample["height_m"])
        fatigue_val = float(sample["fatigue_index"])
        vel_val = float(sample["velocity_variance"])
        sep_dist = float(sample.get("separation_distance_m", 1.0))
        approach_spd = float(sample.get("approach_speed_mps", 0.0))

        f_buf.append(fatigue_val)
        v_buf.append(vel_val)

        f_arr = np.fromiter(f_buf, dtype=np.float64, count=len(f_buf))
        v_arr = np.fromiter(v_buf, dtype=np.float64, count=len(v_buf))

        f_mean = float(np.mean(f_arr))
        f_std = float(np.sqrt(max(float(np.var(f_arr, ddof=0)), 0.0)))
        v_mean = float(np.mean(v_arr))
        v_max = float(np.max(v_arr))

        ergonomic_reach_abs = abs(
            self._preprocessor.elbow_height_ratio * height_m
            - self._preprocessor.nominal_bench_height_m
        )
        prox_hazard = float(_compute_proximity_hazard(sep_dist, approach_spd))

        raw_vector = np.array(
            [
                [
                    height_m,
                    ergonomic_reach_abs,
                    fatigue_val,
                    f_mean,
                    f_std,
                    fatigue_val - f_mean,
                    vel_val,
                    v_mean,
                    v_max,
                    prox_hazard,
                ]
            ],
            dtype=np.float64,
        )
        return np.asarray(self._preprocessor.scaler.transform(raw_vector), dtype=np.float64)
