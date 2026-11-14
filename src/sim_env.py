"""Biomechanical worker-cobot shift telemetry simulator.

Models muscle fatigue dynamics using Ma's differential fatigue-recovery law,
signal-dependent human motor variability, Ornstein-Uhlenbeck wearable sensor
drift, and periodic human-robot handoff separation trajectories.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class SafetyTier(IntEnum):
    """Discrete supervisory safety states derived from composite ergonomic risk."""

    NOMINAL = 0
    CAUTION = 1
    HIGH_RISK = 2


@dataclass(frozen=True)
class ShiftSimulatorConfig:
    """Physical and physiological parameters for synthetic HRC telemetry."""

    dt_sec: float = 0.5
    fatigue_rate: float = 0.024
    recovery_rate: float = 0.016
    nominal_bench_height_m: float = 1.05
    elbow_height_ratio: float = 0.63
    min_worker_height_m: float = 1.52
    max_worker_height_m: float = 1.96
    caution_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.70


def _simulate_workload_schedule(
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate realistic alternating assembly-task load and micro-rest cycles."""
    load = np.zeros(n_steps, dtype=np.float64)
    idx = 0
    is_work_phase = True
    while idx < n_steps:
        if is_work_phase:
            duration = int(rng.integers(28, 58))
            intensity = float(rng.uniform(0.55, 1.0))
            load[idx : min(n_steps, idx + duration)] = intensity
        else:
            duration = int(rng.integers(10, 24))
            load[idx : min(n_steps, idx + duration)] = float(rng.uniform(0.02, 0.15))
        idx += duration
        is_work_phase = not is_work_phase
    return load


def _simulate_worker_episode(
    worker_id: int,
    n_steps: int,
    height_m: float,
    config: ShiftSimulatorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate a single operator shift episode with latent physiological state."""
    workload = _simulate_workload_schedule(n_steps, rng)
    latent_fatigue = np.zeros(n_steps, dtype=np.float64)
    latent_fatigue[0] = float(rng.uniform(0.04, 0.18))

    ou_noise = 0.0
    observed_fatigue = np.zeros(n_steps, dtype=np.float64)
    velocity_variance = np.zeros(n_steps, dtype=np.float64)
    separation_dist = np.zeros(n_steps, dtype=np.float64)
    approach_speed = np.zeros(n_steps, dtype=np.float64)
    risk_score = np.zeros(n_steps, dtype=np.float64)

    worker_fatigability = float(rng.uniform(0.85, 1.20))
    motor_baseline = float(rng.uniform(0.008, 0.022))
    handoff_period = float(rng.uniform(16.0, 26.0))
    phase_offset = float(rng.uniform(0.0, 2.0 * np.pi))

    ergonomic_posture_mismatch = abs(
        config.elbow_height_ratio * height_m - config.nominal_bench_height_m
    )
    posture_strain = np.clip(ergonomic_posture_mismatch / 0.25, 0.0, 1.0)

    for t in range(n_steps):
        if t > 0:
            prev_f = latent_fatigue[t - 1]
            u_load = workload[t]
            # Ma's dynamic muscle fatigue-recovery differential update
            df_dt = (
                config.fatigue_rate * worker_fatigability * u_load * (1.0 - prev_f)
                - config.recovery_rate * (1.0 - u_load) * prev_f
            )
            latent_fatigue[t] = float(np.clip(prev_f + df_dt * config.dt_sec, 0.0, 1.0))

        f_true = latent_fatigue[t]
        ou_noise = 0.82 * ou_noise + float(rng.normal(0.0, 0.022))
        observed_fatigue[t] = float(np.clip(f_true + ou_noise, 0.0, 1.0))

        # Signal-dependent motor noise with intermittent micro-hesitation bursts
        hesitation_burst = (
            float(rng.exponential(0.065)) if rng.random() < (0.04 + 0.18 * f_true) else 0.0
        )
        raw_vel_var = (
            motor_baseline
            + 0.24 * (f_true**1.75)
            + 0.04 * workload[t]
            + hesitation_burst
            + float(rng.normal(0.0, 0.006))
        )
        velocity_variance[t] = float(np.clip(raw_vel_var, 0.002, 1.50))

        # Periodic collaborative handoff motion between bench and shared workspace
        time_s = t * config.dt_sec
        cycle_angle = (2.0 * np.pi * time_s / handoff_period) + phase_offset
        d_hr = 0.95 + 0.45 * np.cos(cycle_angle) + float(rng.normal(0.0, 0.035))
        separation_dist[t] = float(np.clip(d_hr, 0.30, 1.80))

        closing_vel = max(0.0, 0.45 * (2.0 * np.pi / handoff_period) * np.sin(cycle_angle))
        approach_speed[t] = float(
            np.clip(closing_vel * (1.0 + 0.35 * workload[t]) + rng.normal(0.0, 0.04), 0.0, 1.80)
        )

        norm_jitter = float(np.clip(velocity_variance[t] / 0.35, 0.0, 1.0))
        proximity_factor = float(
            np.clip((1.35 - separation_dist[t]) / 1.05, 0.0, 1.0)
            * (0.55 + 0.45 * np.clip(approach_speed[t] / 0.8, 0.0, 1.0))
        )
        aleatoric = float(rng.normal(0.0, 0.025))
        composite_risk = (
            0.52 * f_true
            + 0.24 * norm_jitter
            + 0.14 * proximity_factor
            + 0.10 * posture_strain
            + aleatoric
        )
        risk_score[t] = float(np.clip(composite_risk, 0.0, 1.0))

    safety_tier = np.where(
        risk_score >= config.high_risk_threshold,
        int(SafetyTier.HIGH_RISK),
        np.where(
            risk_score >= config.caution_risk_threshold,
            int(SafetyTier.CAUTION),
            int(SafetyTier.NOMINAL),
        ),
    )

    steps = np.arange(n_steps, dtype=np.int64)
    return pd.DataFrame(
        {
            "worker_id": np.full(n_steps, worker_id, dtype=np.int64),
            "step": steps,
            "timestamp_s": steps * config.dt_sec,
            "height_m": np.full(n_steps, round(height_m, 4), dtype=np.float64),
            "fatigue_index": np.round(observed_fatigue, 6),
            "velocity_variance": np.round(velocity_variance, 6),
            "separation_distance_m": np.round(separation_dist, 6),
            "approach_speed_mps": np.round(approach_speed, 6),
            "risk_score": np.round(risk_score, 6),
            "safety_tier": safety_tier.astype(np.int64),
        }
    )


def generate_telemetry_dataframe(
    n_workers: int = 12,
    steps_per_worker: int = 250,
    seed: int = 42,
    config: ShiftSimulatorConfig | None = None,
) -> pd.DataFrame:
    """Generate a multi-operator collaborative robotics telemetry dataset.

    Args:
        n_workers: Number of distinct human operators to simulate.
        steps_per_worker: Number of time steps per operator shift episode.
        seed: Deterministic random seed.
        config: Optional simulator constants override.

    Returns:
        DataFrame ordered by ``(worker_id, step)`` containing sensor observations
        and supervised ergonomic safety targets.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if steps_per_worker < 2:
        raise ValueError(f"steps_per_worker must be >= 2, got {steps_per_worker}")

    cfg = config or ShiftSimulatorConfig()
    rng = np.random.default_rng(seed)

    # Stratified stature sampling ensures coverage across 5th-95th percentile adults
    strata = np.linspace(cfg.min_worker_height_m, cfg.max_worker_height_m, n_workers + 1)
    heights = [
        float(rng.uniform(strata[i], strata[i + 1])) for i in range(n_workers)
    ]
    rng.shuffle(heights)

    episodes = [
        _simulate_worker_episode(
            worker_id=worker_idx,
            n_steps=steps_per_worker,
            height_m=heights[worker_idx],
            config=cfg,
            rng=rng,
        )
        for worker_idx in range(n_workers)
    ]
    return pd.concat(episodes, ignore_index=True)


def stream_telemetry(
    df: pd.DataFrame,
    fault_injection_prob: float = 0.0,
    seed: int = 123,
) -> Iterator[dict[str, float | int]]:
    """Yield per-step telemetry dictionaries with optional sensor-fault injection.

    Args:
        df: Telemetry DataFrame generated by ``generate_telemetry_dataframe``.
        fault_injection_prob: Probability in ``[0, 1]`` of corrupting a yielded
            frame with a ``NaN`` dropout or out-of-range spike for resilience testing.
        seed: Random seed governing fault injection.
    """
    if not 0.0 <= fault_injection_prob <= 1.0:
        raise ValueError("fault_injection_prob must lie in [0.0, 1.0]")

    rng = np.random.default_rng(seed)
    records = df.to_dict(orient="records")
    for record in records:
        sample: dict[str, float | int] = dict(record)
        if fault_injection_prob > 0.0 and rng.random() < fault_injection_prob:
            fault_mode = int(rng.integers(0, 3))
            if fault_mode == 0:
                sample["fatigue_index"] = float("nan")
            elif fault_mode == 1:
                sample["velocity_variance"] = -0.25
            else:
                sample["height_m"] = 3.20
        yield sample
