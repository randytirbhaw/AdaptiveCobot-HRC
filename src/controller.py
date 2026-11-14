"""ISO/TS 15066 cobot supervisory controller and 6-DOF UR5e kinematics.

Computes runtime velocity scaling (``v_scale`` in ``[0.10, 1.00]``) and
anthropometric vertical handoff offset (``delta_z`` in ``[-0.25, +0.25]`` m)
with asymmetric slew-rate limiting, conformal safety margins, and immediate
fail-safe fallback on corrupted or out-of-bounds telemetry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model import AdaptiveHRCPredictor, RiskPrediction
from src.preprocessing import OnlineWindowBuffer, TelemetryPreprocessor
from src.sim_env import SafetyTier

MIN_V_SCALE: float = 0.10
MAX_V_SCALE: float = 1.00
MIN_DELTA_Z_M: float = -0.25
MAX_DELTA_Z_M: float = 0.25

FAIL_SAFE_V_SCALE: float = 0.10
FAIL_SAFE_DELTA_Z_M: float = 0.0

TIER_NAMES: dict[int, str] = {
    int(SafetyTier.NOMINAL): "NOMINAL",
    int(SafetyTier.CAUTION): "CAUTION",
    int(SafetyTier.HIGH_RISK): "HIGH_RISK",
}


@dataclass(frozen=True)
class ControllerConfig:
    """ISO/TS 15066 Clause 5.5.4 and ergonomic handoff control parameters."""

    min_v_scale: float = MIN_V_SCALE
    max_v_scale: float = MAX_V_SCALE
    min_delta_z_m: float = MIN_DELTA_Z_M
    max_delta_z_m: float = MAX_DELTA_Z_M
    nominal_bench_height_m: float = 1.05
    elbow_height_ratio: float = 0.63
    fatigue_reach_relief_m: float = 0.06
    reaction_time_s: float = 0.15
    stopping_time_s: float = 0.12
    intrusion_distance_m: float = 0.18
    base_uncertainty_m: float = 0.08
    conformal_uncertainty_gain_m: float = 0.32
    max_decel_step: float = 0.28
    max_accel_step: float = 0.08
    max_dz_step_m: float = 0.04
    tier_v_caps: tuple[float, float, float] = (1.00, 0.62, 0.28)
    valid_height_bounds_m: tuple[float, float] = (1.20, 2.30)
    valid_fatigue_bounds: tuple[float, float] = (0.0, 1.0)
    valid_vel_var_bounds: tuple[float, float] = (0.0, 5.0)
    valid_separation_bounds_m: tuple[float, float] = (0.05, 5.0)
    valid_approach_speed_bounds_mps: tuple[float, float] = (0.0, 4.0)


@dataclass(frozen=True)
class ControlCommand:
    """Validated runtime cobot supervisory command."""

    v_scale: float
    delta_z: float
    target_z_m: float
    protective_distance_m: float
    safety_state: str
    is_fallback: bool


def _is_valid_finite(value: object, bounds: tuple[float, float]) -> bool:
    """Check that ``value`` is a real finite scalar inside inclusive ``bounds``."""
    if value is None or isinstance(value, bool):
        return False
    try:
        scalar = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if not math.isfinite(scalar):
        return False
    return bounds[0] <= scalar <= bounds[1]


class AdaptiveCobotController:
    """Real-time cobot velocity scaling and ergonomic workspace envelope supervisor."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._prev_v_scale: float | None = None
        self._prev_delta_z: float | None = None

    def reset(self) -> None:
        """Reset internal slew-rate limiter state."""
        self._prev_v_scale = None
        self._prev_delta_z = None

    def validate_telemetry(self, sample: Mapping[str, float | int | None]) -> bool:
        """Return ``True`` only if all required sensor fields are finite and in-bounds."""
        cfg = self.config
        if not _is_valid_finite(sample.get("height_m"), cfg.valid_height_bounds_m):
            return False
        if not _is_valid_finite(sample.get("fatigue_index"), cfg.valid_fatigue_bounds):
            return False
        if not _is_valid_finite(sample.get("velocity_variance"), cfg.valid_vel_var_bounds):
            return False
        if "separation_distance_m" in sample and not _is_valid_finite(
            sample.get("separation_distance_m"), cfg.valid_separation_bounds_m
        ):
            return False
        if "approach_speed_mps" in sample and not _is_valid_finite(
            sample.get("approach_speed_mps"), cfg.valid_approach_speed_bounds_mps
        ):
            return False
        return True

    def _fail_safe_command(self) -> ControlCommand:
        """Immediate deterministic fail-safe command bypassing slew limiters."""
        self._prev_v_scale = FAIL_SAFE_V_SCALE
        self._prev_delta_z = FAIL_SAFE_DELTA_Z_M
        return ControlCommand(
            v_scale=FAIL_SAFE_V_SCALE,
            delta_z=FAIL_SAFE_DELTA_Z_M,
            target_z_m=self.config.nominal_bench_height_m,
            protective_distance_m=1.20,
            safety_state="FAIL_SAFE",
            is_fallback=True,
        )

    def compute_command(
        self,
        telemetry: Mapping[str, float | int | None],
        prediction: RiskPrediction | None,
    ) -> ControlCommand:
        """Compute bounded ``(v_scale, delta_z)`` from telemetry and conformal ML output."""
        cfg = self.config
        if prediction is None or not self.validate_telemetry(telemetry):
            return self._fail_safe_command()

        if not (
            _is_valid_finite(prediction.risk_point, (0.0, 1.0))
            and _is_valid_finite(prediction.risk_upper_95, (0.0, 1.0))
            and prediction.safety_tier in TIER_NAMES
        ):
            return self._fail_safe_command()

        height_m = float(telemetry["height_m"])  # type: ignore[arg-type]
        sep_dist_m = float(telemetry.get("separation_distance_m") or 1.05)
        approach_mps = float(telemetry.get("approach_speed_mps") or 0.0)
        r_upper = float(prediction.risk_upper_95)

        # ISO/TS 15066 Clause 5.5.4 protective separation distance inflated by conformal risk
        z_uncertainty = cfg.base_uncertainty_m + cfg.conformal_uncertainty_gain_m * r_upper
        protective_dist_m = (
            approach_mps * (cfg.reaction_time_s + cfg.stopping_time_s)
            + cfg.intrusion_distance_m
            + z_uncertainty
        )

        # Allowable speed ratio from separation margin + ergonomic risk + discrete tier cap
        ssm_margin = (sep_dist_m - protective_dist_m) / 0.75
        v_ssm = float(np.clip(0.35 + 0.65 * ssm_margin, cfg.min_v_scale, cfg.max_v_scale))
        v_ergo = float(np.clip(1.0 - 0.85 * r_upper, cfg.min_v_scale, cfg.max_v_scale))
        v_tier_cap = cfg.tier_v_caps[prediction.safety_tier]
        raw_v = min(v_ssm, v_ergo, v_tier_cap)

        # Asymmetric slew-rate limiting: fast braking, gentle acceleration
        if self._prev_v_scale is None:
            smoothed_v = raw_v
        else:
            delta_v = raw_v - self._prev_v_scale
            bounded_dv = float(np.clip(delta_v, -cfg.max_decel_step, cfg.max_accel_step))
            smoothed_v = self._prev_v_scale + bounded_dv

        v_scale = float(np.clip(smoothed_v, cfg.min_v_scale, cfg.max_v_scale))

        # Anthropometric neutral elbow height minus progressive shoulder-fatigue relief
        neutral_elbow_z = cfg.elbow_height_ratio * height_m
        raw_dz = (neutral_elbow_z - cfg.nominal_bench_height_m) - (
            cfg.fatigue_reach_relief_m * r_upper
        )
        clamped_raw_dz = float(np.clip(raw_dz, cfg.min_delta_z_m, cfg.max_delta_z_m))

        if self._prev_delta_z is None:
            delta_z = clamped_raw_dz
        else:
            step_dz = float(
                np.clip(clamped_raw_dz - self._prev_delta_z, -cfg.max_dz_step_m, cfg.max_dz_step_m)
            )
            delta_z = float(
                np.clip(self._prev_delta_z + step_dz, cfg.min_delta_z_m, cfg.max_delta_z_m)
            )

        self._prev_v_scale = v_scale
        self._prev_delta_z = delta_z

        return ControlCommand(
            v_scale=round(v_scale, 6),
            delta_z=round(delta_z, 6),
            target_z_m=round(cfg.nominal_bench_height_m + delta_z, 6),
            protective_distance_m=round(protective_dist_m, 6),
            safety_state=TIER_NAMES[prediction.safety_tier],
            is_fallback=False,
        )


class UR5eKinematics:
    """Analytical Denavit-Hartenberg forward & damped-least-squares inverse kinematics."""

    # Standard UR5e DH link parameters (meters & radians)
    A_M: tuple[float, ...] = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
    D_M: tuple[float, ...] = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
    ALPHA_RAD: tuple[float, ...] = (math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0)
    PEDESTAL_Z_M: float = 0.78

    @classmethod
    def forward_chain(cls, q_rad: np.ndarray) -> np.ndarray:
        """Compute 3D joint positions ``(7, 3)`` from base pedestal to tool flange."""
        if len(q_rad) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(q_rad)}")

        transform = np.eye(4, dtype=np.float64)
        transform[2, 3] = cls.PEDESTAL_Z_M
        points = np.zeros((7, 3), dtype=np.float64)
        points[0] = transform[:3, 3]

        for idx in range(6):
            ct, st = math.cos(float(q_rad[idx])), math.sin(float(q_rad[idx]))
            ca, sa = math.cos(cls.ALPHA_RAD[idx]), math.sin(cls.ALPHA_RAD[idx])
            a_i, d_i = cls.A_M[idx], cls.D_M[idx]

            dh_step = np.array(
                [
                    [ct, -st * ca, st * sa, a_i * ct],
                    [st, ct * ca, -ct * sa, a_i * st],
                    [0.0, sa, ca, d_i],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            transform = transform @ dh_step
            points[idx + 1] = transform[:3, 3]

        return points

    @classmethod
    def solve_ik_position(
        cls,
        target_xyz: np.ndarray,
        q_init: np.ndarray | None = None,
        damping: float = 0.06,
        max_iters: int = 35,
    ) -> np.ndarray:
        """Damped least-squares position IK keeping wrist oriented toward handoff zone."""
        q = (
            np.array([0.0, -1.15, 1.55, -1.95, -1.57, 0.0], dtype=np.float64)
            if q_init is None
            else np.asarray(q_init, dtype=np.float64).copy()
        )
        eps = 1e-5
        eye3 = np.eye(3, dtype=np.float64)

        for _ in range(max_iters):
            ee_pos = cls.forward_chain(q)[-1]
            err = target_xyz - ee_pos
            if float(np.linalg.norm(err)) < 1e-3:
                break

            jac = np.zeros((3, 6), dtype=np.float64)
            for j in range(6):
                q_pert = q.copy()
                q_pert[j] += eps
                jac[:, j] = (cls.forward_chain(q_pert)[-1] - ee_pos) / eps

            jjt = jac @ jac.T + (damping**2) * eye3
            dq = jac.T @ np.linalg.solve(jjt, err)
            q += np.clip(dq, -0.25, 0.25)

        return q

    @staticmethod
    def minimum_jerk_scalar(tau: float) -> float:
        """Normalized minimum-jerk profile ``s(tau)`` for ``tau`` in ``[0, 1]``."""
        u = float(np.clip(tau, 0.0, 1.0))
        return 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)


def evaluate_control_policies(
    eval_df: pd.DataFrame,
    preprocessor: TelemetryPreprocessor,
    predictor: AdaptiveHRCPredictor,
) -> pd.DataFrame:
    """Benchmark Adaptive Conformal Controller against Static and Naive Reactive baselines."""
    online_buf = OnlineWindowBuffer(preprocessor)
    adaptive_ctrl = AdaptiveCobotController()
    cfg = adaptive_ctrl.config

    records: dict[str, list[dict[str, float]]] = {
        "Static Nominal (v=0.85, dZ=0)": [],
        "Naive Reactive Threshold": [],
        "Adaptive Conformal HRC (Ours)": [],
    }

    prev_worker = None
    prev_naive_v = 0.85

    for row in eval_df.to_dict(orient="records"):
        w_id = int(row["worker_id"])
        if prev_worker != w_id:
            adaptive_ctrl.reset()
            online_buf.reset(w_id)
            prev_naive_v = 0.85
            prev_worker = w_id

        x_scaled = online_buf.push_and_transform(row)
        pred = predictor.predict_one(x_scaled)
        cmd = adaptive_ctrl.compute_command(row, pred)

        true_risk = float(row["risk_score"])
        sep_m = float(row["separation_distance_m"])
        app_mps = float(row["approach_speed_mps"])
        ideal_z = cfg.elbow_height_ratio * float(row["height_m"]) - (
            cfg.fatigue_reach_relief_m * true_risk
        )
        ideal_z = float(
            np.clip(
                ideal_z,
                cfg.nominal_bench_height_m + cfg.min_delta_z_m,
                cfg.nominal_bench_height_m + cfg.max_delta_z_m,
            )
        )

        # Ground-truth safe speed limit from true latent risk and separation
        true_sp = (
            app_mps * (cfg.reaction_time_s + cfg.stopping_time_s)
            + cfg.intrusion_distance_m
            + cfg.base_uncertainty_m
            + cfg.conformal_uncertainty_gain_m * true_risk
        )
        safe_v_limit = min(
            float(np.clip(0.35 + 0.65 * ((sep_m - true_sp) / 0.75), 0.10, 1.00)),
            float(np.clip(1.0 - 0.85 * true_risk, 0.10, 1.00)),
        )

        # Policy 1: Static Nominal
        static_v = 0.85
        static_z = cfg.nominal_bench_height_m
        records["Static Nominal (v=0.85, dZ=0)"].append(
            {
                "violation": float(static_v > safe_v_limit + 0.03),
                "ergo_err_cm": abs(static_z - ideal_z) * 100.0,
                "dv": 0.0,
                "safe_throughput": static_v * float(static_v <= safe_v_limit + 0.03),
            }
        )

        # Policy 2: Instantaneous Naive Threshold (no rolling features, no conformal bound, no slew)
        inst_f = float(row["fatigue_index"])
        naive_v = 0.25 if inst_f > 0.65 else (0.55 if inst_f > 0.35 else 0.95)
        naive_z = cfg.nominal_bench_height_m
        records["Naive Reactive Threshold"].append(
            {
                "violation": float(naive_v > safe_v_limit + 0.03),
                "ergo_err_cm": abs(naive_z - ideal_z) * 100.0,
                "dv": abs(naive_v - prev_naive_v),
                "safe_throughput": naive_v * float(naive_v <= safe_v_limit + 0.03),
            }
        )
        prev_naive_v = naive_v

        # Policy 3: Adaptive Conformal Controller
        records["Adaptive Conformal HRC (Ours)"].append(
            {
                "violation": float(cmd.v_scale > safe_v_limit + 0.03),
                "ergo_err_cm": abs(cmd.target_z_m - ideal_z) * 100.0,
                "dv": 0.0,
                "safe_throughput": cmd.v_scale * float(cmd.v_scale <= safe_v_limit + 0.03),
            }
        )

    # Compute sequential |dv| smoothness properly per policy
    adaptive_vs = []
    online_buf.reset()
    adaptive_ctrl.reset()
    prev_w = None
    prev_v = None
    for idx, row in enumerate(eval_df.to_dict(orient="records")):
        w_id = int(row["worker_id"])
        if prev_w != w_id:
            adaptive_ctrl.reset()
            online_buf.reset(w_id)
            prev_v = None
            prev_w = w_id
        c = adaptive_ctrl.compute_command(row, predictor.predict_one(online_buf.push_and_transform(row)))
        dv_val = 0.0 if prev_v is None else abs(c.v_scale - prev_v)
        records["Adaptive Conformal HRC (Ours)"][idx]["dv"] = dv_val
        prev_v = c.v_scale
        adaptive_vs.append(c.v_scale)

    summary_rows = []
    for policy_name, items in records.items():
        sub = pd.DataFrame(items)
        summary_rows.append(
            {
                "Policy": policy_name,
                "Safety Violation Rate (%)": round(float(sub["violation"].mean()) * 100.0, 2),
                "Mean Ergonomic Z Error (cm)": round(float(sub["ergo_err_cm"].mean()), 2),
                "Mean Command Jerk |Δv|": round(float(sub["dv"].mean()), 4),
                "Safe Throughput Index": round(float(sub["safe_throughput"].mean()), 3),
            }
        )
    return pd.DataFrame(summary_rows)
