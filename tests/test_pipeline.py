"""Unit, invariant, parity, and safety-boundary tests for AdaptiveCobot-HRC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.controller import (
    FAIL_SAFE_DELTA_Z_M,
    FAIL_SAFE_V_SCALE,
    MAX_DELTA_Z_M,
    MAX_V_SCALE,
    MIN_DELTA_Z_M,
    MIN_V_SCALE,
    AdaptiveCobotController,
    UR5eKinematics,
    evaluate_control_policies,
)
from src.model import (
    AdaptiveHRCPredictor,
    RiskPrediction,
    split_by_worker_groups,
    train_and_evaluate_pipeline,
)
from src.preprocessing import (
    ENGINEERED_FEATURE_NAMES,
    OnlineWindowBuffer,
    TelemetryPreprocessor,
)
from src.sim_env import generate_telemetry_dataframe, stream_telemetry


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Shared deterministic multi-operator telemetry dataset for fast module tests."""
    return generate_telemetry_dataframe(n_workers=12, steps_per_worker=180, seed=42)


@pytest.fixture(scope="module")
def trained_artifacts(synthetic_df: pd.DataFrame):
    """Fitted preprocessor, dual-head conformal model, and test-split metrics."""
    return train_and_evaluate_pipeline(synthetic_df, window_size=10, seed=42)


def test_sim_env_shapes_determinism_and_physical_bounds(synthetic_df: pd.DataFrame) -> None:
    expected_cols = {
        "worker_id",
        "step",
        "timestamp_s",
        "height_m",
        "fatigue_index",
        "velocity_variance",
        "separation_distance_m",
        "approach_speed_mps",
        "risk_score",
        "safety_tier",
    }
    assert synthetic_df.shape == (12 * 180, 10)
    assert expected_cols.issubset(set(synthetic_df.columns))

    # Reproducibility check
    repeat_df = generate_telemetry_dataframe(n_workers=12, steps_per_worker=180, seed=42)
    pd.testing.assert_frame_equal(synthetic_df, repeat_df)

    # Physical & physiological domain bounds
    assert synthetic_df["height_m"].between(1.50, 2.00).all()
    assert synthetic_df["fatigue_index"].between(0.0, 1.0).all()
    assert (synthetic_df["velocity_variance"] > 0.0).all()
    assert synthetic_df["risk_score"].between(0.0, 1.0).all()
    assert set(synthetic_df["safety_tier"].unique()).issubset({0, 1, 2})


def test_stream_telemetry_fault_injection() -> None:
    small_df = generate_telemetry_dataframe(n_workers=1, steps_per_worker=40, seed=7)
    clean_frames = list(stream_telemetry(small_df, fault_injection_prob=0.0, seed=7))
    assert len(clean_frames) == 40

    faulty_frames = list(stream_telemetry(small_df, fault_injection_prob=1.0, seed=7))
    ctrl = AdaptiveCobotController()
    assert all(not ctrl.validate_telemetry(frame) for frame in faulty_frames)


def test_preprocessing_batch_vs_online_ring_buffer_parity(synthetic_df: pd.DataFrame) -> None:
    train_df, _, test_df = split_by_worker_groups(synthetic_df, seed=42)
    prep = TelemetryPreprocessor(window_size=10)

    with pytest.raises(RuntimeError):
        prep.transform(test_df)

    X_train = prep.fit_transform(train_df)
    assert X_train.shape == (len(train_df), len(ENGINEERED_FEATURE_NAMES))
    assert np.isfinite(X_train).all()

    X_test_batch = prep.transform(test_df)
    online_buf = OnlineWindowBuffer(prep)
    X_test_online = np.vstack(
        [online_buf.push_and_transform(row) for row in test_df.to_dict(orient="records")]
    )

    np.testing.assert_allclose(X_test_batch, X_test_online, rtol=1e-7, atol=1e-8)


def test_worker_group_split_has_zero_operator_leakage(synthetic_df: pd.DataFrame) -> None:
    train_df, cal_df, test_df = split_by_worker_groups(synthetic_df, seed=42)
    train_w = set(train_df["worker_id"].unique())
    cal_w = set(cal_df["worker_id"].unique())
    test_w = set(test_df["worker_id"].unique())

    assert train_w.isdisjoint(cal_w)
    assert train_w.isdisjoint(test_w)
    assert cal_w.isdisjoint(test_w)


def test_model_monotonicity_conformal_coverage_and_serialization(
    synthetic_df: pd.DataFrame,
    trained_artifacts,
    tmp_path: Path,
) -> None:
    prep, predictor, metrics = trained_artifacts

    assert metrics.r2 > 0.75
    assert metrics.rmse < 0.12
    assert metrics.macro_f1 > 0.72
    assert metrics.conformal_coverage >= 0.88
    assert metrics.single_sample_latency_ms < 5.0

    # Verify strict safety monotonicity wrt fatigue and velocity variance
    _, _, test_df = split_by_worker_groups(synthetic_df, seed=42)
    X_base = prep.transform(test_df.iloc[:80].copy())
    base_risk, base_upper, _, _ = predictor.predict_arrays(X_base)

    X_higher_fatigue = X_base.copy()
    # Indices 2, 3, 6, 7, 8 correspond to fatigue & velocity_variance features (+1 monotone)
    X_higher_fatigue[:, [2, 3, 6, 7, 8]] += 0.75
    perturbed_risk, perturbed_upper, _, _ = predictor.predict_arrays(X_higher_fatigue)

    assert np.all(perturbed_risk >= base_risk - 1e-6)
    assert np.all(perturbed_upper >= base_upper - 1e-6)

    # Serialization roundtrip
    artifact_path = tmp_path / "hrc_predictor.joblib"
    predictor.save(artifact_path)
    reloaded = AdaptiveHRCPredictor.load(artifact_path)
    reloaded_pred = reloaded.predict_one(X_base[:1])
    orig_pred = predictor.predict_one(X_base[:1])
    assert reloaded_pred.risk_point == pytest.approx(orig_pred.risk_point, abs=1e-7)
    assert reloaded_pred.safety_tier == orig_pred.safety_tier


def test_controller_safety_clamping_ergonomics_and_failsafe() -> None:
    ctrl = AdaptiveCobotController()

    # Sweep extreme synthetic states to verify hard safety clamps never break
    rng = np.random.default_rng(99)
    for _ in range(150):
        sample = {
            "height_m": float(rng.uniform(1.25, 2.25)),
            "fatigue_index": float(rng.uniform(0.0, 1.0)),
            "velocity_variance": float(rng.uniform(0.0, 4.5)),
            "separation_distance_m": float(rng.uniform(0.10, 2.50)),
            "approach_speed_mps": float(rng.uniform(0.0, 3.0)),
        }
        r_pt = float(rng.uniform(0.0, 1.0))
        r_up = min(1.0, r_pt + 0.08)
        tier = int(rng.integers(0, 3))
        pred = RiskPrediction(
            risk_point=r_pt,
            risk_upper_95=r_up,
            conformal_margin=0.08,
            safety_tier=tier,
            tier_probabilities=(0.3, 0.3, 0.4),
        )
        cmd = ctrl.compute_command(sample, pred)
        assert MIN_V_SCALE <= cmd.v_scale <= MAX_V_SCALE
        assert MIN_DELTA_Z_M <= cmd.delta_z <= MAX_DELTA_Z_M
        assert not cmd.is_fallback

    # Anthropometric ergonomics directional check (tall vs short operator at equal risk)
    nominal_pred = RiskPrediction(0.15, 0.22, 0.07, 0, (0.85, 0.12, 0.03))
    ctrl.reset()
    short_cmd = ctrl.compute_command(
        {"height_m": 1.54, "fatigue_index": 0.1, "velocity_variance": 0.02},
        nominal_pred,
    )
    ctrl.reset()
    tall_cmd = ctrl.compute_command(
        {"height_m": 1.92, "fatigue_index": 0.1, "velocity_variance": 0.02},
        nominal_pred,
    )
    assert tall_cmd.delta_z > short_cmd.delta_z

    # Immediate fail-safe fallback on corrupted/out-of-range telemetry (bypasses slew limiter)
    corrupted_inputs = [
        {"height_m": float("nan"), "fatigue_index": 0.2, "velocity_variance": 0.03},
        {"height_m": 1.75, "fatigue_index": float("inf"), "velocity_variance": 0.03},
        {"height_m": 1.75, "fatigue_index": -0.1, "velocity_variance": 0.03},
        {"height_m": 1.75, "fatigue_index": 0.3, "velocity_variance": -0.05},
        {"height_m": 0.85, "fatigue_index": 0.3, "velocity_variance": 0.03},
    ]
    for bad_sample in corrupted_inputs:
        fallback_cmd = ctrl.compute_command(bad_sample, nominal_pred)
        assert fallback_cmd.v_scale == FAIL_SAFE_V_SCALE
        assert fallback_cmd.delta_z == FAIL_SAFE_DELTA_Z_M
        assert fallback_cmd.safety_state == "FAIL_SAFE"
        assert fallback_cmd.is_fallback is True

    # Missing prediction must also fail safe
    none_pred_cmd = ctrl.compute_command(
        {"height_m": 1.75, "fatigue_index": 0.2, "velocity_variance": 0.03},
        None,
    )
    assert none_pred_cmd.is_fallback is True
    assert none_pred_cmd.v_scale == FAIL_SAFE_V_SCALE


def test_ur5e_kinematics_and_policy_ablation(
    synthetic_df: pd.DataFrame,
    trained_artifacts,
) -> None:
    prep, predictor, _ = trained_artifacts

    target_pose = np.array([0.48, 0.18, 1.12], dtype=np.float64)
    q_sol = UR5eKinematics.solve_ik_position(target_pose)
    chain = UR5eKinematics.forward_chain(q_sol)
    assert chain.shape == (7, 3)
    assert np.isfinite(chain).all()
    assert float(np.linalg.norm(chain[-1] - target_pose)) < 5e-3

    assert UR5eKinematics.minimum_jerk_scalar(0.0) == pytest.approx(0.0)
    assert UR5eKinematics.minimum_jerk_scalar(1.0) == pytest.approx(1.0)

    _, _, test_df = split_by_worker_groups(synthetic_df, seed=42)
    ablation = evaluate_control_policies(test_df, prep, predictor)
    ours = ablation.loc[ablation["Policy"] == "Adaptive Conformal HRC (Ours)"].iloc[0]
    static = ablation.loc[ablation["Policy"] == "Static Nominal (v=0.85, dZ=0)"].iloc[0]

    assert ours["Safety Violation Rate (%)"] < static["Safety Violation Rate (%)"]
    assert ours["Mean Ergonomic Z Error (cm)"] < static["Mean Ergonomic Z Error (cm)"]


def test_interactive_twin_plotly_builders(trained_artifacts) -> None:
    from app import build_3d_workcell_figure, build_telemetry_figure, run_interactive_shift

    prep, predictor, _ = trained_artifacts
    trace_df = run_interactive_shift(
        prep=prep,
        predictor=predictor,
        worker_height_m=1.82,
        fatigue_multiplier=1.1,
        fault_prob=0.05,
        n_steps=40,
    )
    assert len(trace_df) == 40
    fig_3d = build_3d_workcell_figure(trace_df.iloc[15], step_idx=15)
    fig_ts = build_telemetry_figure(trace_df, current_t=float(trace_df.iloc[15]["timestamp_s"]))
    assert len(fig_3d.data) >= 5
    assert len(fig_ts.data) >= 4

