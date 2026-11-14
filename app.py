"""Interactive 3D Industry 5.0 Cobot Digital Twin & Conformal Telemetry Cockpit.

Launch locally with:
    streamlit run app.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.controller import (
    AdaptiveCobotController,
    UR5eKinematics,
    evaluate_control_policies,
)
from src.model import ModelConfig, split_by_worker_groups, train_and_evaluate_pipeline
from src.preprocessing import OnlineWindowBuffer
from src.sim_env import (
    ShiftSimulatorConfig,
    generate_telemetry_dataframe,
    stream_telemetry,
)

STATE_HEX: dict[str, str] = {
    "NOMINAL": "#10B981",
    "CAUTION": "#F59E0B",
    "HIGH_RISK": "#EF4444",
    "FAIL_SAFE": "#FB7185",
}


@st.cache_resource
def load_trained_engine(conformal_alpha: float):
    """Train and cache the dual-head conformal XGBoost pipeline."""
    base_df = generate_telemetry_dataframe(n_workers=12, steps_per_worker=250, seed=42)
    cfg = ModelConfig(conformal_alpha=conformal_alpha, random_state=42)
    prep, predictor, metrics = train_and_evaluate_pipeline(
        base_df,
        window_size=10,
        model_config=cfg,
        seed=42,
    )
    _, _, test_df = split_by_worker_groups(base_df, seed=42)
    ablation_df = evaluate_control_policies(test_df, prep, predictor)
    return prep, predictor, metrics, ablation_df


def run_interactive_shift(
    prep,
    predictor,
    worker_height_m: float,
    fatigue_multiplier: float,
    fault_prob: float,
    n_steps: int = 180,
) -> pd.DataFrame:
    """Simulate a custom operator episode and execute closed-loop HRC control."""
    sim_cfg = ShiftSimulatorConfig(
        fatigue_rate=0.024 * fatigue_multiplier,
        min_worker_height_m=worker_height_m,
        max_worker_height_m=worker_height_m,
    )
    ep_df = generate_telemetry_dataframe(
        n_workers=1,
        steps_per_worker=n_steps,
        seed=108,
        config=sim_cfg,
    )

    online_buf = OnlineWindowBuffer(prep)
    controller = AdaptiveCobotController()
    rows = []

    for clean_row, live_sample in zip(
        ep_df.to_dict(orient="records"),
        stream_telemetry(ep_df, fault_injection_prob=fault_prob, seed=77),
    ):
        x_scaled = online_buf.push_and_transform(clean_row)
        pred = predictor.predict_one(x_scaled)
        cmd = controller.compute_command(live_sample, pred)

        rows.append(
            {
                "step": int(clean_row["step"]),
                "timestamp_s": float(clean_row["timestamp_s"]),
                "height_m": float(clean_row["height_m"]),
                "fatigue_index": float(clean_row["fatigue_index"]),
                "velocity_variance": float(clean_row["velocity_variance"]),
                "separation_distance_m": float(clean_row["separation_distance_m"]),
                "risk_point": pred.risk_point,
                "risk_upper_95": pred.risk_upper_95,
                "v_scale": cmd.v_scale,
                "delta_z_cm": cmd.delta_z * 100.0,
                "target_z_m": cmd.target_z_m,
                "protective_distance_m": cmd.protective_distance_m,
                "safety_state": cmd.safety_state,
                "is_fallback": cmd.is_fallback,
            }
        )
    return pd.DataFrame(rows)


def build_3d_workcell_figure(cur: pd.Series, step_idx: int) -> go.Figure:
    """Create interactive 3D Plotly scene with 6-DOF UR5e arm, human, and safety zone."""
    state = str(cur["safety_state"])
    zone_color = STATE_HEX.get(state, "#10B981")

    tau = 0.5 * (1.0 - math.cos(0.18 * step_idx * float(cur["v_scale"])))
    s_mj = UR5eKinematics.minimum_jerk_scalar(tau)
    target_xyz = np.array(
        [0.28 + 0.22 * s_mj, -0.16 + 0.32 * s_mj, float(cur["target_z_m"])],
        dtype=np.float64,
    )
    q_sol = UR5eKinematics.solve_ik_position(target_xyz)
    chain = UR5eKinematics.forward_chain(q_sol)

    theta = np.linspace(0.0, 2.0 * math.pi, 72)
    sp_r = float(cur["protective_distance_m"])

    h_m = float(cur["height_m"])
    hx = float(np.clip(0.35 + float(cur["separation_distance_m"]) * 0.65, 0.62, 1.42))
    hy = 0.08
    hz = float(cur["target_z_m"])

    fig = go.Figure()

    # Pedestal
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, UR5eKinematics.PEDESTAL_Z_M],
            mode="lines",
            line={"color": "#64748B", "width": 12},
            name="Pedestal",
        )
    )

    # 6-DOF UR5e Kinematic Chain
    fig.add_trace(
        go.Scatter3d(
            x=chain[:, 0],
            y=chain[:, 1],
            z=chain[:, 2],
            mode="lines+markers",
            line={"color": "#38BDF8", "width": 9},
            marker={"size": 5, "color": "#E0F2FE"},
            name="UR5e 6-DOF Arm",
        )
    )

    # End-effector active state beacon
    fig.add_trace(
        go.Scatter3d(
            x=[chain[-1, 0]],
            y=[chain[-1, 1]],
            z=[chain[-1, 2]],
            mode="markers",
            marker={"size": 9, "color": zone_color},
            name=f"Tool Flange ({state})",
        )
    )

    # ISO/TS 15066 Protective Separation Ring
    fig.add_trace(
        go.Scatter3d(
            x=sp_r * np.cos(theta),
            y=sp_r * np.sin(theta),
            z=np.zeros_like(theta),
            mode="lines",
            line={"color": zone_color, "width": 6},
            name="ISO/TS 15066 Protective Zone",
        )
    )

    # Ergonomic Handoff Reference Bar
    fig.add_trace(
        go.Scatter3d(
            x=[0.42, 0.58],
            y=[-0.24, 0.24],
            z=[hz, hz],
            mode="lines",
            line={"color": "#A78BFA", "width": 5, "dash": "dash"},
            name="Ergonomic Handoff Z",
        )
    )

    # Stylized Human Operator Skeleton
    hip_z, sh_z, head_z = 0.53 * h_m, 0.82 * h_m, 0.96 * h_m
    human_x = [hx, hx, hx, None, hx, hx, hx, None, hx, 0.55 * hx + 0.45 * (hx - 0.26), hx - 0.26]
    human_y = [hy, hy, hy, None, hy - 0.11, hy, hy + 0.11, None, hy - 0.15, hy - 0.08, hy - 0.02]
    human_z = [hip_z, sh_z, head_z, None, 0.0, hip_z, 0.0, None, sh_z, 0.63 * h_m - 0.04, hz]
    fig.add_trace(
        go.Scatter3d(
            x=human_x,
            y=human_y,
            z=human_z,
            mode="lines+markers",
            line={"color": "#F8FAFC", "width": 6},
            marker={"size": 4, "color": "#F8FAFC"},
            name=f"Operator ({h_m:.2f} m)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        height=490,
        title=f"3D Workcell Digital Twin  |  State: {state}  |  S_p = {sp_r:.2f} m",
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "x": 0.01},
        scene={
            "xaxis": {"range": [-0.6, 1.55], "title": "X (m)"},
            "yaxis": {"range": [-0.9, 0.9], "title": "Y (m)"},
            "zaxis": {"range": [0.0, 2.0], "title": "Z (m)"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.25, "y": 1.0, "z": 1.05},
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 0.95}},
        },
    )
    return fig


def build_telemetry_figure(trace_df: pd.DataFrame, current_t: float) -> go.Figure:
    """Create synchronized multi-panel Plotly telemetry chart."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Fatigue & Split-Conformal Risk Ribbon",
            "ISO/TS 15066 Cobot Velocity Scaling (v_scale)",
            "Ergonomic Vertical Handoff Offset (ΔZ cm)",
        ),
    )

    t = trace_df["timestamp_s"]
    fig.add_trace(
        go.Scatter(
            x=t,
            y=trace_df["risk_upper_95"],
            line={"color": "#A78BFA", "width": 1.8},
            name="Conformal Upper Risk",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=trace_df["risk_point"],
            fill="tonexty",
            fillcolor="rgba(167, 139, 250, 0.25)",
            line={"color": "#818CF8", "width": 1.2, "dash": "dot"},
            name="Point Risk Estimate",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=trace_df["fatigue_index"],
            line={"color": "#38BDF8", "width": 1.6},
            name="Observed Fatigue",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=trace_df["v_scale"],
            line={"color": "#10B981", "width": 2.2},
            name="v_scale",
        ),
        row=2,
        col=1,
    )
    fb = trace_df[trace_df["is_fallback"]]
    if not fb.empty:
        fig.add_trace(
            go.Scatter(
                x=fb["timestamp_s"],
                y=fb["v_scale"],
                mode="markers",
                marker={"color": "#FB7185", "size": 8, "symbol": "x"},
                name="Fail-Safe Trigger",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=trace_df["delta_z_cm"],
            line={"color": "#F59E0B", "width": 2.0},
            name="ΔZ (cm)",
        ),
        row=3,
        col=1,
    )

    fig.add_vline(x=current_t, line_width=1.5, line_dash="dash", line_color="#F8FAFC")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#111827",
        height=490,
        margin={"l": 12, "r": 12, "t": 36, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0.0},
    )
    fig.update_yaxes(range=[0.0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0.0, 1.05], row=2, col=1)
    fig.update_yaxes(range=[-26.0, 26.0], row=3, col=1)
    fig.update_xaxes(title_text="Shift Time (s)", row=3, col=1)
    return fig


def main() -> None:
    st.set_page_config(
        page_title="AdaptiveCobot-HRC Digital Twin",
        layout="wide",
    )

    st.title("AdaptiveCobot-HRC — Industry 5.0 Supervisory Digital Twin")
    st.caption(
        "Biomechanical Fatigue ODE  •  Causal O(1) Ring-Buffer Features  •  "
        "Monotonic Split-Conformal XGBoost  •  ISO/TS 15066 Cobot Control"
    )

    with st.sidebar:
        st.header("Workcell & Operator Parameters")
        worker_height = st.slider("Operator Stature (m)", 1.52, 1.96, 1.84, 0.01)
        fatigue_mult = st.slider("Task Workload Intensity", 0.50, 1.60, 1.10, 0.05)
        conf_level = st.slider("Conformal Coverage Target (1 - α)", 0.85, 0.99, 0.95, 0.01)
        fault_prob = st.slider("Sensor Fault Injection Probability", 0.00, 0.15, 0.03, 0.01)

    prep, predictor, metrics, ablation_df = load_trained_engine(round(1.0 - conf_level, 2))
    trace_df = run_interactive_shift(
        prep=prep,
        predictor=predictor,
        worker_height_m=worker_height,
        fatigue_multiplier=fatigue_mult,
        fault_prob=fault_prob,
    )

    with st.sidebar:
        step_idx = st.slider("Inspect Shift Step", 0, len(trace_df) - 1, 95, 1)

    cur = trace_df.iloc[step_idx]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Commanded v_scale", f"{cur['v_scale']:.2f}", f"State: {cur['safety_state']}")
    c2.metric("Handoff Offset ΔZ", f"{cur['delta_z_cm']:+.1f} cm", f"Z = {cur['target_z_m']:.2f} m")
    c3.metric("Held-Out R² / Macro-F1", f"{metrics.r2:.3f} / {metrics.macro_f1:.3f}")
    c4.metric("Empirical Coverage", f"{metrics.conformal_coverage * 100.0:.1f}%")
    c5.metric("CPU Control Latency", f"{metrics.single_sample_latency_ms:.3f} ms")

    left_col, right_col = st.columns([1.05, 1.20])
    with left_col:
        st.plotly_chart(build_3d_workcell_figure(cur, step_idx), use_container_width=True)
    with right_col:
        st.plotly_chart(
            build_telemetry_figure(trace_df, float(cur["timestamp_s"])),
            use_container_width=True,
        )

    st.subheader("Held-Out Operator Policy Benchmark")
    st.dataframe(ablation_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
