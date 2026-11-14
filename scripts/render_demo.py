"""Generate deterministic synthetic dataset, ablation report, and dark-mode hero GIF."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from src.controller import (
    AdaptiveCobotController,
    UR5eKinematics,
    evaluate_control_policies,
)
from src.model import split_by_worker_groups, train_and_evaluate_pipeline
from src.preprocessing import OnlineWindowBuffer
from src.sim_env import ShiftSimulatorConfig, generate_telemetry_dataframe

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_CSV_PATH = ROOT_DIR / "data" / "synthetic_telemetry.csv"
HERO_GIF_PATH = ROOT_DIR / "assets" / "demo.gif"

BG_DARK = "#0B0F19"
PANEL_BG = "#111827"
GRID_COLOR = "#1F2937"
TEXT_PRIMARY = "#F3F4F6"
TEXT_MUTED = "#9CA3AF"
ACCENT_CYAN = "#38BDF8"
ACCENT_VIOLET = "#A78BFA"
ACCENT_EMERALD = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"

STATE_COLORS: dict[str, str] = {
    "NOMINAL": ACCENT_EMERALD,
    "CAUTION": ACCENT_AMBER,
    "HIGH_RISK": ACCENT_RED,
    "FAIL_SAFE": "#FB7185",
}


def _human_skeleton_segments(
    x_pos: float,
    y_pos: float,
    height_m: float,
    reach_x: float,
    reach_z: float,
) -> list[np.ndarray]:
    """Construct stylized 3D human operator limb polylines for workspace rendering."""
    hip_z = 0.53 * height_m
    shoulder_z = 0.82 * height_m
    head_z = 0.96 * height_m

    spine = np.array([[x_pos, y_pos, hip_z], [x_pos, y_pos, shoulder_z], [x_pos, y_pos, head_z]])
    legs = np.array(
        [
            [x_pos, y_pos - 0.11, 0.0],
            [x_pos, y_pos, hip_z],
            [x_pos, y_pos + 0.11, 0.0],
        ]
    )
    elbow_x = 0.55 * x_pos + 0.45 * reach_x
    elbow_z = 0.63 * height_m - 0.04
    right_arm = np.array(
        [
            [x_pos, y_pos - 0.16, shoulder_z],
            [elbow_x, y_pos - 0.08, elbow_z],
            [reach_x, y_pos - 0.02, reach_z],
        ]
    )
    left_arm = np.array(
        [
            [x_pos, y_pos + 0.16, shoulder_z],
            [x_pos - 0.12, y_pos + 0.18, 0.62 * height_m],
        ]
    )
    return [spine, legs, right_arm, left_arm]


def build_episode_trace(episode_df: pd.DataFrame, prep, predictor) -> pd.DataFrame:
    """Simulate closed-loop runtime execution with a mid-shift sensor dropout test."""
    buf = OnlineWindowBuffer(prep)
    ctrl = AdaptiveCobotController()
    rows = []

    records = episode_df.reset_index(drop=True).to_dict(orient="records")
    fault_window = {92, 93, 94}

    for idx, rec in enumerate(records):
        x_scaled = buf.push_and_transform(rec)
        pred = predictor.predict_one(x_scaled)

        live_input = dict(rec)
        if idx in fault_window:
            live_input["fatigue_index"] = float("nan")

        cmd = ctrl.compute_command(live_input, pred)
        rows.append(
            {
                "step": int(rec["step"]),
                "timestamp_s": float(rec["timestamp_s"]),
                "height_m": float(rec["height_m"]),
                "fatigue_index": float(rec["fatigue_index"]),
                "velocity_variance": float(rec["velocity_variance"]),
                "separation_distance_m": float(rec["separation_distance_m"]),
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


def render_hero_gif(trace_df: pd.DataFrame, output_path: Path) -> None:
    """Render sleek dark-mode 3D cobot workcell + live conformal telemetry GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": BG_DARK,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": GRID_COLOR,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_MUTED,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "font.size": 9,
        }
    )

    fig = plt.figure(figsize=(12.2, 6.4), dpi=105)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.12, 1.28],
        hspace=0.36,
        wspace=0.20,
        left=0.04,
        right=0.97,
        top=0.89,
        bottom=0.08,
    )

    fig.suptitle(
        "AdaptiveCobot-HRC  |  ISO/TS 15066 Conformal XGBoost Digital Twin",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
        y=0.96,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_risk = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_risk)
    ax_dz = fig.add_subplot(gs[2, 1], sharex=ax_risk)

    frame_indices = np.linspace(0, len(trace_df) - 1, 56, dtype=int)
    t_all = trace_df["timestamp_s"].to_numpy()
    t_max = float(t_all[-1])

    theta_ring = np.linspace(0.0, 2.0 * math.pi, 64)
    q_prev = np.array([0.0, -1.15, 1.55, -1.95, -1.57, 0.0], dtype=np.float64)
    q_trajectory: list[np.ndarray] = []

    # Precompute smooth 6-DOF cobot joint trajectories for all animation frames
    sim_phase = 0.0
    for f_idx in frame_indices:
        row = trace_df.iloc[int(f_idx)]
        sim_phase += 0.24 * float(row["v_scale"])
        tau = 0.5 * (1.0 - math.cos(sim_phase))
        s_mj = UR5eKinematics.minimum_jerk_scalar(tau)
        ee_target = np.array(
            [
                0.26 + 0.24 * s_mj,
                -0.18 + 0.36 * s_mj,
                float(row["target_z_m"]),
            ],
            dtype=np.float64,
        )
        q_prev = UR5eKinematics.solve_ik_position(ee_target, q_init=q_prev)
        q_trajectory.append(q_prev.copy())

    def _style_2d_axis(ax, title: str, ylabel: str, ylim: tuple[float, float]) -> None:
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, fontsize=9.5, color=TEXT_PRIMARY, loc="left", pad=4)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xlim(0.0, t_max)
        ax.set_ylim(*ylim)
        ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.85)

    def update(anim_i: int) -> list:
        idx = int(frame_indices[anim_i])
        sub = trace_df.iloc[: idx + 1]
        cur = trace_df.iloc[idx]
        state = str(cur["safety_state"])
        state_col = STATE_COLORS.get(state, ACCENT_EMERALD)

        # --- 3D Collaborative Workcell Panel ---
        ax3d.cla()
        ax3d.set_facecolor(BG_DARK)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor(GRID_COLOR)
        ax3d.yaxis.pane.set_edgecolor(GRID_COLOR)
        ax3d.zaxis.pane.set_edgecolor(GRID_COLOR)
        ax3d.grid(True, color=GRID_COLOR, alpha=0.4)

        # Cobot pedestal + 6-DOF UR5e kinematic chain
        ax3d.plot([0.0, 0.0], [0.0, 0.0], [0.0, UR5eKinematics.PEDESTAL_Z_M], color="#64748B", lw=6)
        chain = UR5eKinematics.forward_chain(q_trajectory[anim_i])
        ax3d.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=ACCENT_CYAN, lw=4.2, marker="o", ms=4.5)
        ax3d.scatter([chain[-1, 0]], [chain[-1, 1]], [chain[-1, 2]], color=state_col, s=55, zorder=10)

        # Ergonomic handoff target plane
        hz = float(cur["target_z_m"])
        ax3d.plot([0.42, 0.58], [-0.22, 0.22], [hz, hz], color=ACCENT_VIOLET, ls="--", lw=1.6, alpha=0.9)

        # Dynamic ISO/TS 15066 protective separation zone on floor
        sp_r = float(cur["protective_distance_m"])
        ax3d.plot(
            sp_r * np.cos(theta_ring),
            sp_r * np.sin(theta_ring),
            np.zeros_like(theta_ring),
            color=state_col,
            lw=2.2,
            alpha=0.95,
        )

        # Human operator articulated figure
        human_x = float(np.clip(0.35 + float(cur["separation_distance_m"]) * 0.65, 0.62, 1.38))
        h_m = float(cur["height_m"])
        for seg in _human_skeleton_segments(human_x, 0.08, h_m, human_x - 0.26, hz):
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#E2E8F0", lw=2.6)
        ax3d.scatter([human_x], [0.08], [0.98 * h_m], color="#E2E8F0", s=52)

        ax3d.set_xlim(-0.55, 1.50)
        ax3d.set_ylim(-0.85, 0.85)
        ax3d.set_zlim(0.0, 1.95)
        ax3d.view_init(elev=22, azim=-52)
        ax3d.set_xlabel("X (m)", labelpad=2)
        ax3d.set_ylabel("Y (m)", labelpad=2)
        ax3d.set_zlabel("Z (m)", labelpad=2)
        ax3d.set_title(
            f"6-DOF UR5e Workcell  |  State: {state}  |  v_scale={cur['v_scale']:.2f}  |  ΔZ={cur['delta_z_cm']:+.1f} cm",
            fontsize=9.5,
            color=state_col,
            pad=8,
        )

        # --- Right Panel 1: Fatigue & Conformal Risk ---
        ax_risk.cla()
        _style_2d_axis(ax_risk, "Biomechanical Fatigue & 95% Conformal Risk Bound", "Score", (0.0, 1.05))
        t_sub = sub["timestamp_s"].to_numpy()
        ax_risk.fill_between(
            t_sub,
            sub["risk_point"].to_numpy(),
            sub["risk_upper_95"].to_numpy(),
            color=ACCENT_VIOLET,
            alpha=0.30,
            label="95% Conformal Margin",
        )
        ax_risk.plot(t_sub, sub["fatigue_index"], color=ACCENT_CYAN, lw=1.5, alpha=0.85, label="Fatigue Index")
        ax_risk.plot(t_sub, sub["risk_upper_95"], color=ACCENT_VIOLET, lw=1.8, label="Upper Risk r_0.95")
        ax_risk.axhline(0.70, color=ACCENT_RED, ls=":", lw=1.0)
        ax_risk.axhline(0.35, color=ACCENT_AMBER, ls=":", lw=1.0)
        ax_risk.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=3)

        # --- Right Panel 2: Cobot Velocity Scaling ---
        ax_vel.cla()
        _style_2d_axis(ax_vel, "ISO/TS 15066 Slew-Limited Velocity Scaling (with NaN Fault Test)", "v_scale", (0.0, 1.08))
        ax_vel.plot(t_sub, sub["v_scale"], color=ACCENT_EMERALD, lw=2.0, label="Commanded v_scale")
        ax_vel.axhline(0.10, color=ACCENT_RED, ls="--", lw=0.9, alpha=0.7, label="Fail-Safe Min (0.10)")
        fb_sub = sub[sub["is_fallback"]]
        if not fb_sub.empty:
            ax_vel.scatter(
                fb_sub["timestamp_s"],
                fb_sub["v_scale"],
                color="#FB7185",
                s=28,
                zorder=5,
                label="NaN Fault Fallback",
            )
        ax_vel.legend(loc="lower left", frameon=False, fontsize=7.5, ncol=3)

        # --- Right Panel 3: Anthropometric Handoff Offset ---
        ax_dz.cla()
        _style_2d_axis(ax_dz, "Ergonomic Vertical Handoff Envelope Offset (ΔZ)", "ΔZ (cm)", (-26.0, 26.0))
        ax_dz.plot(t_sub, sub["delta_z_cm"], color=ACCENT_AMBER, lw=1.9)
        ax_dz.axhline(0.0, color=TEXT_MUTED, ls=":", lw=0.9)
        ax_dz.set_xlabel("Shift Time (s)", fontsize=8.5)

        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=95, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=11))
    plt.close(fig)


def main() -> None:
    DATA_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_telemetry_dataframe(n_workers=12, steps_per_worker=250, seed=42)
    df.to_csv(DATA_CSV_PATH, index=False)

    prep, predictor, metrics = train_and_evaluate_pipeline(df, window_size=10, seed=42)
    _, _, test_df = split_by_worker_groups(df, seed=42)
    ablation_df = evaluate_control_policies(test_df, prep, predictor)

    # Unseen 90th-percentile stature operator episode showcasing anthropometric +ΔZ & fatigue relief
    demo_cfg = ShiftSimulatorConfig(
        fatigue_rate=0.028,
        min_worker_height_m=1.89,
        max_worker_height_m=1.89,
    )
    demo_episode = generate_telemetry_dataframe(
        n_workers=1,
        steps_per_worker=250,
        seed=2026,
        config=demo_cfg,
    )
    trace_df = build_episode_trace(demo_episode, prep, predictor)
    render_hero_gif(trace_df, HERO_GIF_PATH)

    summary = {
        "dataset_rows": int(len(df)),
        "n_workers": int(df["worker_id"].nunique()),
        "metrics": {
            "rmse": round(metrics.rmse, 4),
            "mae": round(metrics.mae, 4),
            "r2": round(metrics.r2, 4),
            "macro_f1": round(metrics.macro_f1, 4),
            "high_risk_recall": round(metrics.high_risk_recall, 4),
            "conformal_coverage_pct": round(metrics.conformal_coverage * 100.0, 2),
            "conformal_quantile": round(metrics.conformal_quantile, 4),
            "single_sample_latency_ms": round(metrics.single_sample_latency_ms, 4),
        },
        "ablation": ablation_df.to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
