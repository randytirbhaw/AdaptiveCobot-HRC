"""Monotonic split-conformal dual-head XGBoost predictor for HRC safety.

Combines:
1. Monotonic XGBRegressor enforcing non-decreasing ergonomic risk with respect
   to fatigue, kinematic jitter, posture mismatch, and proximity hazard.
2. Finite-sample split conformal calibration providing distribution-free
   one-sided upper confidence bounds (95% default) on unseen operators.
3. 3-tier XGBClassifier predicting supervisory safety regimes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import TelemetryPreprocessor
from src.sim_env import SafetyTier

# Feature order matches src.preprocessing.ENGINEERED_FEATURE_NAMES:
# (height_m, ergonomic_reach_abs_m, fatigue_index, fatigue_roll_mean,
#  fatigue_roll_std, fatigue_delta, velocity_variance, vel_var_roll_mean,
#  vel_var_roll_max, proximity_hazard)
MONOTONE_CONSTRAINTS: tuple[int, ...] = (0, 1, 1, 1, 0, 0, 1, 1, 1, 1)


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and conformal calibration settings."""

    n_estimators: int = 180
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    early_stopping_rounds: int = 20
    conformal_alpha: float = 0.05
    random_state: int = 42


@dataclass(frozen=True)
class RiskPrediction:
    """Point estimate, conformal upper risk bound, and discrete tier output."""

    risk_point: float
    risk_upper_95: float
    conformal_margin: float
    safety_tier: int
    tier_probabilities: tuple[float, float, float]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Held-out operator generalization and calibration metrics."""

    rmse: float
    mae: float
    r2: float
    macro_f1: float
    high_risk_recall: float
    conformal_coverage: float
    conformal_quantile: float
    single_sample_latency_ms: float


def split_by_worker_groups(
    df: pd.DataFrame,
    train_frac: float = 0.65,
    cal_frac: float = 0.18,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition multi-episode telemetry strictly by operator ``worker_id``."""
    workers = df["worker_id"].unique().copy()
    if len(workers) < 3:
        raise ValueError("Need at least 3 distinct worker_id groups for train/cal/test split.")

    rng = np.random.default_rng(seed)
    rng.shuffle(workers)

    n_total = len(workers)
    n_train = max(1, int(round(n_total * train_frac)))
    n_cal = max(1, int(round(n_total * cal_frac)))
    if n_train + n_cal >= n_total:
        n_train = max(1, n_total - 2)
        n_cal = 1

    train_ids = set(workers[:n_train].tolist())
    cal_ids = set(workers[n_train : n_train + n_cal].tolist())
    test_ids = set(workers[n_train + n_cal :].tolist())

    return (
        df[df["worker_id"].isin(train_ids)].reset_index(drop=True),
        df[df["worker_id"].isin(cal_ids)].reset_index(drop=True),
        df[df["worker_id"].isin(test_ids)].reset_index(drop=True),
    )


class AdaptiveHRCPredictor:
    """Dual-head monotonic XGBoost model with split-conformal safety calibration."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._regressor = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            early_stopping_rounds=self.config.early_stopping_rounds,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            tree_method="hist",
            objective="reg:squarederror",
            n_jobs=4,
            random_state=self.config.random_state,
        )
        self._classifier = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            early_stopping_rounds=self.config.early_stopping_rounds,
            tree_method="hist",
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=4,
            random_state=self.config.random_state,
        )
        self.conformal_quantile: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_risk_train: np.ndarray,
        y_tier_train: np.ndarray,
        X_cal: np.ndarray,
        y_risk_cal: np.ndarray,
        y_tier_cal: np.ndarray,
    ) -> AdaptiveHRCPredictor:
        """Train both heads and calibrate one-sided split-conformal upper quantile."""
        self._regressor.fit(
            X_train,
            y_risk_train,
            eval_set=[(X_cal, y_risk_cal)],
            verbose=False,
        )
        self._classifier.fit(
            X_train,
            y_tier_train,
            eval_set=[(X_cal, y_tier_cal)],
            verbose=False,
        )

        # Finite-sample one-sided conformal upper bound on calibration residuals
        cal_preds = np.clip(self._regressor.predict(X_cal), 0.0, 1.0)
        residuals = y_risk_cal - cal_preds
        n_cal = len(residuals)
        q_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - self.config.conformal_alpha)) / n_cal)
        self.conformal_quantile = float(max(0.0, float(np.quantile(residuals, q_level))))
        self.is_fitted = True
        return self

    def predict_arrays(
        self,
        X_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized batch inference returning ``(risk_point, risk_upper, tier, probs)``."""
        if not self.is_fitted:
            raise RuntimeError("AdaptiveHRCPredictor must be fitted before inference.")

        X_arr = np.atleast_2d(np.asarray(X_scaled, dtype=np.float32))
        reg_raw = self._regressor.get_booster().inplace_predict(X_arr)
        risk_point = np.clip(np.asarray(reg_raw, dtype=np.float64), 0.0, 1.0)
        risk_upper = np.clip(risk_point + self.conformal_quantile, 0.0, 1.0)

        clf_probs = np.asarray(
            self._classifier.get_booster().inplace_predict(X_arr),
            dtype=np.float64,
        )
        if clf_probs.ndim == 1:
            clf_probs = clf_probs.reshape(-1, 3)
        tiers = np.argmax(clf_probs, axis=1).astype(np.int64)
        return risk_point, risk_upper, tiers, clf_probs

    def predict_one(self, x_scaled: np.ndarray) -> RiskPrediction:
        """Low-latency single-sample prediction for real-time control loops."""
        risk_pt, risk_up, tiers, probs = self.predict_arrays(x_scaled)
        p_row = probs[0]
        return RiskPrediction(
            risk_point=float(risk_pt[0]),
            risk_upper_95=float(risk_up[0]),
            conformal_margin=self.conformal_quantile,
            safety_tier=int(tiers[0]),
            tier_probabilities=(float(p_row[0]), float(p_row[1]), float(p_row[2])),
        )

    def predict_batch(self, X_scaled: np.ndarray) -> list[RiskPrediction]:
        """Predict structured ``RiskPrediction`` records for a feature matrix."""
        risk_pt, risk_up, tiers, probs = self.predict_arrays(X_scaled)
        return [
            RiskPrediction(
                risk_point=float(risk_pt[i]),
                risk_upper_95=float(risk_up[i]),
                conformal_margin=self.conformal_quantile,
                safety_tier=int(tiers[i]),
                tier_probabilities=(
                    float(probs[i, 0]),
                    float(probs[i, 1]),
                    float(probs[i, 2]),
                ),
            )
            for i in range(len(risk_pt))
        ]

    def evaluate(
        self,
        X_test: np.ndarray,
        y_risk_test: np.ndarray,
        y_tier_test: np.ndarray,
    ) -> EvaluationMetrics:
        """Compute regression, classification, conformal coverage, and latency metrics."""
        risk_pt, risk_up, pred_tiers, _ = self.predict_arrays(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_risk_test, risk_pt)))
        mae = float(mean_absolute_error(y_risk_test, risk_pt))
        r2 = float(r2_score(y_risk_test, risk_pt))
        macro_f1 = float(f1_score(y_tier_test, pred_tiers, average="macro", zero_division=0))

        high_mask = y_tier_test == int(SafetyTier.HIGH_RISK)
        if np.any(high_mask):
            high_recall = float(
                recall_score(
                    y_tier_test == int(SafetyTier.HIGH_RISK),
                    pred_tiers == int(SafetyTier.HIGH_RISK),
                    zero_division=0,
                )
            )
        else:
            high_recall = 1.0

        coverage = float(np.mean(y_risk_test <= risk_up))

        sample_vec = X_test[:1]
        # Warmup booster cache before timing single-step control latency
        _ = self.predict_one(sample_vec)
        n_timing_iters = 100
        t0 = time.perf_counter()
        for _ in range(n_timing_iters):
            _ = self.predict_one(sample_vec)
        latency_ms = ((time.perf_counter() - t0) / n_timing_iters) * 1000.0

        return EvaluationMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            macro_f1=macro_f1,
            high_risk_recall=high_recall,
            conformal_coverage=coverage,
            conformal_quantile=self.conformal_quantile,
            single_sample_latency_ms=latency_ms,
        )

    def save(self, path: str | Path) -> None:
        """Serialize fitted predictor to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted AdaptiveHRCPredictor.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> AdaptiveHRCPredictor:
        """Load serialized predictor from disk."""
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected AdaptiveHRCPredictor artifact, got {type(loaded)!r}")
        return loaded


def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    window_size: int = 10,
    model_config: ModelConfig | None = None,
    seed: int = 42,
) -> tuple[TelemetryPreprocessor, AdaptiveHRCPredictor, EvaluationMetrics]:
    """End-to-end leakage-free group split, preprocessing, training, and evaluation."""
    train_df, cal_df, test_df = split_by_worker_groups(df, seed=seed)
    preprocessor = TelemetryPreprocessor(window_size=window_size)

    X_train = preprocessor.fit_transform(train_df)
    X_cal = preprocessor.transform(cal_df)
    X_test = preprocessor.transform(test_df)

    predictor = AdaptiveHRCPredictor(config=model_config)
    predictor.fit(
        X_train=X_train,
        y_risk_train=train_df["risk_score"].to_numpy(dtype=np.float64),
        y_tier_train=train_df["safety_tier"].to_numpy(dtype=np.int64),
        X_cal=X_cal,
        y_risk_cal=cal_df["risk_score"].to_numpy(dtype=np.float64),
        y_tier_cal=cal_df["safety_tier"].to_numpy(dtype=np.int64),
    )

    metrics = predictor.evaluate(
        X_test=X_test,
        y_risk_test=test_df["risk_score"].to_numpy(dtype=np.float64),
        y_tier_test=test_df["safety_tier"].to_numpy(dtype=np.int64),
    )
    return preprocessor, predictor, metrics
