from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from .config import DEFAULT_THRESHOLD, LGBM_PARAMS, MODEL_DIR
from .features import FEATURE_NAMES
from .metrics import entity_f05


def train_lgbm(X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    fit_kw = {}
    if X_val is not None:
        fit_kw["eval_set"] = [(X_val, y_val)]
    model.fit(X, y, **fit_kw)
    return model


def predict_proba(model: lgb.LGBMClassifier, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0,))
    return model.predict_proba(X)[:, 1]


def sweep_threshold(
    scores_by_s1: dict[str, list[tuple[str, float]]],
    gold: dict[str, list[str]],
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick threshold that maximises macro F0.5, not F1."""
    grid = grid if grid is not None else np.linspace(0.35, 0.95, 25)
    best_t, best_s = DEFAULT_THRESHOLD, -1.0
    for t in grid:
        preds = {
            sid: [oid for oid, p in pairs if p >= t]
            for sid, pairs in scores_by_s1.items()
        }
        # include gold keys that had no candidates
        for k in gold:
            preds.setdefault(k, [])
        s = sum(entity_f05(preds[k], gold[k]) for k in gold) / max(len(gold), 1)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


def save_bundle(model, threshold: float, path: Path | None = None) -> Path:
    path = path or (MODEL_DIR / "matcher.joblib")
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURE_NAMES}, path)
    return path


def load_bundle(path: Path | None = None) -> dict:
    path = path or (MODEL_DIR / "matcher.joblib")
    return joblib.load(path)
