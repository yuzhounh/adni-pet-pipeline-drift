"""Pure-synthetic unit tests for the molecular PET pipeline drift and visual-read ROC engine (zero private data required)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pet_drift_engine import (
    fit_eval_pet_cox,
    fit_eval_pet_logistic,
    fit_predict_tau_subtypes,
    icc_a1,
    youden_optimal_threshold,
)


def test_youden_threshold_shifts_with_6mm_offset():
    rng = np.random.default_rng(7)
    y = np.array([0] * 150 + [1] * 100)
    s_old = np.concatenate([rng.normal(1.12, 0.06, 150), rng.normal(1.48, 0.14, 100)])
    # Simulate +0.08 upward shift in 6mm Tau SUVR
    s_6mm = s_old + 0.08
    r_old = youden_optimal_threshold(y, s_old)
    r_new = youden_optimal_threshold(y, s_6mm)
    assert r_old["auroc"] > 0.95 and abs(r_old["auroc"] - r_new["auroc"]) < 1e-4
    assert abs((r_new["opt_thresh"] - r_old["opt_thresh"]) - 0.08) < 0.02
    assert icc_a1(s_old, s_6mm) < 0.96


def test_tau_spatial_subtyping_and_restd_rescue():
    rng = np.random.default_rng(19)
    X_tr = np.vstack([
        rng.normal([1.10, 1.08, 1.05], 0.05, (90, 3)),
        rng.normal([1.35, 1.25, 1.15], 0.06, (90, 3)),
        rng.normal([1.75, 1.65, 1.50], 0.08, (90, 3)),
    ])
    X_ev = np.vstack([
        rng.normal([1.10, 1.08, 1.05], 0.05, (45, 3)),
        rng.normal([1.35, 1.25, 1.15], 0.06, (45, 3)),
        rng.normal([1.75, 1.65, 1.50], 0.08, (45, 3)),
    ])
    # Simulate strong +0.20 PVC upward shift
    res = fit_predict_tau_subtypes(X_tr, X_ev, X_tr + 0.20, X_ev + 0.20, "kmeans", 3, seed=42)
    flip_dir = np.mean(res["A_naive_direct_transfer"] != res["source_on_source"])
    flip_re = np.mean(res["B_target_restd_rescue"] != res["source_on_source"])
    assert flip_dir > 0.20 and flip_re < 0.05


def test_pet_logistic_and_cox_shift_rescue():
    rng = np.random.default_rng(55)
    X_tr = rng.normal(0, 1, (300, 3))
    X_ev_old = rng.normal(0, 1, (150, 3))
    X_ev_pvc = X_ev_old.copy(); X_ev_pvc[:, 2] += 1.2
    b = np.array([0.3, -0.1, 1.1])
    y_tr = rng.binomial(1, 1.0 / (1.0 + np.exp(-(X_tr @ b))))
    y_ev = rng.binomial(1, 1.0 / (1.0 + np.exp(-(X_ev_old @ b))))
    r_dir = fit_eval_pet_logistic(X_tr, y_tr, X_ev_pvc, y_ev, restd_cols=None)
    r_res = fit_eval_pet_logistic(X_tr, y_tr, X_ev_pvc, y_ev, restd_cols=[2])
    assert r_dir["oe_ratio"] > 1.35 and abs(r_res["oe_ratio"] - 1.0) < abs(r_dir["oe_ratio"] - 1.0)

    t_tr = np.minimum(rng.exponential(35.0 * np.exp(-X_tr @ b)), 72.0) + 0.5
    e_tr = (t_tr < 55.0).astype(int)
    t_ev = np.minimum(rng.exponential(35.0 * np.exp(-X_ev_old @ b)), 72.0) + 0.5
    e_ev = (t_ev < 55.0).astype(int)
    c_dir = fit_eval_pet_cox(X_tr, t_tr, e_tr, X_ev_pvc, t_ev, e_ev, ["age", "sex", "tau"], restd_cols=None)
    c_res = fit_eval_pet_cox(X_tr, t_tr, e_tr, X_ev_pvc, t_ev, e_ev, ["age", "sex", "tau"], restd_cols=[2])
    assert c_dir["oe_36m"] > 1.35 and abs(c_res["oe_36m"] - 1.0) < abs(c_dir["oe_36m"] - 1.0)


def run_all() -> None:
    for fn in (test_youden_threshold_shifts_with_6mm_offset, test_tau_spatial_subtyping_and_restd_rescue, test_pet_logistic_and_cox_shift_rescue):
        fn()
        print(f"[PASS] {fn.__name__}")
    print("All 3/3 molecular PET drift synthetic unit tests passed.")


if __name__ == "__main__":
    run_all()
