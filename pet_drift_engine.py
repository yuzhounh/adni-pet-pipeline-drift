"""Independent analytical engine for molecular PET 6mm reprocessing, PVC shift, visual-read ROC calibration, and AT staging drift."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture


def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return float("nan")
    mat = np.column_stack([a, b])
    grand = mat.mean()
    row_m = mat.mean(axis=1)
    col_m = mat.mean(axis=0)
    ms_r = 2.0 * np.sum((row_m - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((col_m - grand) ** 2)
    ms_e = np.sum((mat - row_m[:, None] - col_m[None, :] + grand) ** 2) / (n - 1)
    denom = ms_r + ms_e + 2.0 * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if abs(denom) > 1e-12 else float("nan")


def youden_optimal_threshold(y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Compute ROC AUROC and Youden-J optimal threshold against visual-read gold standard."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auroc": float("nan"), "opt_thresh": float("nan"), "opt_sens": float("nan"), "opt_spec": float("nan"), "opt_acc": float("nan")}
    auroc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    j_idx = int(np.argmax(tpr - fpr))
    opt_t = float(thr[j_idx])
    pred = (s >= opt_t).astype(np.int64)
    return {
        "n": int(len(y)),
        "auroc": round(auroc, 4),
        "opt_thresh": round(opt_t, 4),
        "opt_sens": round(float(tpr[j_idx] * 100.0), 2),
        "opt_spec": round(float((1.0 - fpr[j_idx]) * 100.0), 2),
        "opt_acc": round(float(np.mean(pred == y) * 100.0), 2),
    }


def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=np.float64)
    p = arr.shape[1]
    med = np.zeros(p, dtype=np.float64)
    Xi = arr.copy()
    for j in range(p):
        ok = np.isfinite(arr[:, j])
        med[j] = float(np.median(arr[ok, j])) if np.any(ok) else 0.0
        Xi[~ok, j] = med[j]
    mu = Xi.mean(axis=0)
    sd = Xi.std(axis=0, ddof=0)
    sd_s = np.where(sd > 1e-12, sd, 1.0)
    return (Xi - mu) / sd_s, med, mu, sd_s


def apply_scaler(X: np.ndarray, med: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xi = np.asarray(X, dtype=np.float64).copy()
    for j in range(Xi.shape[1]):
        bad = ~np.isfinite(Xi[:, j])
        if np.any(bad):
            Xi[bad, j] = med[j]
    return (Xi - mu) / sd


def fit_predict_tau_subtypes(
    X_tr_src: np.ndarray, X_ev_src: np.ndarray, X_tr_tgt: np.ndarray, X_ev_tgt: np.ndarray,
    algo: str, k: int, seed: int = 20261123,
) -> Dict[str, np.ndarray]:
    """Fit Tau spatial clusterer on source pipeline and compare direct transfer, target Z-restd, and refitted target."""
    Xs_tr, med_s, mu_s, sd_s = fit_scaler(X_tr_src)
    Xt_tr, med_t, mu_t, sd_t = fit_scaler(X_tr_tgt)
    Xs_ev = apply_scaler(X_ev_src, med_s, mu_s, sd_s)
    Xt_ev_dir = apply_scaler(X_ev_tgt, med_s, mu_s, sd_s)
    Xt_ev_restd = fit_scaler(X_ev_tgt)[0]
    Xt_ev_tgt_sc = apply_scaler(X_ev_tgt, med_t, mu_t, sd_t)

    def _fit_one(X_in):
        if algo == "kmeans":
            est = KMeans(n_clusters=k, n_init=30, tol=1e-5, random_state=seed)
            lbl = est.fit_predict(X_in)
            cent = est.cluster_centers_
        elif algo == "gmm_diag":
            est = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, tol=1e-4, n_init=15, max_iter=300, random_state=seed)
            est.fit(X_in)
            lbl = est.predict(X_in)
            cent = est.means_
        elif algo == "ward":
            est = AgglomerativeClustering(n_clusters=k, linkage="ward")
            lbl = est.fit_predict(X_in)
            cent = np.zeros((k, X_in.shape[1]), dtype=np.float64)
            for c in range(k):
                m = lbl == c
                if np.any(m):
                    cent[c] = X_in[m].mean(axis=0)
        else:
            raise ValueError(algo)
        return est, cent

    def _pred_one(est, cent, X_q):
        if algo in ("kmeans", "gmm_diag"):
            return est.predict(X_q).astype(np.int64)
        dists = np.sum((X_q[:, None, :] - cent[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1).astype(np.int64)

    est_s, cent_s = _fit_one(Xs_tr)
    est_t, cent_t = _fit_one(Xt_tr)

    # Canonical ascending Tau burden order on source centroids (higher SUVR = higher cluster index)
    ord_s = np.argsort(cent_s.mean(axis=1), kind="mergesort")
    perm_s = np.empty_like(ord_s); perm_s[ord_s] = np.arange(k, dtype=np.int64)
    cent_s_ord = cent_s[ord_s]

    # Align refitted target centroids to source canonical centroids via Hungarian
    diff = cent_t[:, None, :] - cent_s_ord[None, :, :]
    cost = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    ri, ci = linear_sum_assignment(cost)
    perm_t = np.empty(k, dtype=np.int64); perm_t[ri] = ci

    return {
        "source_on_source": perm_s[_pred_one(est_s, cent_s, Xs_ev)],
        "A_naive_direct_transfer": perm_s[_pred_one(est_s, cent_s, Xt_ev_dir)],
        "B_target_restd_rescue": perm_s[_pred_one(est_s, cent_s, Xt_ev_restd)],
        "C_refitted_target": perm_t[_pred_one(est_t, cent_t, Xt_ev_tgt_sc)],
    }


def fit_eval_pet_logistic(
    X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, y_ev: np.ndarray,
    restd_cols: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    Xs_tr, med, mu, sd = fit_scaler(X_tr)
    Xs_ev = apply_scaler(X_ev, med, mu, sd)
    if restd_cols:
        Xt_ev, _, _, _ = fit_scaler(X_ev)
        for j in restd_cols:
            Xs_ev[:, j] = Xt_ev[:, j]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
        return {"brier": float("nan"), "auroc": float("nan"), "oe_ratio": float("nan"), "bias_pct": float("nan")}
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xs_tr, y_tr.astype(np.int64))
    p = np.clip(clf.predict_proba(Xs_ev)[:, 1], 1e-6, 1.0 - 1e-6)
    pm, ob = float(np.mean(p)), float(np.mean(y_ev))
    return {
        "brier": round(float(brier_score_loss(y_ev, p)), 4),
        "auroc": round(float(roc_auc_score(y_ev, p)), 4),
        "oe_ratio": round(float(pm / ob), 4) if ob > 1e-8 else float("nan"),
        "bias_pct": round(float((pm - ob) * 100.0), 2),
    }


def fit_eval_pet_cox(
    X_tr: np.ndarray, t_tr: np.ndarray, e_tr: np.ndarray,
    X_ev: np.ndarray, t_ev: np.ndarray, e_ev: np.ndarray,
    fnames: List[str],
    restd_cols: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    Xs_tr, med, mu, sd = fit_scaler(X_tr)
    Xs_ev = apply_scaler(X_ev, med, mu, sd)
    if restd_cols:
        Xt_ev, _, _, _ = fit_scaler(X_ev)
        for j in restd_cols:
            Xs_ev[:, j] = Xt_ev[:, j]
    if int(np.sum(e_tr)) < 10 or int(np.sum(e_ev)) < 5:
        return {"c_index": float("nan"), "oe_36m": float("nan"), "bias_36m_pct": float("nan")}
    df_tr = pd.DataFrame(Xs_tr, columns=fnames); df_tr["T"] = t_tr; df_tr["E"] = e_tr
    cph = CoxPHFitter(penalizer=0.01 / len(df_tr), l1_ratio=0.0).fit(df_tr, "T", "E", fit_options={"step_size": 0.5})
    df_ev = pd.DataFrame(Xs_ev, columns=fnames); df_ev["T"] = t_ev; df_ev["E"] = e_ev
    c_idx = float(cph.score(df_ev, scoring_method="concordance_index"))
    sf = cph.predict_survival_function(df_ev[fnames], times=[36.0])
    pred_36 = float(np.mean(1.0 - sf.iloc[0].to_numpy(float)))
    km = KaplanMeierFitter().fit(t_ev, event_observed=e_ev)
    obs_36 = float(1.0 - km.predict(36.0))
    return {
        "c_index": round(c_idx, 4),
        "oe_36m": round(float(pred_36 / obs_36), 4) if obs_36 > 1e-8 else float("nan"),
        "bias_36m_pct": round(float((pred_36 - obs_36) * 100.0), 2),
    }
