"""End-to-end ADNI molecular PET 6mm reprocessing, PVC shift, visual-read ROC calibration, AT staging, and clinical prognosis pipeline."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
import hashlib
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pet_drift_engine import (
    fit_eval_pet_cox,
    fit_eval_pet_logistic,
    fit_predict_tau_subtypes,
    icc_a1,
    youden_optimal_threshold,
)


ADNI_PHASES = ("ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4")
TAU_8_ROI_PAIRS = [
    ("entorhinal", "CTX_LH_ENTORHINAL_SUVR", "CTX_RH_ENTORHINAL_SUVR"),
    ("parahippocampal", "CTX_LH_PARAHIPPOCAMPAL_SUVR", "CTX_RH_PARAHIPPOCAMPAL_SUVR"),
    ("fusiform", "CTX_LH_FUSIFORM_SUVR", "CTX_RH_FUSIFORM_SUVR"),
    ("inferiortemporal", "CTX_LH_INFERIORTEMPORAL_SUVR", "CTX_RH_INFERIORTEMPORAL_SUVR"),
    ("middletemporal", "CTX_LH_MIDDLETEMPORAL_SUVR", "CTX_RH_MIDDLETEMPORAL_SUVR"),
    ("posteriorcingulate", "CTX_LH_POSTERIORCINGULATE_SUVR", "CTX_RH_POSTERIORCINGULATE_SUVR"),
    ("precuneus", "CTX_LH_PRECUNEUS_SUVR", "CTX_RH_PRECUNEUS_SUVR"),
    ("superiorfrontal", "CTX_LH_SUPERIORFRONTAL_SUVR", "CTX_RH_SUPERIORFRONTAL_SUVR"),
]
TAU_8_COLS = [r[0] for r in TAU_8_ROI_PAIRS]
CLIN_5 = ["age", "sex_male", "educ", "mmse", "cdrsb"]


def _det_holdout(ids: pd.Series, strata: pd.Series, seed_tag: str, frac: float = 0.30) -> np.ndarray:
    df = pd.DataFrame({"id": ids.astype(str).values, "s": strata.astype(str).values})
    df["_h"] = [hashlib.sha256(f"{seed_tag}|{x}".encode("utf-8")).hexdigest() for x in df["id"]]
    is_eval = np.zeros(len(df), dtype=bool)
    for _, grp in df.groupby("s", sort=True):
        ord_i = grp.sort_values(["_h", "id"]).index.values
        n_ev = int(np.floor(len(ord_i) * frac + 0.5))
        if len(ord_i) >= 2 and n_ev == 0:
            n_ev = 1
        is_eval[ord_i[:n_ev]] = True
    return is_eval


def _norm_key(df: pd.DataFrame, dcol: str) -> pd.DataFrame:
    d = df.copy()
    d["RID"] = pd.to_numeric(d["RID"], errors="coerce")
    d["scan_dt"] = pd.to_datetime(d[dcol], errors="coerce")
    d = d.dropna(subset=["RID", "scan_dt"]).copy()
    d["RID"] = d["RID"].astype(int)
    d["_k"] = d["RID"].astype(str) + "|" + d["scan_dt"].dt.strftime("%Y-%m-%d")
    return d


def build_clinical_anchor(adni_raw_dir: str, adni_extract_dir: str) -> pd.DataFrame:
    v26 = os.path.join(adni_extract_dir, "v2026_10", "csv")
    adsl = pd.read_csv(os.path.join(v26, "ADSL.csv"), low_memory=False)
    adsl["RID"] = pd.to_numeric(adsl["SUBJID"], errors="coerce")
    enrl = adsl[(adsl["ENRLFL"] == "Y") & (adsl["ORIGPROT"].isin(ADNI_PHASES))].dropna(subset=["RID"]).drop_duplicates("RID").copy()
    enrl["RID"] = enrl["RID"].astype(int)

    dx_cands = sorted(glob.glob(os.path.join(adni_raw_dir, "DXSUM*.csv")))
    if not dx_cands:
        raise FileNotFoundError(f"No DXSUM*.csv in {adni_raw_dir}")
    dx = pd.read_csv(dx_cands[-1], low_memory=False)
    dx["RID"] = pd.to_numeric(dx["RID"], errors="coerce")
    dx["visit_dt"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["dx"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx = dx[dx["RID"].isin(enrl["RID"]) & dx["visit_dt"].notna() & dx["dx"].isin([1, 2, 3])].copy()
    dx["RID"] = dx["RID"].astype(int)
    dx = dx.sort_values(["RID", "visit_dt"])

    bl_dx = dx.groupby("RID", as_index=False).first().rename(columns={"visit_dt": "bl_dt", "dx": "bl_dx"})
    base = enrl[["RID", "ORIGPROT", "AGE", "SEX", "EDUC"]].merge(bl_dx[["RID", "bl_dt", "bl_dx"]], on="RID", how="inner")

    demog = pd.read_csv(os.path.join(v26, "PTDEMOG.csv"), usecols=["RID", "VISDATE", "PTEDUCAT"], low_memory=False).sort_values(["RID", "VISDATE"]).groupby("RID", as_index=False).first()
    base = base.merge(demog[["RID", "PTEDUCAT"]], on="RID", how="left")
    base["age"] = pd.to_numeric(base["AGE"], errors="coerce")
    base["sex_male"] = np.where(base["SEX"] == "Male", 1.0, np.where(base["SEX"] == "Female", 0.0, np.nan))
    base["educ"] = pd.to_numeric(base["EDUC"], errors="coerce").fillna(pd.to_numeric(base["PTEDUCAT"], errors="coerce"))

    adqs = pd.read_csv(os.path.join(v26, "ADQS.csv"), usecols=["SUBJID", "PARAMCD", "AVAL", "ADT", "ABLFL"], low_memory=False)
    adqs["RID"] = pd.to_numeric(adqs["SUBJID"], errors="coerce")
    adqs = adqs[adqs["RID"].isin(base["RID"]) & adqs["PARAMCD"].isin(["MMSCORE", "CDRSB"])].copy()
    adqs["RID"] = adqs["RID"].astype(int)
    adqs["visit_dt"] = pd.to_datetime(adqs["ADT"], errors="coerce")
    adqs["AVAL"] = pd.to_numeric(adqs["AVAL"], errors="coerce")
    adqs = adqs.dropna(subset=["AVAL", "visit_dt"])

    base = base.set_index("RID")
    for pcd, cn in [("MMSCORE", "mmse"), ("CDRSB", "cdrsb")]:
        s = adqs[adqs["PARAMCD"] == pcd].merge(base[["bl_dt"]].reset_index(), on="RID")
        s["diff"] = (s["visit_dt"] - s["bl_dt"]).dt.days.abs()
        s["is_abl"] = (s["ABLFL"] == "Y").astype(int)
        s = s[(s["is_abl"] == 1) | (s["diff"] <= 180)].sort_values(["RID", "is_abl", "diff"], ascending=[True, False, True])
        base[cn] = s.groupby("RID")["AVAL"].first()

    # Longitudinal outcomes from PET scan date (stored in dx & cdr for scan-anchored follow-up too!)
    base = base.reset_index().sort_values("RID").reset_index(drop=True)
    base["stratum"] = base["ORIGPROT"].astype(str) + "_" + base["bl_dx"].astype(int).astype(str)
    base["is_eval"] = _det_holdout(base["RID"].map(lambda r: f"ADNI_{int(r):05d}"), base["stratum"], "LC08-pet-pipeline-eval-20261123", frac=0.30)

    # Return anchor + raw dx/cdr longitudinal tables so outcomes can be anchored strictly to PET scan date!
    cdr = pd.read_csv(os.path.join(v26, "CDR.csv"), usecols=["RID", "VISDATE", "CDRSB"], low_memory=False)
    cdr["RID"] = pd.to_numeric(cdr["RID"], errors="coerce")
    cdr["visit_dt"] = pd.to_datetime(cdr["VISDATE"], errors="coerce")
    cdr["cdrsb_val"] = pd.to_numeric(cdr["CDRSB"], errors="coerce")
    cdr = cdr.dropna(subset=["RID", "visit_dt", "cdrsb_val"]).copy()
    cdr["RID"] = cdr["RID"].astype(int)
    return base, dx[["RID", "visit_dt", "dx"]], cdr[["RID", "visit_dt", "cdrsb_val"]]


def attach_scan_anchored_outcomes(
    pet_first_df: pd.DataFrame, dx_long: pd.DataFrame, cdr_long: pd.DataFrame, final_cut: pd.Timestamp
) -> pd.DataFrame:
    """Attach scan-anchored clinical diagnosis at scan, 36m dementia conversion (strict window closure), survival, and 24m delta CDR-SB."""
    df = pet_first_df.copy()
    # Nearest diagnosis within +/- 180d of PET scan date
    md = dx_long.merge(df[["RID", "scan_dt"]], on="RID", how="inner")
    md["diff"] = (md["visit_dt"] - md["scan_dt"]).dt.days.abs()
    dx_at = md[md["diff"] <= 180].sort_values(["RID", "diff"]).drop_duplicates("RID").set_index("RID")["dx"]

    # Post-scan dementia conversion & follow-up
    md["fu_m"] = (md["visit_dt"] - md["scan_dt"]).dt.days / 30.4375
    pos = md[md["fu_m"] > 0]
    ev_dem = pos[pos["dx"] == 3].groupby("RID")["fu_m"].min()
    max_fu = pos.groupby("RID")["fu_m"].max()

    # CDR-SB at scan and 24m (18-30m) post-scan
    mc = cdr_long.merge(df[["RID", "scan_dt"]], on="RID", how="inner")
    mc["fu_m"] = (mc["visit_dt"] - mc["scan_dt"]).dt.days / 30.4375
    c0 = mc[mc["fu_m"].abs() <= 6.0].assign(d=lambda x: x["fu_m"].abs()).sort_values(["RID", "d"]).drop_duplicates("RID").set_index("RID")["cdrsb_val"]
    c24 = mc[(mc["fu_m"] >= 18.0) & (mc["fu_m"] <= 30.0)].assign(d=lambda x: (x["fu_m"] - 24.0).abs()).sort_values(["RID", "d"]).drop_duplicates("RID").set_index("RID")["cdrsb_val"]

    df = df.set_index("RID")
    df["dx_at_scan"] = dx_at
    df["ev_dem_m"] = ev_dem
    df["max_fu_m"] = max_fu
    df["cdrsb_at_scan"] = c0
    df["cdrsb_24m_post"] = c24
    closed_30 = (df["scan_dt"] + pd.to_timedelta(30.0 * 30.4375, unit="D")) <= final_cut
    df.loc[~closed_30, "cdrsb_24m_post"] = np.nan
    df["dcdrsb_24m"] = df["cdrsb_24m_post"] - df["cdrsb_at_scan"]

    # MCI at scan -> 36m dementia conversion (strict window closure) & survival
    mci_s = df["dx_at_scan"] == 2
    df["surv_event"] = np.where(mci_s & df["max_fu_m"].notna(), df["ev_dem_m"].notna().astype(float), np.nan)
    df["surv_time_m"] = np.where(df["surv_event"] == 1.0, df["ev_dem_m"], np.where(df["surv_event"] == 0.0, df["max_fu_m"], np.nan))
    c36 = mci_s & ((df["scan_dt"] + pd.to_timedelta(39.0 * 30.4375, unit="D")) <= final_cut)
    df["mci_conv_36m"] = np.where(c36 & (df["ev_dem_m"] <= 39.0), 1.0, np.where(c36 & (df["max_fu_m"] >= 33.0), 0.0, np.nan))
    # Also all non-demented (CN+MCI) -> 36m dementia/impairment conversion for discordant group clinical outcome check
    nd_c36 = df["dx_at_scan"].isin([1, 2]) & ((df["scan_dt"] + pd.to_timedelta(39.0 * 30.4375, unit="D")) <= final_cut)
    df["nd_dem_36m"] = np.where(nd_c36 & (df["ev_dem_m"] <= 39.0), 1.0, np.where(nd_c36 & (df["max_fu_m"] >= 33.0), 0.0, np.nan))
    return df.reset_index()


def load_tau_8roi(csv_path: str, dcol: str, ref_norm_col: Optional[str] = None) -> pd.DataFrame:
    need = ["RID", dcol, "META_TEMPORAL_SUVR"] + [c for _, l, r in TAU_8_ROI_PAIRS for c in (l, r)]
    if ref_norm_col:
        need.append(ref_norm_col)
    df = pd.read_csv(csv_path, usecols=need, low_memory=False)
    df = _norm_key(df, dcol)
    ref_v = pd.to_numeric(df[ref_norm_col], errors="coerce") if ref_norm_col else pd.Series(1.0, index=df.index)
    ref_v = ref_v.where(ref_v > 0.1, np.nan)
    df["meta_suvr"] = pd.to_numeric(df["META_TEMPORAL_SUVR"], errors="coerce") / ref_v
    for name, lc, rc in TAU_8_ROI_PAIRS:
        df[name] = 0.5 * (pd.to_numeric(df[lc], errors="coerce") + pd.to_numeric(df[rc], errors="coerce")) / ref_v
    return df[["RID", "scan_dt", "_k", "meta_suvr"] + TAU_8_COLS]


def main() -> None:
    parser = argparse.ArgumentParser(description="ADNI Molecular PET 6mm Reprocessing, PVC Shift, Visual Read & AT Staging Drift Pipeline")
    parser.add_argument("--adni-raw-dir", default=os.environ.get("ADNI_RAW_DIR", "data/raw/adni_route_a"))
    parser.add_argument("--adni-extract-dir", default=os.environ.get("ADNI_EXTRACT_DIR", "data/raw/adni_extract"))
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    mode = "pilot" if args.pilot else "full"
    v25 = os.path.join(args.adni_extract_dir, "v2025_07", "csv")
    v26 = os.path.join(args.adni_extract_dir, "v2026_10", "csv")
    final_cut = pd.Timestamp("2026-11-20")
    print(f"=== Molecular PET 6mm / PVC / Visual-Read Drift Pipeline ({mode}) ===", flush=True)

    anchor, dx_long, cdr_long = build_clinical_anchor(args.adni_raw_dir, args.adni_extract_dir)
    enrl_rids = set(anchor["RID"])

    # Load all PET tables
    fbp_old = _norm_key(pd.read_csv(os.path.join(v25, "ucberkeleyav45.csv"), usecols=["RID", "EXAMDATE", "SUMMARYSUVR_WHOLECEREBNORM", "SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"], low_memory=False), "EXAMDATE")
    fbb_old = _norm_key(pd.read_csv(os.path.join(v25, "ucberkeleyfbb.csv"), usecols=["RID", "EXAMDATE", "SUMMARYSUVR_WHOLECEREBNORM", "SUMMARYSUVR_WHOLECEREBNORM_1.08CUTOFF"], low_memory=False), "EXAMDATE")
    amy_6mm = _norm_key(pd.read_csv(os.path.join(v26, "UCBERKELEY_AMY_6MM.csv"), usecols=["RID", "SCANDATE", "TRACER", "SUMMARY_SUVR", "CENTILOIDS", "AMYLOID_STATUS"], low_memory=False), "SCANDATE")
    phc_amy = _norm_key(pd.read_csv(os.path.join(v26, "ADSP_PHC_PET_Amyloid_Simple.csv"), usecols=["RID", "PHC_SCANDATE", "PHC_TRACER", "PHC_CENTILOIDS", "PHC_AMYLOID_STATUS"], low_memory=False), "PHC_SCANDATE")

    tau_old = load_tau_8roi(os.path.join(v25, "ucberkeleyav1451.csv"), "EXAMDATE", None)
    tau_pvc_old = _norm_key(pd.read_csv(os.path.join(v25, "ucberkeleyav1451_pvc.csv"), usecols=["RID", "EXAMDATE", "INFERIOR_CEREBGM_SUVR", "META_TEMPORAL_SUVR"], low_memory=False), "EXAMDATE")
    tau_pvc_old["meta_suvr"] = pd.to_numeric(tau_pvc_old["META_TEMPORAL_SUVR"], errors="coerce") / pd.to_numeric(tau_pvc_old["INFERIOR_CEREBGM_SUVR"], errors="coerce")
    tau_6mm = load_tau_8roi(os.path.join(v26, "UCBERKELEY_TAU_6MM.csv"), "SCANDATE", None)
    tau_pvc_6mm = load_tau_8roi(os.path.join(v26, "UCBERKELEY_TAUPVC_6MM.csv"), "SCANDATE", None)

    amy_read = _norm_key(pd.read_csv(os.path.join(v26, "AMYREAD.csv"), usecols=["RID", "SCANDATE", "OUTCOME"], low_memory=False), "SCANDATE")
    amy_read["vis_pos"] = np.where(amy_read["OUTCOME"].astype(str).str.startswith("Elevated"), 1, np.where(amy_read["OUTCOME"].astype(str).str.startswith("Non Elevated"), 0, np.nan))
    tau_read = _norm_key(pd.read_csv(os.path.join(v26, "GOTHENBURG_VISUAL_READS_AV1451.csv"), usecols=["RID", "SCANDATE", "VISUAL_READ_BINARY"], low_memory=False), "SCANDATE")
    tau_read["vis_pos"] = np.where(tau_read["VISUAL_READ_BINARY"] == "Positive", 1, np.where(tau_read["VISUAL_READ_BINARY"] == "Negative", 0, np.nan))

    # FDG negative control
    fdg25 = _norm_key(pd.read_csv(os.path.join(v25, "ucberkeleyfdg.csv"), low_memory=False), "EXAMDATE")
    fdg26 = _norm_key(pd.read_csv(os.path.join(v26, "UCBERKELEYFDG_8mm.csv"), low_memory=False), "EXAMDATE")
    f25_r = fdg25.pivot_table(index="_k", columns="ROINAME", values="MEAN", aggfunc="first")
    f26_r = fdg26.pivot_table(index="_k", columns="ROINAME", values="MEAN", aggfunc="first")
    fdg_m = pd.concat([(f25_r["metaroi"] / f25_r["pons-vermis"]).rename("old"), (f26_r["MetaROI"] / f26_r["Top50PonsVermis"]).rename("new")], axis=1).dropna()

    # --- AXIS 1: A+/T+ THRESHOLD FLIP, GRAY-ZONE & JOINT AT MATRIX + FDG CONTROL ---
    flip_rows = []
    # 1A. Amyloid comparisons
    for cname, df_a, col_val_a, col_bin_a, df_b, col_val_b, col_bin_b, tracer_flt in [
        ("Amyloid_FBP_Native_to_6mm", fbp_old, "SUMMARYSUVR_WHOLECEREBNORM", "SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF", amy_6mm[amy_6mm["TRACER"] == "FBP"], "SUMMARY_SUVR", "AMYLOID_STATUS", "FBP"),
        ("Amyloid_FBB_Native_to_6mm", fbb_old, "SUMMARYSUVR_WHOLECEREBNORM", "SUMMARYSUVR_WHOLECEREBNORM_1.08CUTOFF", amy_6mm[amy_6mm["TRACER"] == "FBB"], "SUMMARY_SUVR", "AMYLOID_STATUS", "FBB"),
        ("Amyloid_6mm_vs_ADSP_PHC", amy_6mm, "CENTILOIDS", "AMYLOID_STATUS", phc_amy, "PHC_CENTILOIDS", "PHC_AMYLOID_STATUS", "ALL"),
    ]:
        ka = df_a.drop_duplicates("_k").set_index("_k")
        kb = df_b.drop_duplicates("_k").set_index("_k")
        sh = ka.index.intersection(kb.index)
        va = pd.to_numeric(ka.loc[sh, col_val_a], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(kb.loc[sh, col_val_b], errors="coerce").to_numpy(float)
        ba = pd.to_numeric(ka.loc[sh, col_bin_a], errors="coerce").to_numpy(float)
        bb = pd.to_numeric(kb.loc[sh, col_bin_b], errors="coerce").to_numpy(float)
        ok = np.isfinite(va) & np.isfinite(vb) & np.isin(ba, [0, 1]) & np.isin(bb, [0, 1])
        cl_ref = pd.to_numeric(amy_6mm.drop_duplicates("_k").set_index("_k").reindex(sh)["CENTILOIDS"], errors="coerce").to_numpy(float)
        gz = ok & np.isfinite(cl_ref) & (cl_ref >= 15.0) & (cl_ref <= 35.0)
        fl_all = int(np.sum(ba[ok] != bb[ok]))
        fl_pn = int(np.sum((ba[ok] == 1) & (bb[ok] == 0)))
        fl_np = int(np.sum((ba[ok] == 0) & (bb[ok] == 1)))
        fl_gz = int(np.sum(ba[gz] != bb[gz]))
        flip_rows.append({
            "comparison": cname, "threshold": "Official A+ cutoff", "n_scans": int(np.sum(ok)),
            "unique_rids": int(ka.loc[sh[ok], "RID"].nunique()),
            "mean_a": round(float(np.mean(va[ok])), 4), "mean_b": round(float(np.mean(vb[ok])), 4),
            "mean_diff": round(float(np.mean(vb[ok] - va[ok])), 4),
            "pearson_r": round(float(np.corrcoef(va[ok], vb[ok])[0, 1]), 5),
            "icc_a1": round(icc_a1(va[ok], vb[ok]), 5),
            "flip_n": fl_all, "flip_pct": round(fl_all / max(int(np.sum(ok)), 1) * 100.0, 2),
            "pos_to_neg": fl_pn, "neg_to_pos": fl_np,
            "gray_zone_n": int(np.sum(gz)), "gray_zone_flip_pct": round(fl_gz / max(int(np.sum(gz)), 1) * 100.0, 2),
        })

    # Centiloid threshold sensitivity (6mm vs ADSP-PHC across CL >= 18, 20, 24, 30, 37)
    k_a6 = amy_6mm.drop_duplicates("_k").set_index("_k")
    k_pa = phc_amy.drop_duplicates("_k").set_index("_k")
    sh_cl = k_a6.index.intersection(k_pa.index)
    cl_a = pd.to_numeric(k_a6.loc[sh_cl, "CENTILOIDS"], errors="coerce").to_numpy(float)
    cl_b = pd.to_numeric(k_pa.loc[sh_cl, "PHC_CENTILOIDS"], errors="coerce").to_numpy(float)
    ok_cl = np.isfinite(cl_a) & np.isfinite(cl_b)
    for th_cl in (18.0, 20.0, 24.0, 30.0, 37.0):
        pa_b, pb_b = (cl_a[ok_cl] >= th_cl).astype(int), (cl_b[ok_cl] >= th_cl).astype(int)
        fl = int(np.sum(pa_b != pb_b))
        flip_rows.append({
            "comparison": "Amyloid_6mm_vs_ADSP_PHC_CL", "threshold": f"CL>={int(th_cl)}", "n_scans": int(np.sum(ok_cl)),
            "unique_rids": int(k_a6.loc[sh_cl[ok_cl], "RID"].nunique()),
            "mean_a": round(float(np.mean(cl_a[ok_cl])), 2), "mean_b": round(float(np.mean(cl_b[ok_cl])), 2),
            "mean_diff": round(float(np.mean(cl_b[ok_cl] - cl_a[ok_cl])), 2),
            "pearson_r": round(float(np.corrcoef(cl_a[ok_cl], cl_b[ok_cl])[0, 1]), 5),
            "icc_a1": round(icc_a1(cl_a[ok_cl], cl_b[ok_cl]), 5),
            "flip_n": fl, "flip_pct": round(fl / int(np.sum(ok_cl)) * 100.0, 2),
            "pos_to_neg": int(np.sum((pa_b == 1) & (pb_b == 0))), "neg_to_pos": int(np.sum((pa_b == 0) & (pb_b == 1))),
            "gray_zone_n": 0, "gray_zone_flip_pct": float("nan"),
        })

    # 1B. Tau comparisons
    for cname, df_a, df_b, th_list in [
        ("Tau_AV1451_Native_to_6mm", tau_old, tau_6mm, (1.20, 1.25, 1.30, 1.37)),
        ("Tau_PVC_NativeNorm_to_6mmPVC", tau_pvc_old, tau_pvc_6mm, (1.25, 1.37, 1.50)),
        ("Tau_6mm_NonPVC_to_6mmPVC", tau_6mm, tau_pvc_6mm, (1.25, 1.30, 1.37)),
    ]:
        ka, kb = df_a.drop_duplicates("_k").set_index("_k"), df_b.drop_duplicates("_k").set_index("_k")
        sh = ka.index.intersection(kb.index)
        va = pd.to_numeric(ka.loc[sh, "meta_suvr"], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(kb.loc[sh, "meta_suvr"], errors="coerce").to_numpy(float)
        ok = np.isfinite(va) & np.isfinite(vb)
        for th in th_list:
            pa_b, pb_b = (va[ok] >= th).astype(int), (vb[ok] >= th).astype(int)
            fl = int(np.sum(pa_b != pb_b))
            flip_rows.append({
                "comparison": cname, "threshold": f"SUVR>={th:.2f}", "n_scans": int(np.sum(ok)),
                "unique_rids": int(ka.loc[sh[ok], "RID"].nunique()),
                "mean_a": round(float(np.mean(va[ok])), 4), "mean_b": round(float(np.mean(vb[ok])), 4),
                "mean_diff": round(float(np.mean(vb[ok] - va[ok])), 4),
                "pearson_r": round(float(np.corrcoef(va[ok], vb[ok])[0, 1]), 5),
                "icc_a1": round(icc_a1(va[ok], vb[ok]), 5),
                "flip_n": fl, "flip_pct": round(fl / int(np.sum(ok)) * 100.0, 2),
                "pos_to_neg": int(np.sum((pa_b == 1) & (pb_b == 0))), "neg_to_pos": int(np.sum((pa_b == 0) & (pb_b == 1))),
                "gray_zone_n": 0, "gray_zone_flip_pct": float("nan"),
            })

    # 1C. FDG negative control
    xo, xn = fdg_m["old"].to_numpy(float), fdg_m["new"].to_numpy(float)
    th_fdg = float(np.percentile(xo, 25))
    fl_fdg = int(np.sum((xo <= th_fdg) != (xn <= th_fdg)))
    flip_rows.append({
        "comparison": "FDG_NegativeControl_Native_to_8mm", "threshold": "Q25 hypometab.", "n_scans": int(len(xo)),
        "unique_rids": int(pd.Series(fdg_m.index).str.split("|").str[0].nunique()),
        "mean_a": round(float(np.mean(xo)), 4), "mean_b": round(float(np.mean(xn)), 4),
        "mean_diff": round(float(np.mean(xn - xo)), 5),
        "pearson_r": round(float(np.corrcoef(xo, xn)[0, 1]), 5), "icc_a1": round(icc_a1(xo, xn), 5),
        "flip_n": fl_fdg, "flip_pct": round(fl_fdg / len(xo) * 100.0, 2),
        "pos_to_neg": int(np.sum((xo <= th_fdg) & (xn > th_fdg))), "neg_to_pos": int(np.sum((xo > th_fdg) & (xn <= th_fdg))),
        "gray_zone_n": 0, "gray_zone_flip_pct": float("nan"),
    })

    # 1D. Joint AT 4-class staging matrix (Native vs 6mm on subjects with paired Amyloid + Tau within 365 days)
    amy_old_all = pd.concat([
        fbp_old[["RID", "scan_dt", "_k"]].assign(a_old=pd.to_numeric(fbp_old["SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF"], errors="coerce")),
        fbb_old[["RID", "scan_dt", "_k"]].assign(a_old=pd.to_numeric(fbb_old["SUMMARYSUVR_WHOLECEREBNORM_1.08CUTOFF"], errors="coerce")),
    ], ignore_index=True).drop_duplicates("_k")
    a_pair = amy_old_all.merge(amy_6mm[["_k", "AMYLOID_STATUS", "CENTILOIDS"]].drop_duplicates("_k"), on="_k", how="inner").dropna(subset=["a_old", "AMYLOID_STATUS"])
    t_pair = tau_old[["RID", "scan_dt", "_k", "meta_suvr"]].drop_duplicates("_k").merge(
        tau_6mm[["_k", "meta_suvr"]].drop_duplicates("_k"), on="_k", suffixes=("_old", "_6mm")
    )
    at_m = a_pair.merge(t_pair, on="RID", suffixes=("_a", "_t"))
    at_m["day_gap"] = (at_m["scan_dt_t"] - at_m["scan_dt_a"]).dt.days.abs()
    at_m = at_m[at_m["day_gap"] <= 365].sort_values(["RID", "day_gap"]).drop_duplicates("RID")
    at_rows = []
    for t_th_old, t_th_new, strat in [(1.25, 1.25, "Fixed_uncalibrated_1.25"), (1.25, 1.30, "Recalibrated_6mm_1.30"), (1.30, 1.30, "Fixed_1.30")]:
        st_old = at_m["a_old"].astype(int).map({0: "A-", 1: "A+"}) + (at_m["meta_suvr_old"] >= t_th_old).astype(int).map({0: "T-", 1: "T+"})
        st_new = at_m["AMYLOID_STATUS"].astype(int).map({0: "A-", 1: "A+"}) + (at_m["meta_suvr_6mm"] >= t_th_new).astype(int).map({0: "T-", 1: "T+"})
        fl = int((st_old != st_new).sum())
        at_rows.append({
            "strategy": strat, "n_subjects": int(len(at_m)),
            "at_reclassified_n": fl, "at_reclassified_pct": round(fl / max(len(at_m), 1) * 100.0, 2),
            "a_only_flips": int((at_m["a_old"].astype(int) != at_m["AMYLOID_STATUS"].astype(int)).sum()),
            "t_only_flips": int(((at_m["meta_suvr_old"] >= t_th_old) != (at_m["meta_suvr_6mm"] >= t_th_new)).sum()),
            "old_ApTp_n": int((st_old == "A+T+").sum()), "new_ApTp_n": int((st_new == "A+T+").sum()),
        })

    # --- AXIS 2: CLINICAL VISUAL-READ GOLD STANDARD ROC & THRESHOLD CALIBRATION ---
    vis_rows = []
    tr_idx = tau_read.dropna(subset=["vis_pos"]).drop_duplicates("_k").set_index("_k")
    for plbl, tdf in [
        ("Tau_Native_AV1451", tau_old),
        ("Tau_6mm_NonPVC", tau_6mm),
        ("Tau_Native_PVC_Norm", tau_pvc_old),
        ("Tau_6mm_PVC", tau_pvc_6mm),
    ]:
        tk = tdf.drop_duplicates("_k").set_index("_k")
        sh = tr_idx.index.intersection(tk.index)
        y_v = tr_idx.loc[sh, "vis_pos"].to_numpy(int)
        sc = pd.to_numeric(tk.loc[sh, "meta_suvr"], errors="coerce").to_numpy(float)
        ok = np.isfinite(sc)
        y_v, sc = y_v[ok], sc[ok]
        yr = youden_optimal_threshold(y_v, sc)
        # Also evaluate fixed legacy cutoffs 1.20, 1.25, 1.30, 1.37
        row = {"modality": "Tau_Gothenburg_Visual", "pipeline": plbl, **yr}
        for th in (1.20, 1.25, 1.30, 1.37):
            row[f"acc_at_{th:.2f}"] = round(float(np.mean((sc >= th).astype(int) == y_v) * 100.0), 2)
        vis_rows.append(row)

    # Amyloid visual read (AMYREAD) vs AMY_6MM & ADSP_PHC_PET_Amyloid_Simple
    ar_idx = amy_read.dropna(subset=["vis_pos"]).drop_duplicates("_k").set_index("_k")
    for plbl, adf, scol, bcol in [
        ("Amyloid_UCB_6mm_CL", amy_6mm, "CENTILOIDS", "AMYLOID_STATUS"),
        ("Amyloid_ADSP_PHC_CL", phc_amy, "PHC_CENTILOIDS", "PHC_AMYLOID_STATUS"),
    ]:
        ak = adf.drop_duplicates("_k").set_index("_k")
        sh = ar_idx.index.intersection(ak.index)
        y_v = ar_idx.loc[sh, "vis_pos"].to_numpy(int)
        sc = pd.to_numeric(ak.loc[sh, scol], errors="coerce").to_numpy(float)
        bn = pd.to_numeric(ak.loc[sh, bcol], errors="coerce").to_numpy(float)
        ok = np.isfinite(sc) & np.isin(bn, [0, 1])
        yr = youden_optimal_threshold(y_v[ok], sc[ok])
        vis_rows.append({
            "modality": "Amyloid_AMYREAD_Visual", "pipeline": plbl, **yr,
            "official_binary_acc": round(float(np.mean(bn[ok] == y_v[ok]) * 100.0), 2),
        })

    # --- AXIS 3: 8-ROI SPATIAL TAU SUBTYPING STABILITY (NATIVE->6MM & 6MM NON-PVC->PVC) ---
    sub_rows = []
    for pname, df_s, df_t in [
        ("Tau_Native_to_6mm", tau_old, tau_6mm),
        ("Tau_6mm_NonPVC_to_6mmPVC", tau_6mm, tau_pvc_6mm),
    ]:
        ks = df_s.sort_values(["RID", "scan_dt"]).drop_duplicates("RID")
        kt = df_t.drop_duplicates("_k")
        m_st = ks.merge(kt[["_k"] + TAU_8_COLS], on="_k", suffixes=("_src", "_tgt"))
        m_st = m_st[m_st["RID"].isin(enrl_rids)].merge(anchor[["RID", "is_eval", "bl_dx"]], on="RID")
        m_st = attach_scan_anchored_outcomes(m_st, dx_long, cdr_long, final_cut)
        m_st = m_st.dropna(subset=[f"{c}_src" for c in TAU_8_COLS] + [f"{c}_tgt" for c in TAU_8_COLS]).sort_values("RID")
        dev, ev = m_st[~m_st["is_eval"]], m_st[m_st["is_eval"]]
        assert len(set(dev["RID"]).intersection(set(ev["RID"]))) == 0
        Xs_tr, Xt_tr = dev[[f"{c}_src" for c in TAU_8_COLS]].to_numpy(float), dev[[f"{c}_tgt" for c in TAU_8_COLS]].to_numpy(float)
        Xs_ev, Xt_ev = ev[[f"{c}_src" for c in TAU_8_COLS]].to_numpy(float), ev[[f"{c}_tgt" for c in TAU_8_COLS]].to_numpy(float)

        ks_grid = (3,) if args.pilot else (2, 3, 4)
        for algo in ("kmeans", "gmm_diag", "ward"):
            for k in ks_grid:
                res_lbl = fit_predict_tau_subtypes(Xs_tr, Xs_ev, Xt_tr, Xt_ev, algo, k, seed=20261123)
                ref_l = res_lbl["source_on_source"]
                for scen in ("A_naive_direct_transfer", "B_target_restd_rescue", "C_refitted_target"):
                    cur_l = res_lbl[scen]
                    sub_rows.append({
                        "pair": pname, "algo": algo, "k": int(k), "scenario": scen,
                        "dev_n": int(len(dev)), "eval_n": int(len(ev)),
                        "ari_vs_source": round(float(adjusted_rand_score(ref_l, cur_l)), 4),
                        "nmi_vs_source": round(float(normalized_mutual_info_score(ref_l, cur_l)), 4),
                        "flip_rate_pct": round(float(np.mean(cur_l != ref_l) * 100.0), 2),
                    })

    # --- AXIS 4A: CLINICAL OUTCOMES OF DISCORDANT vs CONCORDANT GROUPS ---
    disc_rows = []
    # Amyloid 6mm vs ADSP-PHC first scan per enrolled subject
    a6_first = amy_6mm[amy_6mm["RID"].isin(enrl_rids)].sort_values(["RID", "scan_dt"]).drop_duplicates("RID")
    m_ap_first = a6_first.merge(phc_amy[["_k", "PHC_CENTILOIDS", "PHC_AMYLOID_STATUS"]].drop_duplicates("_k"), on="_k", how="inner")
    m_ap_first = attach_scan_anchored_outcomes(m_ap_first, dx_long, cdr_long, final_cut)
    m_ap_first = m_ap_first[m_ap_first["AMYLOID_STATUS"].isin([0, 1]) & m_ap_first["PHC_AMYLOID_STATUS"].isin([0, 1])].copy()
    m_ap_first["grp"] = np.where(
        (m_ap_first["AMYLOID_STATUS"] == 0) & (m_ap_first["PHC_AMYLOID_STATUS"] == 0), "Concordant_A-",
        np.where((m_ap_first["AMYLOID_STATUS"] == 1) & (m_ap_first["PHC_AMYLOID_STATUS"] == 1), "Concordant_A+",
                 np.where((m_ap_first["AMYLOID_STATUS"] == 1) & (m_ap_first["PHC_AMYLOID_STATUS"] == 0), "Discordant_6mm+_PHC-", "Discordant_6mm-_PHC+"))
    )
    for gname, gdf in m_ap_first.groupby("grp"):
        nd = gdf["nd_dem_36m"].dropna()
        mci = gdf["mci_conv_36m"].dropna()
        dc = gdf["dcdrsb_24m"].dropna()
        disc_rows.append({
            "comparison": "Amyloid_6mm_vs_ADSP_PHC", "group": gname, "n_total": int(len(gdf)),
            "mean_cl_6mm": round(float(gdf["CENTILOIDS"].mean()), 1),
            "nd_36m_dem_n": int(len(nd)), "nd_36m_dem_rate_pct": round(float(nd.mean() * 100.0), 2) if len(nd) >= 5 else float("nan"),
            "mci_36m_conv_n": int(len(mci)), "mci_36m_conv_rate_pct": round(float(mci.mean() * 100.0), 2) if len(mci) >= 5 else float("nan"),
            "dcdrsb_24m_n": int(len(dc)), "mean_dcdrsb_24m": round(float(dc.mean()), 3) if len(dc) >= 5 else float("nan"),
        })

    # Tau Native vs 6mm discordant (@1.25 cutoff)
    t_first = tau_old[tau_old["RID"].isin(enrl_rids)].sort_values(["RID", "scan_dt"]).drop_duplicates("RID")
    m_t_first = t_first[["RID", "scan_dt", "_k", "meta_suvr"]].merge(tau_6mm[["_k", "meta_suvr"]].drop_duplicates("_k"), on="_k", suffixes=("_old", "_6mm"))
    m_t_first = attach_scan_anchored_outcomes(m_t_first, dx_long, cdr_long, final_cut)
    to_p, tn_p = (m_t_first["meta_suvr_old"] >= 1.25).astype(int), (m_t_first["meta_suvr_6mm"] >= 1.25).astype(int)
    m_t_first["grp"] = np.where((to_p == 0) & (tn_p == 0), "Concordant_T-", np.where((to_p == 1) & (tn_p == 1), "Concordant_T+", np.where((to_p == 0) & (tn_p == 1), "Discordant_Old-_6mm+", "Discordant_Old+_6mm-")))
    for gname, gdf in m_t_first.groupby("grp"):
        nd = gdf["nd_dem_36m"].dropna()
        mci = gdf["mci_conv_36m"].dropna()
        dc = gdf["dcdrsb_24m"].dropna()
        disc_rows.append({
            "comparison": "Tau_Native_vs_6mm_1.25", "group": gname, "n_total": int(len(gdf)),
            "mean_cl_6mm": round(float(gdf["meta_suvr_6mm"].mean()), 3),
            "nd_36m_dem_n": int(len(nd)), "nd_36m_dem_rate_pct": round(float(nd.mean() * 100.0), 2) if len(nd) >= 5 else float("nan"),
            "mci_36m_conv_n": int(len(mci)), "mci_36m_conv_rate_pct": round(float(mci.mean() * 100.0), 2) if len(mci) >= 5 else float("nan"),
            "dcdrsb_24m_n": int(len(dc)), "mean_dcdrsb_24m": round(float(dc.mean()), 3) if len(dc) >= 5 else float("nan"),
        })

    # --- AXIS 4B: DOWNSTREAM 36m PREDICTION & COX SURVIVAL PIPELINE TRANSFER ---
    pred_rows = []
    # Pair 1: Amyloid FBP Native SUVR -> 6mm SUVR
    fbp_first = fbp_old[fbp_old["RID"].isin(enrl_rids)].sort_values(["RID", "scan_dt"]).drop_duplicates("RID")
    p_amy = fbp_first[["RID", "scan_dt", "_k", "SUMMARYSUVR_WHOLECEREBNORM"]].merge(
        amy_6mm[amy_6mm["TRACER"] == "FBP"][["_k", "SUMMARY_SUVR", "CENTILOIDS"]].drop_duplicates("_k"), on="_k"
    ).merge(anchor[["RID", "is_eval"] + CLIN_5], on="RID")
    p_amy = attach_scan_anchored_outcomes(p_amy, dx_long, cdr_long, final_cut).rename(
        columns={"SUMMARYSUVR_WHOLECEREBNORM": "pet_src", "SUMMARY_SUVR": "pet_tgt"}
    )

    # Pair 2: Amyloid 6mm CL -> ADSP-PHC CL
    p_cl = a6_first[["RID", "scan_dt", "_k", "CENTILOIDS"]].merge(
        phc_amy[["_k", "PHC_CENTILOIDS"]].drop_duplicates("_k"), on="_k"
    ).merge(anchor[["RID", "is_eval"] + CLIN_5], on="RID")
    p_cl = attach_scan_anchored_outcomes(p_cl, dx_long, cdr_long, final_cut).rename(
        columns={"CENTILOIDS": "pet_src", "PHC_CENTILOIDS": "pet_tgt"}
    )

    # Pair 3: Tau Native -> 6mm
    p_tau = t_first[["RID", "scan_dt", "_k", "meta_suvr"]].merge(
        tau_6mm[["_k", "meta_suvr"]].drop_duplicates("_k"), on="_k", suffixes=("_src", "_tgt")
    ).merge(anchor[["RID", "is_eval"] + CLIN_5], on="RID")
    p_tau = attach_scan_anchored_outcomes(p_tau, dx_long, cdr_long, final_cut).rename(
        columns={"meta_suvr_src": "pet_src", "meta_suvr_tgt": "pet_tgt"}
    )

    # Pair 4: Tau 6mm Non-PVC -> 6mm PVC
    t6_first = tau_6mm[tau_6mm["RID"].isin(enrl_rids)].sort_values(["RID", "scan_dt"]).drop_duplicates("RID")
    p_pvc = t6_first[["RID", "scan_dt", "_k", "meta_suvr"]].merge(
        tau_pvc_6mm[["_k", "meta_suvr"]].drop_duplicates("_k"), on="_k", suffixes=("_src", "_tgt")
    ).merge(anchor[["RID", "is_eval"] + CLIN_5], on="RID")
    p_pvc = attach_scan_anchored_outcomes(p_pvc, dx_long, cdr_long, final_cut).rename(
        columns={"meta_suvr_src": "pet_src", "meta_suvr_tgt": "pet_tgt"}
    )

    for pname, pdf in [
        ("Amyloid_FBP_Native_to_6mm", p_amy),
        ("Amyloid_6mm_to_ADSP_PHC_CL", p_cl),
        ("Tau_Native_to_6mm", p_tau),
        ("Tau_6mm_NonPVC_to_PVC", p_pvc),
    ]:
        for pnl_name, extra in [("PET_only", []), ("Clin_plus_PET", CLIN_5)]:
            sc = extra + ["pet_src"]
            tc = extra + ["pet_tgt"]
            pet_idx = [len(extra)]
            fnames = extra + ["pet"]
            m36 = pdf[pdf["mci_conv_36m"].notna()].dropna(subset=["pet_src", "pet_tgt"])
            dev36, ev36 = m36[~m36["is_eval"]], m36[m36["is_eval"]]
            ms = pdf[pdf["surv_event"].notna()].dropna(subset=["pet_src", "pet_tgt"])
            devs, evs = ms[~ms["is_eval"]], ms[ms["is_eval"]]
            if len(dev36) < 40 or len(ev36) < 20:
                continue
            for strat, xtr36, xev36, xtrs, xevs, ridx in [
                ("1_source_on_source", dev36[sc], ev36[sc], devs[sc], evs[sc], None),
                ("2_naive_direct_src_on_tgt", dev36[sc], ev36[tc], devs[sc], evs[tc], None),
                ("3_target_restd_rescue", dev36[sc], ev36[tc], devs[sc], evs[tc], pet_idx),
                ("4_refitted_tgt_on_tgt", dev36[tc], ev36[tc], devs[tc], evs[tc], None),
            ]:
                r36 = fit_eval_pet_logistic(xtr36.to_numpy(float), dev36["mci_conv_36m"].to_numpy(float), xev36.to_numpy(float), ev36["mci_conv_36m"].to_numpy(float), restd_cols=ridx)
                rcox = fit_eval_pet_cox(xtrs.to_numpy(float), devs["surv_time_m"].to_numpy(float), devs["surv_event"].to_numpy(int), xevs.to_numpy(float), evs["surv_time_m"].to_numpy(float), evs["surv_event"].to_numpy(int), fnames, restd_cols=ridx)
                pred_rows.append({
                    "pair": pname, "panel": pnl_name, "strategy": strat,
                    "dev_n_36m": int(len(dev36)), "eval_n_36m": int(len(ev36)),
                    "brier_36m": r36["brier"], "auroc_36m": r36["auroc"],
                    "oe_36m": r36["oe_ratio"], "bias_36m_pct": r36["bias_pct"],
                    "dev_n_surv": int(len(devs)), "eval_n_surv": int(len(evs)),
                    "cox_c_index": rcox["c_index"], "cox_oe_36m": rcox["oe_36m"], "cox_bias_36m_pct": rcox["bias_36m_pct"],
                })

    sub_dir = os.path.join(args.out_dir, f"pet_drift_{mode}")
    os.makedirs(sub_dir, exist_ok=True)
    pd.DataFrame(flip_rows).to_csv(os.path.join(sub_dir, "pet_threshold_flips_and_fdg.csv"), index=False)
    pd.DataFrame(at_rows).to_csv(os.path.join(sub_dir, "joint_at_staging_reclassification.csv"), index=False)
    pd.DataFrame(vis_rows).to_csv(os.path.join(sub_dir, "visual_read_roc_calibration.csv"), index=False)
    pd.DataFrame(sub_rows).to_csv(os.path.join(sub_dir, "tau_spatial_subtyping_transfer.csv"), index=False)
    pd.DataFrame(disc_rows).to_csv(os.path.join(sub_dir, "discordant_group_clinical_outcomes.csv"), index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(sub_dir, "prediction_survival_pet_transfer.csv"), index=False)

    meta = {"mode": mode, "elapsed_seconds": round(time.time() - t0, 2), "n_enrolled_anchor": int(len(anchor))}
    with open(os.path.join(sub_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"=== PET Drift Pipeline ({mode}) FINISHED in {time.time() - t0:.1f}s -> {sub_dir} ===", flush=True)


if __name__ == "__main__":
    main()
