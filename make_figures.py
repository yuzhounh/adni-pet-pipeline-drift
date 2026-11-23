"""Generate 6 publication-grade PDF/PNG figures for ADNI molecular PET 6mm reprocessing, PVC shift, visual-read ROC & AT drift."""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAL = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#D95F02",
    "crimson": "#B2182B",
    "purple": "#7570B3",
    "gold": "#E6AB02",
    "slate": "#4D4D4D",
    "blue": "#2166AC",
}


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 7.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
    })


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig1(res_dir: str, fig_dir: str) -> None:
    """Figure 1: Amyloid Gray-Zone Flips, Tau/PVC Fixed-Cutoff Flips & FDG Negative Control."""
    fl = pd.read_csv(os.path.join(res_dir, "pet_threshold_flips_and_fdg.csv"))
    at = pd.read_csv(os.path.join(res_dir, "joint_at_staging_reclassification.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))

    # Panel A: Amyloid Global vs 15-35 CL Gray-Zone Flip Rate (%) + FDG negative control
    ax = axes[0]
    a_keys = ["Amyloid_FBP_Native_to_6mm", "Amyloid_FBB_Native_to_6mm", "Amyloid_6mm_vs_ADSP_PHC", "FDG_NegativeControl_Native_to_8mm"]
    a_lbls = ["FBP\nNative->6mm", "FBB\nNative->6mm", "6mm vs.\nADSP-PHC", "FDG Control\nNative->8mm"]
    sub_a = fl[fl["comparison"].isin(a_keys)].set_index("comparison").loc[a_keys]
    x = np.arange(len(a_keys))
    w = 0.35
    v_all = sub_a["flip_pct"].to_numpy(float)
    v_gz = np.nan_to_num(sub_a["gray_zone_flip_pct"].to_numpy(float), nan=0.0)
    b1 = ax.bar(x - w / 2, v_all, width=w, color=PAL["navy"], label="All paired scans")
    b2 = ax.bar(x[:3] + w / 2, v_gz[:3], width=w, color=PAL["crimson"], label="15-35 CL gray zone")
    for bar, v in zip(b1, v_all):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.2)
    for bar, v in zip(b2, v_gz[:3]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.4, color=PAL["crimson"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(a_lbls, fontsize=7.9)
    ax.set_ylim(0, 48)
    ax.set_ylabel("Binary Status Reclassification (%)")
    ax.set_title("A. Amyloid A+ Flips & FDG Control")
    ax.legend(frameon=True, loc="upper left", fontsize=7.5)

    # Panel B: Tau PET Threshold Flip Rates (Native->6mm, 6mm NoPVC->PVC, PVC Native->6mm)
    ax = axes[1]
    t1 = fl[fl["comparison"] == "Tau_AV1451_Native_to_6mm"]
    t2 = fl[fl["comparison"] == "Tau_6mm_NonPVC_to_6mmPVC"]
    t3 = fl[fl["comparison"] == "Tau_PVC_NativeNorm_to_6mmPVC"]
    ths = ["SUVR>=1.25", "SUVR>=1.37"]
    xt = np.arange(len(ths))
    w3 = 0.25
    for idx, (df_t, clr, lbl) in enumerate([
        (t1, PAL["teal"], "AV1451 Native -> 6mm"),
        (t2, PAL["coral"], "6mm Non-PVC -> 6mm PVC"),
        (t3, PAL["crimson"], "Native PVC (norm) -> 6mm PVC"),
    ]):
        s = df_t.set_index("threshold")
        vals = [s.loc[th, "flip_pct"] for th in ths]
        bars = ax.bar(xt + (idx - 1) * w3, vals, width=w3, color=clr, label=lbl)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xt); ax.set_xticklabels(["Fixed SUVR >= 1.25", "Fixed SUVR >= 1.37"])
    ax.set_ylim(0, 72)
    ax.set_ylabel("Tau T+ Reclassification (%)")
    ax.set_title("B. Tau PET T+ Flips Under Fixed Cutoffs")
    ax.legend(frameon=True, loc="upper left", fontsize=7.3)

    # Panel C: Joint AT 4-Stage Reclassification (n=854) & A+T+ Count Shift
    ax = axes[2]
    at_lbls = ["Fixed T>=1.25\n(Uncalibrated)", "Recalibrated\n(1.25->1.30)", "Fixed T>=1.30\n(Both)"]
    xa = np.arange(len(at))
    bars = ax.bar(xa, at["at_reclassified_pct"], width=0.52, color=[PAL["crimson"], PAL["teal"], PAL["navy"]])
    for i, (bar, r) in enumerate(zip(bars, at.itertuples())):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                f"{r.at_reclassified_pct:.1f}%\n(A+T+: {r.old_ApTp_n}->{r.new_ApTp_n})",
                ha="center", va="bottom", fontsize=7.3)
    ax.set_xticks(xa); ax.set_xticklabels(at_lbls, fontsize=7.8)
    ax.set_ylim(0, 15.2)
    ax.set_ylabel("Joint AT Stage Reclassification (%)")
    ax.set_title("C. Joint AT Staging Shift (n=854 Paired)")

    fig.tight_layout()
    _save(fig, fig_dir, "fig1_pet_at_threshold_flips")


def make_fig2(res_dir: str, fig_dir: str) -> None:
    """Figure 2: Clinical Visual-Read Gold-Standard Calibration (Gothenburg Tau n=789-835 & AMYREAD n=689-801)."""
    vis = pd.read_csv(os.path.join(res_dir, "visual_read_roc_calibration.csv"))
    tau_v = vis[vis["modality"] == "Tau_Gothenburg_Visual"].set_index("pipeline")
    amy_v = vis[vis["modality"] == "Amyloid_AMYREAD_Visual"].set_index("pipeline")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    # Panel A: Tau Gothenburg Visual Agreement across Fixed Cutoffs (1.20, 1.25, 1.30, 1.37) + Optimal
    ax = axes[0]
    cuts = [1.20, 1.25, 1.30, 1.37]
    for plbl, clr, mk, lbl in [
        ("Tau_Native_AV1451", PAL["navy"], "o", "Native AV1451 (opt 1.26)"),
        ("Tau_6mm_NonPVC", PAL["crimson"], "s", "6mm Non-PVC (opt 1.28)"),
        ("Tau_6mm_PVC", PAL["coral"], "^", "6mm PVC (opt 1.39)"),
    ]:
        r = tau_v.loc[plbl]
        ys = [r[f"acc_at_{c:.2f}"] for c in cuts]
        ax.plot(cuts, ys, color=clr, lw=2.2, marker=mk, ms=6, label=lbl)
    ax.annotate("Uncalibrated 1.20:\n6mm drops -11.0 pp", xy=(1.20, 71.74), xytext=(1.235, 65.0),
                arrowprops=dict(arrowstyle="->", color=PAL["crimson"], lw=1.2), fontsize=7.5, color=PAL["crimson"], fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    ax.set_xticks(cuts); ax.set_xticklabels([f"{c:.2f}" for c in cuts])
    ax.set_ylim(58, 99)
    ax.set_xlabel("Applied Meta-Temporal SUVR Threshold")
    ax.set_ylabel("Agreement with Visual Read (%)")
    ax.set_title("A. Tau Visual-Read Concordance vs. Cutoff")
    ax.legend(frameon=True, loc="lower right", fontsize=7.2)

    # Panel B: Optimal Youden Threshold & AUROC across all 4 Tau Pipelines
    ax = axes[1]
    p_ord = ["Tau_Native_AV1451", "Tau_6mm_NonPVC", "Tau_6mm_PVC", "Tau_Native_PVC_Norm"]
    p_lbls = ["Native\nNon-PVC", "6mm\nNon-PVC", "6mm\nPVC", "Native PVC\n(Ref-Norm)"]
    xp = np.arange(len(p_ord))
    opts = [tau_v.loc[p, "opt_thresh"] for p in p_ord]
    aurocs = [tau_v.loc[p, "auroc"] for p in p_ord]
    accs = [tau_v.loc[p, "opt_acc"] for p in p_ord]
    bars = ax.bar(xp, opts, width=0.52, color=[PAL["navy"], PAL["teal"], PAL["coral"], PAL["crimson"]])
    for bar, th, au, ac in zip(bars, opts, aurocs, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, th + 0.03, f"Thr {th:.2f}\nAUC {au:.3f}\nAcc {ac:.1f}%",
                ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(xp); ax.set_xticklabels(p_lbls, fontsize=7.9)
    ax.set_ylim(0.8, 2.02)
    ax.set_ylabel("Youden-Optimal Meta-Temporal SUVR")
    ax.set_title("B. Pipeline-Specific Optimal Tau Thresholds")

    # Panel C: Amyloid Visual-Read Gold Standard (AMYREAD): UCB 6mm vs ADSP-PHC Optimal Centiloids
    ax = axes[2]
    a_ord = ["Amyloid_UCB_6mm_CL", "Amyloid_ADSP_PHC_CL"]
    a_lbls = ["UC Berkeley\n6mm CL\n(n=801)", "ADSP-PHC\nHarmonized CL\n(n=689)"]
    xa = np.arange(2)
    a_opts = [amy_v.loc[p, "opt_thresh"] for p in a_ord]
    a_aucs = [amy_v.loc[p, "auroc"] for p in a_ord]
    a_accs = [amy_v.loc[p, "opt_acc"] for p in a_ord]
    bars = ax.bar(xa, a_opts, width=0.45, color=[PAL["navy"], PAL["purple"]])
    for bar, clv, au, ac in zip(bars, a_opts, a_aucs, a_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, clv + 0.6, f"Opt {clv:.1f} CL\nAUC {au:.4f}\nAcc {ac:.1f}%",
                ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(xa); ax.set_xticklabels(a_lbls, fontsize=8.3)
    ax.set_ylim(0, 30.5)
    ax.set_ylabel("Youden-Optimal Centiloid Threshold (CL)")
    ax.set_title("C. Amyloid Visual Read: 8 CL Scale Offset")

    fig.tight_layout()
    _save(fig, fig_dir, "fig2_visual_read_calibration")


def make_fig3(res_dir: str, fig_dir: str) -> None:
    """Figure 3: Discordant Patient Clinical Progression + Spatial Tau Subtyping & Cox Calibration Transfer."""
    dc = pd.read_csv(os.path.join(res_dir, "discordant_group_clinical_outcomes.csv"))
    su = pd.read_csv(os.path.join(res_dir, "tau_spatial_subtyping_transfer.csv"))
    pr = pd.read_csv(os.path.join(res_dir, "prediction_survival_pet_transfer.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))

    # Panel A: Clinical Progression (36m Dementia Rate % & 24m dCDR-SB) of Concordant vs Discordant Groups
    ax = axes[0]
    rows_show = [
        ("Amyloid_6mm_vs_ADSP_PHC", "Concordant_A-", "A-/A-\n(n=643)", PAL["teal"]),
        ("Amyloid_6mm_vs_ADSP_PHC", "Discordant_6mm+_PHC-", "6mm+/PHC-\n(n=43)", PAL["gold"]),
        ("Amyloid_6mm_vs_ADSP_PHC", "Concordant_A+", "A+/A+\n(n=405)", PAL["crimson"]),
        ("Tau_Native_vs_6mm_1.25", "Concordant_T-", "T-/T-\n(n=381)", PAL["blue"]),
        ("Tau_Native_vs_6mm_1.25", "Discordant_Old-_6mm+", "Old-/6mm+\n(n=48)", PAL["coral"]),
        ("Tau_Native_vs_6mm_1.25", "Concordant_T+", "T+/T+\n(n=103)", PAL["purple"]),
    ]
    vals_r = [dc[(dc["comparison"] == c) & (dc["group"] == g)]["nd_36m_dem_rate_pct"].iloc[0] for c, g, _, _ in rows_show]
    vals_c = [dc[(dc["comparison"] == c) & (dc["group"] == g)]["mean_dcdrsb_24m"].iloc[0] for c, g, _, _ in rows_show]
    xd = np.arange(len(rows_show))
    bars = ax.bar(xd, vals_r, width=0.60, color=[r[3] for r in rows_show])
    for bar, vr, vc in zip(bars, vals_r, vals_c):
        ax.text(bar.get_x() + bar.get_width() / 2, vr + 0.7, f"{vr:.1f}%\n(+{vc:.2f})", ha="center", va="bottom", fontsize=6.7)
    ax.axvline(2.5, color="#888888", lw=0.9, ls="--")
    ax.set_xticks(xd); ax.set_xticklabels([r[2] for r in rows_show], fontsize=7.2)
    ax.set_ylim(0, 36.5)
    ax.set_ylabel("36m Dementia Progression % (24m dCDR-SB)")
    ax.set_title("A. Clinical Outcomes of Discordant Scans")

    # Panel B: 8-ROI Spatial Tau Subtyping Flips (6mm Non-PVC -> 6mm PVC, K=3)
    ax = axes[1]
    sp = su[(su["pair"] == "Tau_6mm_NonPVC_to_6mmPVC") & (su["k"] == 3)]
    algos = ["kmeans", "gmm_diag", "ward"]
    albls = ["K-means\n(K=3)", "Diag. GMM\n(K=3)", "Ward\n(K=3)"]
    xs = np.arange(3)
    w = 0.26
    for si, (scen, clr, lbl) in enumerate([
        ("A_naive_direct_transfer", PAL["crimson"], "Naive direct transfer"),
        ("B_target_restd_rescue", PAL["teal"], "Target Z-restd rescue"),
        ("C_refitted_target", PAL["navy"], "Refitted PVC model"),
    ]):
        vals = [sp[(sp["algo"] == al) & (sp["scenario"] == scen)]["flip_rate_pct"].iloc[0] for al in algos]
        bars = ax.bar(xs + (si - 1) * w, vals, width=w, color=clr, label=lbl)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.1)
    ax.set_xticks(xs); ax.set_xticklabels(albls)
    ax.set_ylim(0, 38)
    ax.set_ylabel("Individual Tau Subtype Flip Rate (%)")
    ax.set_title("B. 8-ROI Tau Spatial Subtyping (NoPVC->PVC)")
    ax.legend(frameon=True, loc="upper left", fontsize=7.3)

    # Panel C: Downstream Cox 36m Survival E/O across PET Pipeline Upgrades (Clin+PET)
    ax = axes[2]
    cp = pr[pr["panel"] == "Clin_plus_PET"]
    p_keys = ["Amyloid_FBP_Native_to_6mm", "Amyloid_6mm_to_ADSP_PHC_CL", "Tau_Native_to_6mm", "Tau_6mm_NonPVC_to_PVC"]
    p_lbls = ["FBP\nNat->6mm", "Amyloid\n6mm->PHC", "Tau\nNat->6mm", "Tau 6mm\nNoPVC->PVC"]
    xp = np.arange(len(p_keys))
    ax.axhline(1.0, color="black", lw=0.9, ls=":")
    for si, (st, clr, lbl) in enumerate([
        ("2_naive_direct_src_on_tgt", PAL["crimson"], "Naive direct transfer"),
        ("3_target_restd_rescue", PAL["teal"], "Target Z-restd rescue"),
        ("4_refitted_tgt_on_tgt", PAL["navy"], "Refitted target"),
    ]):
        vals = [cp[(cp["pair"] == pk) & (cp["strategy"] == st)]["cox_oe_36m"].iloc[0] for pk in p_keys]
        bars = ax.bar(xp + (si - 1) * w, vals, width=w, color=clr, label=lbl)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.7, rotation=90)
    ax.set_xticks(xp); ax.set_xticklabels(p_lbls, fontsize=7.6)
    ax.set_ylim(0.75, 1.78)
    ax.set_ylabel("Cox 36-Month E/O Ratio (Clin+PET)")
    ax.set_title("C. Downstream Cox Survival Calibration Shift")
    ax.legend(frameon=True, loc="upper left", fontsize=7.2)

    fig.tight_layout()
    _save(fig, fig_dir, "fig3_discordant_outcomes_subtyping_and_survival")


def make_supp_figures(res_dir: str, fig_dir: str) -> None:
    fl = pd.read_csv(os.path.join(res_dir, "pet_threshold_flips_and_fdg.csv"))
    su = pd.read_csv(os.path.join(res_dir, "tau_spatial_subtyping_transfer.csv"))
    pr = pd.read_csv(os.path.join(res_dir, "prediction_survival_pet_transfer.csv"))

    # Fig S1: Centiloid Cutoff Sensitivity Curve (6mm vs ADSP-PHC CL >= 18..37) & Tau Native->6mm Curve (1.20..1.37)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3))
    cl = fl[fl["comparison"] == "Amyloid_6mm_vs_ADSP_PHC_CL"]
    ax1.bar(cl["threshold"], cl["flip_pct"], width=0.52, color=PAL["purple"])
    for i, r in enumerate(cl.itertuples()):
        ax1.text(i, r.flip_pct + 0.15, f"{r.flip_pct:.2f}%\n({r.flip_n})", ha="center", va="bottom", fontsize=7.8)
    ax1.set_ylim(0, 7.8); ax1.set_ylabel("A+ Reclassification Rate (%)"); ax1.set_title("A. 6mm vs. ADSP-PHC Flips Across Centiloid Cutoffs")

    tn = fl[fl["comparison"] == "Tau_AV1451_Native_to_6mm"]
    ax2.bar(tn["threshold"], tn["flip_pct"], width=0.52, color=PAL["coral"])
    for i, r in enumerate(tn.itertuples()):
        ax2.text(i, r.flip_pct + 0.4, f"{r.flip_pct:.2f}%\n({r.flip_n})", ha="center", va="bottom", fontsize=7.8)
    ax2.set_ylim(0, 20.5); ax2.set_ylabel("T+ Reclassification Rate (%)"); ax2.set_title("B. AV1451 Native->6mm Flips Across Tau SUVR Cutoffs")
    fig.tight_layout()
    _save(fig, fig_dir, "figS1_cutoff_sensitivity_curves")

    # Fig S2: Spatial Tau Subtyping Heatmap across K=2,3,4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.2))
    pairs = ["Tau_Native_to_6mm", "Tau_6mm_NonPVC_to_6mmPVC"]
    plbls = ["Tau Native -> 6mm", "6mm Non-PVC -> PVC"]
    cols_o = [(al, k) for al in ("kmeans", "gmm_diag", "ward") for k in (2, 3, 4)]
    clbls = [f"{'KM' if a=='kmeans' else ('GMM' if a=='gmm_diag' else 'Ward')}\nK={k}" for a, k in cols_o]
    for ax, scen, title in [(ax1, "A_naive_direct_transfer", "A. Naive Direct Transfer Flip (%)"), (ax2, "B_target_restd_rescue", "B. Target Z-Re-standardization Flip (%)")]:
        mat = np.zeros((2, len(cols_o)))
        for i, p in enumerate(pairs):
            for j, (al, k) in enumerate(cols_o):
                mat[i, j] = su[(su["pair"] == p) & (su["algo"] == al) & (su["k"] == k) & (su["scenario"] == scen)]["flip_rate_pct"].iloc[0]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=35, aspect="auto")
        ax.set_xticks(range(len(clbls))); ax.set_xticklabels(clbls, fontsize=7.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(plbls if ax == ax1 else ["", ""])
        ax.set_title(title); ax.grid(False)
        for i in range(2):
            for j in range(len(cols_o)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.0, color="white" if v > 18 else "black")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    fig.tight_layout()
    _save(fig, fig_dir, "figS2_tau_subtyping_k_heatmap")

    # Fig S3: PET-Only vs Clin+PET 36m Logistic E/O and Brier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    p_keys = ["Amyloid_FBP_Native_to_6mm", "Amyloid_6mm_to_ADSP_PHC_CL", "Tau_Native_to_6mm", "Tau_6mm_NonPVC_to_PVC"]
    p_lbls = ["FBP Nat->6mm", "6mm->ADSP-PHC", "Tau Nat->6mm", "Tau NoPVC->PVC"]
    xp = np.arange(4); w = 0.25
    for ax, met, ylbl, title, ylim in [
        (ax1, "oe_36m", "36m MCI->Dem Logistic E/O", "A. PET-Only 36m Logistic E/O", (0.70, 1.72)),
        (ax2, "brier_36m", "36m MCI->Dem Brier Score", "B. PET-Only 36m Brier Score", (0.09, 0.18)),
    ]:
        sub = pr[pr["panel"] == "PET_only"]
        if met == "oe_36m":
            ax.axhline(1.0, color="black", lw=0.9, ls=":")
        for si, (st, clr, lbl) in enumerate([
            ("1_source_on_source", PAL["slate"], "Source baseline"),
            ("2_naive_direct_src_on_tgt", PAL["crimson"], "Naive direct"),
            ("4_refitted_tgt_on_tgt", PAL["navy"], "Refitted target"),
        ]):
            vals = [sub[(sub["pair"] == pk) & (sub["strategy"] == st)][met].iloc[0] for pk in p_keys]
            bars = ax.bar(xp + (si - 1) * w, vals, width=w, color=clr, label=lbl)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + (0.015 if met == "oe_36m" else 0.002), f"{v:.3f}" if met == "brier_36m" else f"{v:.2f}", ha="center", va="bottom", fontsize=7.0, rotation=90)
        ax.set_xticks(xp); ax.set_xticklabels(p_lbls, fontsize=7.8)
        ax.set_ylim(*ylim); ax.set_ylabel(ylbl); ax.set_title(title); ax.legend(frameon=True, loc="upper left", fontsize=7.4)
    fig.tight_layout()
    _save(fig, fig_dir, "figS3_pet_logistic_transfer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", default="outputs/pet_drift_full")
    parser.add_argument("--fig-dir", default="outputs/figures")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    _setup()
    make_fig1(args.res_dir, args.fig_dir)
    make_fig2(args.res_dir, args.fig_dir)
    make_fig3(args.res_dir, args.fig_dir)
    make_supp_figures(args.res_dir, args.fig_dir)
    print(f"Generated 6 molecular PET drift figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
