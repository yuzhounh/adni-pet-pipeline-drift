"""Programmatically generate zero-overfull booktabs LaTeX tables from PET drift CSVs."""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def _fmt(v, dec=3):
    if pd.isna(v) or not np.isfinite(v):
        return "--"
    if abs(v) < 0.5 * (10.0 ** (-dec)):
        return f"{0.0:.{dec}f}"
    if v < 0:
        return f"$-${abs(v):.{dec}f}"
    return f"{v:.{dec}f}"


def make_tables(res_dir: str, tab_dir: str) -> None:
    os.makedirs(tab_dir, exist_ok=True)
    fl = pd.read_csv(os.path.join(res_dir, "pet_threshold_flips_and_fdg.csv"))
    vis = pd.read_csv(os.path.join(res_dir, "visual_read_roc_calibration.csv"))
    dc = pd.read_csv(os.path.join(res_dir, "discordant_group_clinical_outcomes.csv"))
    pr = pd.read_csv(os.path.join(res_dir, "prediction_survival_pet_transfer.csv"))

    # Table 1: Amyloid, Tau, PVC & FDG Negative Control Threshold Reclassification
    picks = [
        ("Amyloid_FBP_Native_to_6mm", "Official A+ cutoff", "Amyloid FBP Nat.$\\to$6mm"),
        ("Amyloid_FBB_Native_to_6mm", "Official A+ cutoff", "Amyloid FBB Nat.$\\to$6mm"),
        ("Amyloid_6mm_vs_ADSP_PHC", "Official A+ cutoff", "Amyloid 6mm vs.\\ PHC"),
        ("Tau_AV1451_Native_to_6mm", "SUVR>=1.20", "Tau AV1451 Nat.$\\to$6mm"),
        ("Tau_AV1451_Native_to_6mm", "SUVR>=1.25", "Tau AV1451 Nat.$\\to$6mm"),
        ("Tau_AV1451_Native_to_6mm", "SUVR>=1.30", "Tau AV1451 Nat.$\\to$6mm"),
        ("Tau_PVC_NativeNorm_to_6mmPVC", "SUVR>=1.37", "Tau PVC Norm$\\to$6mm PVC"),
        ("Tau_6mm_NonPVC_to_6mmPVC", "SUVR>=1.25", "Tau 6mm NoPVC$\\to$PVC"),
        ("Tau_6mm_NonPVC_to_6mmPVC", "SUVR>=1.37", "Tau 6mm NoPVC$\\to$PVC"),
        ("FDG_NegativeControl_Native_to_8mm", "Q25 hypometab.", "FDG Control Nat.$\\to$8mm"),
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Cross-pipeline Amyloid, Tau, PVC and FDG (negative control) quantitative agreement and binary status reclassification across ADNI molecular PET scans.}",
        r"\label{tab:lc08_flips}",
        r"\ManuscriptTableSize{7.2}{9.0}",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"Pipeline Comparison & Cutoff & Scans & $\Delta$ Mean & $r$ & ICC(A,1) & Flip \% & $+\!\to\!-$ & $-\!\to\!+$ & Gray Zone$^{\text{a}}$ \\",
        r"\midrule",
    ]
    for i, (comp, th, lbl) in enumerate(picks):
        if i in (3, 9):
            lines.append(r"\midrule")
        r = fl[(fl["comparison"] == comp) & (fl["threshold"] == th)].iloc[0]
        gz = f"{r['gray_zone_flip_pct']:.1f}\\%" if pd.notna(r["gray_zone_flip_pct"]) else "--"
        th_tex = th.replace(">=", r"$\ge$")
        lines.append(
            f"{lbl} & {th_tex} & {int(r['n_scans']):,} & {_fmt(r['mean_diff'], 4)} & "
            f"{_fmt(r['pearson_r'], 4)} & {_fmt(r['icc_a1'], 4)} & {r['flip_pct']:.2f}\\% & "
            f"{int(r['pos_to_neg'])} & {int(r['neg_to_pos'])} & {gz} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Notes.} $^{\text{a}}$Amyloid A+ reclassification restricted to the 15--35 Centiloid clinical-trial gray zone.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    with open(os.path.join(tab_dir, "table1_pet_flips.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Table 2: Clinical Visual-Read Gold Standard Calibration (Tau Gothenburg & Amyloid AMYREAD)
    pmap = {
        "Tau_Native_AV1451": "Tau Native AV1451",
        "Tau_6mm_NonPVC": "Tau 6mm Non-PVC",
        "Tau_Native_PVC_Norm": "Tau Native PVC (Ref-Norm)",
        "Tau_6mm_PVC": "Tau 6mm PVC",
        "Amyloid_UCB_6mm_CL": "Amyloid UCB 6mm (CL)",
        "Amyloid_ADSP_PHC_CL": "Amyloid ADSP-PHC (CL)",
    }
    lines2 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Independent clinical visual-read gold-standard ROC discrimination, pipeline-specific Youden optimal cutoffs and fixed-cutoff accuracy loss (Gothenburg Tau visual reads; \texttt{AMYREAD} Amyloid reads).}",
        r"\label{tab:lc08_visual}",
        r"\ManuscriptTableSize{7.6}{9.4}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"PET Pipeline & $n$ & AUROC & Opt.\ Thr & Sens.\ & Spec.\ & Opt.\ Acc & Acc@1.20 & Acc@1.30 \\",
        r"\midrule",
    ]
    for i, (_, r) in enumerate(vis.iterrows()):
        if i == 4:
            lines2.append(r"\midrule")
        a120 = f"{r['acc_at_1.20']:.1f}\\%" if pd.notna(r.get("acc_at_1.20")) else "--"
        a130 = f"{r['acc_at_1.30']:.1f}\\%" if pd.notna(r.get("acc_at_1.30")) else f"{r['official_binary_acc']:.1f}\\%$^*$"
        lines2.append(
            f"{pmap[r['pipeline']]} & {int(r['n'])} & {_fmt(r['auroc'], 4)} & {_fmt(r['opt_thresh'], 3)} & "
            f"{r['opt_sens']:.1f}\\% & {r['opt_spec']:.1f}\\% & {r['opt_acc']:.1f}\\% & {a120} & {a130} \\\\"
        )
    lines2 += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Notes.} $^*$Official binary A+ status agreement with \texttt{AMYREAD} visual reads (90.5\% UCB 6mm vs.\ 89.1\% ADSP-PHC).",
        r"\end{minipage}",
        r"\end{table}",
    ]
    with open(os.path.join(tab_dir, "table2_visual_gold_standard.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines2) + "\n")

    # Table 3: Clinical Progression of Concordant vs Discordant Groups
    gmap = {
        "Concordant_A-": "Concordant A$-$/A$-$",
        "Discordant_6mm+_PHC-": "Discordant 6mm$+$ / PHC$-$",
        "Discordant_6mm-_PHC+": "Discordant 6mm$-$ / PHC$+$",
        "Concordant_A+": "Concordant A$+$/A$+$",
        "Concordant_T-": "Concordant T$-$/T$-$",
        "Discordant_Old-_6mm+": "Discordant Old$-$ / 6mm$+$",
        "Discordant_Old+_6mm-": "Discordant Old$+$ / 6mm$-$",
        "Concordant_T+": "Concordant T$+$/T$+$",
    }
    lines3 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Longitudinal 36-month dementia conversion and 24-month $\Delta$CDR-SB progression among pipeline-concordant versus pipeline-discordant ADNI participants.}",
        r"\label{tab:lc08_discordant}",
        r"\ManuscriptTableSize{7.4}{9.2}",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Comparison & Status Group & $N$ & Mean 6mm & $N_{36\text{m}}$ & 36m Dem & MCI 36m & 24m $\Delta$CDR-SB \\",
        r"\midrule",
    ]
    order_d = [
        ("Amyloid_6mm_vs_ADSP_PHC", "Concordant_A-"),
        ("Amyloid_6mm_vs_ADSP_PHC", "Discordant_6mm+_PHC-"),
        ("Amyloid_6mm_vs_ADSP_PHC", "Discordant_6mm-_PHC+"),
        ("Amyloid_6mm_vs_ADSP_PHC", "Concordant_A+"),
        ("Tau_Native_vs_6mm_1.25", "Concordant_T-"),
        ("Tau_Native_vs_6mm_1.25", "Discordant_Old-_6mm+"),
        ("Tau_Native_vs_6mm_1.25", "Discordant_Old+_6mm-"),
        ("Tau_Native_vs_6mm_1.25", "Concordant_T+"),
    ]
    for i, (comp, grp) in enumerate(order_d):
        if i == 4:
            lines3.append(r"\midrule")
        r = dc[(dc["comparison"] == comp) & (dc["group"] == grp)].iloc[0]
        c_lbl = "Amyloid 6mm/PHC" if i == 0 else ("Tau Nat./6mm (1.25)" if i == 4 else "")
        mr = f"{r['mci_36m_conv_rate_pct']:.1f}\\%" if pd.notna(r["mci_36m_conv_rate_pct"]) else "--"
        lines3.append(
            f"{c_lbl} & {gmap[grp]} & {int(r['n_total']):,} & {_fmt(r['mean_cl_6mm'], 2)} & "
            f"{int(r['nd_36m_dem_n']):,} & {r['nd_36m_dem_rate_pct']:.2f}\\% & {mr} & {_fmt(r['mean_dcdrsb_24m'], 3)} \\\\"
        )
    lines3 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tab_dir, "table3_discordant_outcomes.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines3) + "\n")

    # Table S1: Downstream Prediction & Cox Survival Transfer Summary (Clin+PET)
    lines_s1 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Downstream 36-month MCI-to-dementia logistic prediction and Cox survival calibration across molecular PET pipeline upgrades (Clinical+PET).}",
        r"\label{tab:lc08_s1_prediction}",
        r"\ManuscriptTableSize{7.6}{9.4}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Upgrade Pair & Strategy & $N_{\text{ev}}$ & BS36 & AUROC & Log.\ E/O & Cox $C$ & Cox E/O & Cox Bias \\",
        r"\midrule",
    ]
    cp = pr[pr["panel"] == "Clin_plus_PET"]
    pm_p = {"Amyloid_FBP_Native_to_6mm": "FBP Nat.$\\to$6mm", "Amyloid_6mm_to_ADSP_PHC_CL": "6mm$\\to$ADSP-PHC", "Tau_Native_to_6mm": "Tau Nat.$\\to$6mm", "Tau_6mm_NonPVC_to_PVC": "Tau 6mm NoPVC$\\to$PVC"}
    sm_p = {"1_source_on_source": "Source baseline", "2_naive_direct_src_on_tgt": "Naive direct", "3_target_restd_rescue": "Target Z-restd", "4_refitted_tgt_on_tgt": "Refitted target"}
    for pi, pk in enumerate(pm_p.keys()):
        if pi > 0:
            lines_s1.append(r"\midrule")
        for si, st in enumerate(sm_p.keys()):
            r = cp[(cp["pair"] == pk) & (cp["strategy"] == st)].iloc[0]
            p_s = pm_p[pk] if si == 0 else ""
            lines_s1.append(
                f"{p_s} & {sm_p[st]} & {int(r['eval_n_36m'])} & {_fmt(r['brier_36m'], 4)} & "
                f"{_fmt(r['auroc_36m'], 4)} & {_fmt(r['oe_36m'], 3)} & {_fmt(r['cox_c_index'], 4)} & "
                f"{_fmt(r['cox_oe_36m'], 3)} & {_fmt(r['cox_bias_36m_pct'], 2)}\\% \\\\"
            )
    lines_s1 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tab_dir, "tableS1_prediction_survival.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_s1) + "\n")

    print(f"Generated 4 molecular PET LaTeX tables in {tab_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", default="outputs/pet_drift_full")
    parser.add_argument("--tab-dir", default="outputs/tables")
    args = parser.parse_args()
    make_tables(args.res_dir, args.tab_dir)


if __name__ == "__main__":
    main()
