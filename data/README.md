# Data Directory

This repository contains **no raw ADNI PET tables**, participant-level SUVRs/Centiloids, or predictions.

## Required External Inputs (Under Approved ADNI Data Use Agreement)

Obtain official study files from the [LONI Image and Data Archive](https://ida.loni.usc.edu):

1. Live `DXSUM*.csv` placed in `data/raw/adni_route_a/` (or `ADNI_RAW_DIR` / `--adni-raw-dir`).
2. Extracted CSV tables from `ADNIMERGE 0.0.1` (`v2025_07/csv/`: `ucberkeleyav45.csv`, `ucberkeleyfbb.csv`, `ucberkeleyav1451.csv`, `ucberkeleyav1451_pvc.csv`, `ucberkeleyfdg.csv`) and `ADNIMERGE2` (`v2026_10/csv/`: `ADSL.csv`, `ADQS.csv`, `PTDEMOG.csv`, `CDR.csv`, `UCBERKELEY_AMY_6MM.csv`, `UCBERKELEY_TAU_6MM.csv`, `UCBERKELEY_TAUPVC_6MM.csv`, `UCBERKELEYFDG_8mm.csv`, `ADSP_PHC_PET_Amyloid_Simple.csv`, `AMYREAD.csv`, `GOTHENBURG_VISUAL_READS_AV1451.csv`) placed under `data/raw/adni_extract/` (or `ADNI_EXTRACT_DIR` / `--adni-extract-dir`).

All files under `data/raw/`, `data/derived/`, and `outputs/` are excluded by `.gitignore`.
