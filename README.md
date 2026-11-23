# ADNI Molecular PET 6mm Reprocessing, PVC Shift and Visual-Read Calibration

Standalone pure-Python pipeline for **Molecular PET 6mm reprocessing, partial-volume correction and Centiloid harmonization reclassify Alzheimer's trial eligibility and require visual-read threshold recalibration**.

## Overview

This repository audits paired ADNI molecular PET scans across native-resolution pipelines, 6mm isotropic smoothing, partial-volume correction (PVC), and ADSP-PHC Centiloid harmonization, benchmarked against two blind clinical visual-read gold standards (`GOTHENBURG_VISUAL_READS_AV1451` and `AMYREAD`) plus an FDG PET negative control:

1. **Binary A+/T+ and joint AT reclassification**: Global and 15--35 Centiloid clinical-trial gray-zone flips across FBP, FBB, ADSP-PHC, AV1451, and PVC boundaries.
2. **Visual-read ROC calibration**: Pipeline-specific Youden-optimal SUVR and Centiloid cutoffs quantifying legacy-cutoff accuracy penalties.
3. **8-ROI spatial tau subtyping**: K-means, diagonal GMM, and Ward transfer and target z-re-standardization rescue across 6mm and PVC transitions.
4. **Discordant clinical progression and downstream survival transfer**: Scan-anchored 36-month dementia conversion, 24-month CDR-SB progression, and Cox survival calibration.

## Quickstart

```bash
pip install -r requirements.txt

# Run pure-synthetic unit tests (no restricted data required)
python tests/test_pet_drift_engine.py

# Run pilot or full four-axis pipeline against local ADNI files
python run_pet_pipeline_drift.py --pilot --out-dir outputs
python run_pet_pipeline_drift.py --out-dir outputs
python make_figures.py --res-dir outputs/pet_drift_full --fig-dir outputs/figures
python make_tables.py --res-dir outputs/pet_drift_full --tab-dir outputs/tables
```
