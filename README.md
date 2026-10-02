# RBQE: Referee-Based Quality Estimation for Polyp Segmentation

Code and derived results for:

> S. Gupta, J. Singla. **Cross-Model Agreement as a Deployment-Time Reliability Signal for Automatic Polyp Segmentation.** Submitted to *Biomedical Signal Processing and Control*.

RBQE estimates the reliability of a deployed segmentation (YOLOv8n-Seg) from its pairwise Agreement Dice with a separately trained referee (Independent YOLO, SegFormer-B0, UNet++, or prompt-coupled MedSAM). No ground truth is needed at deployment; ground truth is used only for retrospective evaluation.

## Data
All datasets are public third-party datasets and are **not** redistributed here:
Kvasir-SEG (training), CVC-ClinicDB, CVC-ColonDB, ETIS-Larib PolypDB, CVC-300 (external evaluation).
Download them from their original sources and set the paths at the top of each script.

## Benchmark conventions
- Standardized external benchmark: **1,223 images** (1,248 external images minus 25 images in which both the primary model and SegFormer-B0 were empty).
- Failure: ground-truth **DSC < 0.50** (236 failures, 987 non-failures).
- Agreement descriptors are oriented so that lower agreement = higher failure risk (Centroid Distance: higher = higher risk).
- Boundary Agreement = Dice between one-pixel boundary rings (mask XOR single-iteration binary erosion).

## Reproducing the reported results

| Paper item | Result | Script |
|---|---|---|
| Table S1 | Independent YOLO Referee training (seed 123) | `scripts/01_train_independent_yolo_referee.py` |
| Table S1 | Primary YOLOv8n-Seg training (seed 42) | `scripts/primary/03_train_yolov8_seg.py` |
| Table 2 | Primary-model external segmentation performance | `scripts/primary/01_external_validation_yolo_segformer.py` |
| — | External referee predictions | `scripts/02_external_validation_yolo_referee.py`, `scripts/03_generate_external_referee_predictions.py` |
| Tables 3, S4, S7, S9 | Independent YOLO agreement features and evaluation | `scripts/04_extract_yolo_seed_agreement.py`, `scripts/05_standardized_yolo_referee_evaluation.py` |
| Table S5 | Independent YOLO bootstrap CI | `scripts/yolo_referee/06_bootstrap_auc.py` |
| Table S7 | Independent YOLO PR-AUC | `scripts/yolo_referee/08_pr_auc.py` |
| Table S2, Fig. S1 | Failure-threshold sensitivity | `scripts/yolo_referee/09_threshold_sensitivity.py`, `scripts/figures/figS1_threshold_sensitivity.py` |
| Table S8, Fig. 4 | Risk–coverage | `scripts/yolo_referee/10_risk_coverage.py` |
| Table 3, Table 4, Table S9 | Non-empty robustness and empty-mask strata (975 images) | `scripts/yolo_referee/11_non_empty_robustness/` |
| Tables 3, S3, S4 | SegFormer-B0 agreement features | `scripts/segformer/20_segformer_agreement_features.py` |
| Table S3 | SegFormer-B0 descriptor-level results (ROC-AUC + Youden metrics) | `scripts/analysis/segformer_descriptor_table_s3.py` |
| Table S5 | SegFormer-B0 bootstrap CI | `scripts/segformer/22A_segformer_bootstrap_auc_clean.py` |
| §5.3, Table S4 | Paired DeLong tests (full and 975-image paired) | `scripts/analysis/paired_delong_and_restricted.py` |
| Tables 3, S4–S6 | UNet++ evaluation and bootstrap | `scripts/unetpp/` |
| Tables 3, S4–S6 | MedSAM (prompt-coupled) evaluation and bootstrap | `scripts/medsam/` |
| Tables 3, 5 | Restricted 1,046-image (non-empty primary) analysis | `scripts/restricted/` |
| Table 5, Fig. 3 | TTA and morphology baselines | `scripts/baselines/` |
| §5.3, Supp. §11 | Deduplicated sensitivity analysis (CVC-300, CVC-ColonDB and ETIS-Larib overlap) | `scripts/analysis/dedup_sensitivity.py` |
| All | Automated check of every number derivable from the released files | `scripts/analysis/verify_paper_numbers.py` |
| Figs. S1, S4 | Supplementary figures | `scripts/figures/` |

Quick check with the released per-image files:
```bash
pip install -r requirements.txt
python scripts/analysis/verify_paper_numbers.py sample_results sample_results/clean_agreement_features.csv   # recomputes 65 manuscript values: expect 65 PASS, 0 FAIL
python scripts/analysis/paired_delong_and_restricted.py sample_results
# Full benchmark: SegFormer-B0 0.9601 vs Independent YOLO 0.9231, dAUC = 0.0370, p = 6.32e-05
# Paired 975-image analysis: 0.88525 vs 0.78257, dAUC = 0.10268, p = 0.00891
```

## Notes
- Scripts contain absolute Windows paths at the top of each file; edit them to match your local data and output folders.
- UNet++ referee training: `scripts/unetpp/Unet_plus_plus.ipynb`, cells 3-6 (ResNet34, Adam 1e-4, batch 8, 25 epochs, 256x256; saves `best_unetplusplus_model.pth`). Later cells in that notebook are unrelated experiments.
- Evaluation is retrospective. Image-level bootstrap and DeLong tests do not model within-sequence correlation between frames.

## Licence and citation
Please cite the paper above if you use this code.
