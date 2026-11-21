# Revision analyses (BSPC major revision)

Scripts, derived data and outputs for every analysis added during the revision.
All commands are run from this `revision/` folder and use the released per-image
files in `../sample_results`.

## Folder contents
- `R*.py` – analysis scripts (R1–R9; the letter is the reviewer-comment group).
- `data/` – derived per-image inputs (no images or masks are redistributed).
- `outputs/` – the printed outputs and result tables reported in the paper.
- `figures/` – revised Figures 3 and 4.

## Reproducing the reported results (no GPU needed)

| Paper item | Command |
|---|---|
| Primary-model confidence baseline (Tables 5 and 7) | `python R1_evaluate_confidence_baseline.py data/primary_confidence.csv ../sample_results` |
| Fig. 4 | `python R1_figure4_risk_coverage.py ../sample_results data/primary_confidence.csv` |
| Sequence-level reanalysis (Table S11) | `python R4_sequence_level_analysis.py ../sample_results data/primary_confidence.csv` |
| Referee quality and error dependence (Table S12) | `python R5C_analyze_referee_quality.py ../sample_results data/primary_confidence.csv data/referee_quality.csv` |
| Per-source common subset (Table S18) | `python R5_per_source_common_subset.py ../sample_results data/primary_confidence.csv` |
| Baseline paired tests (Table 7, Table S13) | `python R6_baseline_paired_tests.py ../sample_results data/primary_confidence.csv data/morphology_features.csv data/tta_uncertainty_features.csv` |
| Fig. 3 | `python R6_figure3_baselines.py` |
| Per-source, PolypDB and condition strata (Table 8, Tables S14–S16) | `python R7C_generalization_analysis.py ../sample_results data/primary_confidence.csv data/morphology_features.csv data/tta_uncertainty_features.csv data/polypdb_results.csv data/image_conditions.csv` |
| Operating points and triage (Table 9, Table S17) | `python R8_operating_points.py ../sample_results data/primary_confidence.csv data/polypdb_results.csv` |
| Resolution-matched YOLO, 1,248-image set, high-agreement failures, common subset (Tables 4, S19–S21) | `python R3C_reviewer3_analysis.py ../sample_results data/primary_confidence.csv data/referee_quality.csv data/yolo512_referee.csv data/agreement_features_independent.csv data/morphology_features.csv data/tta_uncertainty_features.csv data/polypdb_results.csv data/polypdb_yolo512.csv` |
| Matched-referee paired tests on 1,046 images (Table S11) | `python R9_matched_referee_tests.py ../sample_results data/primary_confidence.csv data/referee_quality.csv data/yolo512_referee.csv` |

`R4_sequence_level_analysis.py` must stay in the same folder; the other analysis scripts import it.
Each command reproduces the corresponding file in `outputs/` exactly.

## Generating the derived data (GPU, datasets and trained weights required)

| Output file | Script |
|---|---|
| `data/primary_confidence.csv` | `R1_extract_primary_confidence.py` |
| `data/yolo512_referee.csv` (trains the 512 × 512 Independent YOLO Referee) | `R3A_yolo_referee_512.py` |
| `data/polypdb_yolo512.csv` | `R3B_yolo512_polypdb.py` |
| Protocol-matched UNet++ referee | `R5B_train_unetpp_matched.py` |
| `data/referee_quality.csv` | `R5A_referee_quality.py` |
| `data/image_conditions.csv` | `R7A_image_conditions.py` |
| `data/polypdb_results.csv`, `data/polypdb_overlap_report.csv` | `R7B_polypdb_external.py` |

These scripts contain Windows paths at the top; edit them for your system. PolypDB is
available from its authors (Jha et al., arXiv:2409.00045) and is not redistributed here.
