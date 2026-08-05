# Referee-Based Quality Estimation (RBQE)

Official implementation accompanying the manuscript

**"Referee-Based Quality Estimation: Agreement Between Independent Segmentation Models as a Reliability Indicator for Polyp Segmentation"**

---

## Overview

RBQE is a deployment-time no-reference segmentation quality estimation framework.

Unlike uncertainty estimation methods that repeatedly evaluate a single segmentation model, RBQE estimates prediction reliability by measuring agreement between independently trained segmentation models.

The repository contains:

- Source code
- Reproducibility notebook
- Sample experimental results
- Figure generation scripts

---

## Repository Structure

```
RBQE
│
├── README.md
├── LICENSE
├── requirements.txt
├── RBQE_Reproducibility.ipynb
│
├── scripts/
├── figures/
├── sample_results/
└── docs/
```

---

## Installation

```bash
git clone https://github.com/sidgupta307/RBQE.git

cd RBQE

pip install -r requirements.txt
```

---

## Reproducing Results

Open

```
RBQE_Reproducibility.ipynb
```

and execute every cell sequentially.

The notebook reproduces

- ROC analysis
- Bootstrap confidence intervals
- DeLong significance tests
- Threshold sensitivity
- Risk-Coverage analysis
- Non-empty robustness
- Feature correlation

using the supplied sample results.

---

## Citation

If this repository contributes to your work, please cite the associated publication.

Citation will be updated after publication.

---

## Contact

Siddharth Gupta
Dr. Jitin Singla

Department of Biosciences and Bioengineering

Indian Institute of Technology Roorkee

India