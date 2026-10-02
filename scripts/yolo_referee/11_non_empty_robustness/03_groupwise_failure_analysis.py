import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee"

INPUT_CSV = os.path.join(
    ROOT,
    "11_non_empty_robustness",
    "annotated_evaluation.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "11_non_empty_robustness",
    "groupwise_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "groupwise_failure_analysis.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("GROUPWISE FAILURE ANALYSIS")
print("=" * 70)

print(f"Images Loaded : {len(df)}")

required_columns = [
    "failure",
    "agreement_dice",
    "agreement_iou",
    "boundary_agreement",
    "area_ratio",
    "centroid_distance",
    "primary_empty",
    "referee_empty"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("All required columns found.")
print()

# ============================================================
# DEFINE GROUPS
# ============================================================

groups = {

    "Both Empty":
        (df["primary_empty"]) &
        (df["referee_empty"]),

    "Primary Empty Only":
        (df["primary_empty"]) &
        (~df["referee_empty"]),

    "Referee Empty Only":
        (~df["primary_empty"]) &
        (df["referee_empty"]),

    "Both Non-Empty":
        (~df["primary_empty"]) &
        (~df["referee_empty"])

}

# ============================================================
# ANALYSIS
# ============================================================

rows = []

for name, mask in groups.items():

    subset = df[mask].copy()

    n = len(subset)

    failures = int(subset["failure"].sum())

    failure_rate = (
        failures / n * 100
        if n > 0 else 0
    )

    row = {

        "Group":
            name,

        "Images":
            n,

        "Failures":
            failures,

        "Failure_Rate_%":
            round(failure_rate, 2),

        "Mean_Agreement_Dice":
            subset["agreement_dice"].mean(),

        "Std_Agreement_Dice":
            subset["agreement_dice"].std(),

        "Mean_Agreement_IoU":
            subset["agreement_iou"].mean(),

        "Mean_Boundary_Agreement":
            subset["boundary_agreement"].mean(),

        "Mean_Area_Ratio":
            subset["area_ratio"].mean(),

        "Mean_Centroid_Distance":
            subset["centroid_distance"].mean()

    }

    rows.append(row)

# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_CSV,
    index=False
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option(
    "display.float_format",
    lambda x: f"{x:.4f}"
)

print(results)

print()

print("=" * 70)
print("TOTAL CHECK")
print("=" * 70)

print("Images   :", results["Images"].sum())
print("Failures :", results["Failures"].sum())

print()

print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_CSV)

print()

print("GROUPWISE FAILURE ANALYSIS COMPLETE")