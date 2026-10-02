import os
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# PATHS
# ============================================================

MASTER_CSV = r"C:\seg_uncertain\chat_gpt\20B_segformer_clean_agreement\clean_agreement_features.csv"

UNET_CSV = r"C:\seg_uncertain\chat_gpt\24_unetpp_agreement\agreement_features.csv"

YOLO_CSV = r"C:\seg_uncertain\chat_gpt\01_external_validation\combined_results.csv"

OUTPUT_DIR = r"C:\seg_uncertain\journal_extension\T1_03_UNetPP_1223"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

master = pd.read_csv(MASTER_CSV)
unet = pd.read_csv(UNET_CSV)
yolo = pd.read_csv(YOLO_CSV)

# ============================================================
# KEEP ONLY YOLO RESULTS
# ============================================================

yolo = yolo[yolo["model"] == "YOLOv8-Seg"].copy()

# ============================================================
# IMAGE KEY
# ============================================================

for df in [master, unet, yolo]:

    df["dataset"] = (
        df["dataset"]
        .str.lower()
        .str.strip()
    )

    df["image"] = (
        df["image"]
        .str.lower()
        .str.strip()
        .apply(lambda x: os.path.splitext(x)[0])
    )

# ============================================================
# FILTER TO CANONICAL 1223 IMAGES
# ============================================================

master_key = master[
    ["dataset", "image"]
].drop_duplicates()

unet = unet.merge(
    master_key,
    on=["dataset", "image"],
    how="inner"
)

yolo = yolo.merge(
    master_key,
    on=["dataset", "image"],
    how="inner"
)

print("Master :", len(master_key))
print("UNet   :", len(unet))
print("YOLO   :", len(yolo))

assert len(unet) == 1223
assert len(yolo) == 1223

# ============================================================
# MERGE
# ============================================================

merged = unet.merge(

    yolo[
        [
            "dataset",
            "image",
            "dice"
        ]
    ],

    on=[
        "dataset",
        "image"
    ]

)

print("Merged :", len(merged))

assert len(merged) == 1223

# ============================================================
# FAILURE LABEL
# ============================================================

merged["failure"] = (

    merged["dice"] < 0.50

).astype(int)

print()

print("Failures     :", merged.failure.sum())
print("Non Failures :", len(merged)-merged.failure.sum())

# ============================================================
# FEATURES
# ============================================================

FEATURES = {

    "Agreement Dice":"agreement_dice",

    "Agreement IoU":"agreement_iou",

    "Area Ratio":"area_ratio",

    "Boundary Agreement":"boundary_agreement",

    "Centroid Distance":"centroid_distance"

}

results=[]

# ============================================================
# ROC
# ============================================================

for name,feature in FEATURES.items():

    df = merged[
        [feature,"failure"]
    ].dropna()

    y_true = df.failure.values
    scores = df[feature].values

    if feature!="centroid_distance":

        scores = -scores

    auc = roc_auc_score(
        y_true,
        scores
    )

    fpr,tpr,thr = roc_curve(
        y_true,
        scores
    )

    idx = (tpr-fpr).argmax()

    threshold = thr[idx]

    pred = scores>=threshold

    acc = accuracy_score(
        y_true,
        pred
    )

    prec = precision_score(
        y_true,
        pred,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0
    )

    results.append({

        "Signal":name,

        "ROC_AUC":auc,

        "Threshold":threshold,

        "Accuracy":acc,

        "Precision":prec,

        "Recall":rec,

        "F1":f1

    })

results = pd.DataFrame(results)

results = results.sort_values(
    "ROC_AUC",
    ascending=False
)

print()
print(results)

# ============================================================
# SAVE
# ============================================================

merged.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "merged_1223.csv"
    ),

    index=False

)

results.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "failure_detection_results.csv"
    ),

    index=False

)

print()
print("Done.")