from ultralytics import YOLO
import torch
import os
import pandas as pd
import shutil
from datetime import datetime
import multiprocessing

# =========================================================
# CONFIG
# =========================================================

DATASET_YAML = (
    r"C:\seg_uncertain\phase2_yolo_training\dataset.yaml"
)

PRETRAINED_MODEL = (
    r"C:\Polygon_polyp_new(05_26)\yolov8n-seg.pt"
)

OUTPUT_ROOT = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
)

PROJECT_NAME = "YOLO_REFEREE_SEED123"

SEED = 123

IMG_SIZE = 640

# PILOT TRAINING
EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 10

DEVICE = 0

# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    # =====================================================
    # CUDA CHECK
    # =====================================================

    print("\n" + "=" * 60)
    print("CUDA CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA NOT AVAILABLE"
        )

    gpu_count = torch.cuda.device_count()

    gpu_name = torch.cuda.get_device_name(
        DEVICE
    )

    print(f"GPU COUNT : {gpu_count}")
    print(f"USING GPU : {DEVICE}")
    print(f"GPU NAME  : {gpu_name}")
    print(
        f"CUDA VERSION : "
        f"{torch.version.cuda}"
    )

    # =====================================================
    # DATASET CHECK
    # =====================================================

    print("\n" + "=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    dataset_root = (
        r"C:\seg_uncertain\phase2_yolo_training"
    )

    train_images = os.path.join(
        dataset_root,
        "images",
        "train"
    )

    train_labels = os.path.join(
        dataset_root,
        "labels",
        "train"
    )

    val_images = os.path.join(
        dataset_root,
        "images",
        "val"
    )

    val_labels = os.path.join(
        dataset_root,
        "labels",
        "val"
    )

    n_train_images = len(
        os.listdir(train_images)
    )

    n_train_labels = len(
        os.listdir(train_labels)
    )

    n_val_images = len(
        os.listdir(val_images)
    )

    n_val_labels = len(
        os.listdir(val_labels)
    )

    print(
        f"Train Images : {n_train_images}"
    )

    print(
        f"Train Labels : {n_train_labels}"
    )

    print(
        f"Val Images   : {n_val_images}"
    )

    print(
        f"Val Labels   : {n_val_labels}"
    )

    assert (
        n_train_images ==
        n_train_labels
    ), "Train mismatch"

    assert (
        n_val_images ==
        n_val_labels
    ), "Val mismatch"

    # =====================================================
    # SAVE CONFIG
    # =====================================================

    config = pd.DataFrame([{

        "date":
            datetime.now(),

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "batch":
            BATCH_SIZE,

        "img_size":
            IMG_SIZE,

        "patience":
            PATIENCE,

        "gpu":
            gpu_name

    }])

    config.to_csv(

        os.path.join(
            OUTPUT_ROOT,
            "experiment_config.csv"
        ),

        index=False
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = YOLO(
        PRETRAINED_MODEL
    )

    # =====================================================
    # TRAIN
    # =====================================================

    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)

    model.train(

        data=DATASET_YAML,

        epochs=EPOCHS,

        imgsz=IMG_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=8,

        seed=SEED,

        deterministic=True,

        patience=PATIENCE,

        optimizer="AdamW",

        lr0=1e-3,

        weight_decay=5e-4,

        cos_lr=True,

        amp=True,

        cache=False,

        project=OUTPUT_ROOT,

        name=PROJECT_NAME,

        exist_ok=True,

        save=True,

        plots=True,

        verbose=True
    )

    # =====================================================
    # OUTPUT CHECKS
    # =====================================================

    run_dir = os.path.join(
        OUTPUT_ROOT,
        PROJECT_NAME
    )

    weights_dir = os.path.join(
        run_dir,
        "weights"
    )

    best_model = os.path.join(
        weights_dir,
        "best.pt"
    )

    last_model = os.path.join(
        weights_dir,
        "last.pt"
    )

    print("\n" + "=" * 60)
    print("VERIFY OUTPUTS")
    print("=" * 60)

    required_files = [

        best_model,

        last_model,

        os.path.join(
            run_dir,
            "results.csv"
        )
    ]

    for f in required_files:

        if os.path.exists(f):

            print(
                f"[OK] {os.path.basename(f)}"
            )

        else:

            print(
                f"[MISSING] {f}"
            )

    # =====================================================
    # COPY IMPORTANT FILES
    # =====================================================

    important_dir = os.path.join(
        OUTPUT_ROOT,
        "important_outputs"
    )

    os.makedirs(
        important_dir,
        exist_ok=True
    )

    files_to_copy = [

        "results.csv",

        "results.png",

        "PR_curve.png",

        "P_curve.png",

        "R_curve.png",

        "F1_curve.png",

        "confusion_matrix.png",

        "confusion_matrix_normalized.png"
    ]

    for file in files_to_copy:

        src = os.path.join(
            run_dir,
            file
        )

        if os.path.exists(src):

            shutil.copy2(

                src,

                os.path.join(
                    important_dir,
                    file
                )
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Run Directory:\n{run_dir}"
    )

    print(
        f"\nBest Model:\n{best_model}"
    )

    print(
        f"\nLast Model:\n{last_model}"
    )


# =========================================================
# WINDOWS SAFE ENTRY POINT
# =========================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()