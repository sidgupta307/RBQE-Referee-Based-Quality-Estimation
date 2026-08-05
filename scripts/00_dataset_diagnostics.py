import os
import cv2
import numpy as np


PRIMARY_ROOT = r"C:\seg_uncertain\chat_gpt\02_external_predictions"
REFEREE_ROOT = r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee\03_external_referee_predictions"


def load_masks(root):
    masks = {}

    for dataset in sorted(os.listdir(root)):

        dataset_dir = os.path.join(root, dataset, "masks")

        if not os.path.isdir(dataset_dir):
            continue

        masks[dataset] = {}

        for fname in sorted(os.listdir(dataset_dir)):

            if not fname.lower().endswith(".png"):
                continue

            path = os.path.join(dataset_dir, fname)

            img = cv2.imread(path, 0)

            area = int(np.count_nonzero(img))

            masks[dataset][fname] = area

    return masks


primary = load_masks(PRIMARY_ROOT)
referee = load_masks(REFEREE_ROOT)


total = 0
primary_empty = 0
referee_empty = 0
both_empty = 0
only_primary = 0
only_referee = 0
both_non_empty = 0


for dataset in primary:

    if dataset not in referee:
        continue

    common = set(primary[dataset].keys()) & set(referee[dataset].keys())

    for name in sorted(common):

        total += 1

        p = primary[dataset][name] == 0
        r = referee[dataset][name] == 0

        if p:
            primary_empty += 1

        if r:
            referee_empty += 1

        if p and r:
            both_empty += 1

        elif p:
            only_primary += 1

        elif r:
            only_referee += 1

        else:
            both_non_empty += 1


print("=" * 60)
print("DATASET DIAGNOSTICS")
print("=" * 60)

print(f"Total matched images      : {total}")
print(f"Primary empty            : {primary_empty}")
print(f"Referee empty            : {referee_empty}")
print(f"Both empty               : {both_empty}")
print(f"Only primary empty       : {only_primary}")
print(f"Only referee empty       : {only_referee}")
print(f"Both non-empty           : {both_non_empty}")

print("=" * 60)