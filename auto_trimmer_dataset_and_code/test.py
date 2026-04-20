# import sys
# import os
# from pathlib import Path
# import json
# import matplotlib as plt

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)
# from dataset_loader import load_trim_dataset
# from autotrimmer_find_params import evaluate_config
# from autotrimmer_find_params import getScore


# cfg_path = Path(
#     r"C:\Users\User\Desktop\diplom\auto_trimmer_dataset_and_code\valid_hard\cfg_trim_back_end.json"
# )
# with cfg_path.open("r", encoding="utf8") as f:
#     trimmer_cfg = json.load(f)


# dataset = load_trim_dataset(
#     r"C:\Users\User\Desktop\diplom\auto_trimmer_dataset_and_code\valid_hard"
# )
# metrics = evaluate_config(dataset, trimmer_cfg)
# print(metrics, f"Score: {getScore(metrics)}")


from pathlib import Path
import json

root = Path(r"C:\Users\User\Desktop\diplom\dataset_collection")
gt_files = list(root.rglob("*_gt.json"))
vug = 0
den = 0
mash = 0
masha = {"how_are_you": 0, "thanks": 0, "bye": 0, "hello": 0}
denis = {"how_are_you": 0, "thanks": 0, "bye": 0, "hello": 0}
for gt_path in gt_files:
    with gt_path.open("r", encoding="utf-8") as f:
        gt = json.load(f)

    gt_root = gt_path.parent
    meta_root = gt_root / gt_path.name.replace("_gt.json", "_meta.json")
    with meta_root.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if "valid" in gt:
        if meta["subject_id"] == "Masha":
            masha[meta["gesture_label"]] += 1
        elif meta["subject_id"] == "Denis":
            denis[meta["gesture_label"]] += 1

print(f"Masha: {masha}, Denis: {denis}")
