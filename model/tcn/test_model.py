# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


import torch
from torch.utils.data import DataLoader
from load_data import load_records
from gesture_ds_class import GestureDataset
from trimmer.autotrimmer import AutoTrimmer
from tcn import GestureTCN
import config as cfg
import preprocessing
from sklearn.metrics import classification_report, confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Loading model and configurations...")

    checkpoint_path = Path(cfg.ARTIFACTS_PATH) / "best_tcn.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Haven't found checkpoint {checkpoint_path}!")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    label2idx = checkpoint["label2idx"]
    idx2label = checkpoint["idx2label"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    target_len = checkpoint["target_len"]
    tcn_cfg = checkpoint["tcn_cfg"]
    sens_cfg = checkpoint["sens_cfg"]

    trimmer = AutoTrimmer(checkpoint["trimmer_cfg"], sens_cfg)
    test_records = [
        r for r in load_records(cfg.DATA_ROOT) if r["subject_id"] in cfg.TEST_ON
    ]
    prep_kwargs = dict(
        trimmer=trimmer,
        n_hall=cfg.SENS_CFG["n_hall"],
        add_hall_diff=cfg.PREPROCESSING_CFG["add_hall_diff"],
    )
    test_samples = preprocessing.preprocess_data(test_records, **prep_kwargs)

    gd_kwargs = dict(
        label2idx=label2idx,
        mean=mean,
        std=std,
        target_len=target_len,
        n_hall=sens_cfg["n_hall"],
    )
    dl_kwargs = dict(
        batch_size=cfg.TCN_CFG["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    test_loader = DataLoader(
        GestureDataset(test_samples, train=False, augment=False, **gd_kwargs),
        shuffle=False,
        **dl_kwargs,
    )

    model = GestureTCN(tcn_cfg, num_classes=len(label2idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    all_results = {}  # {subject_id: {'preds': [], 'labels': []}}```

    with torch.no_grad():
        for x, lengths, y, subject_ids in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = y.numpy()

            for i, s_id in enumerate(subject_ids):
                if s_id not in all_results:
                    all_results[s_id] = {"preds": [], "labels": []}
                all_results[s_id]["preds"].append(preds[i])
                all_results[s_id]["labels"].append(targets[i])

    print("\n" + "=" * 30)
    print("[TEST] Detailed report on test subjects")
    print("=" * 30)

    for s_id, data in all_results.items():
        print(f"\n>>> Subject: {s_id}")
        y_true = data["labels"]
        y_pred = data["preds"]

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=[idx2label[i] for i in range(len(label2idx))],
                zero_division=0,
            )
        )

        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion matrix for {s_id}:")
        print(cm)
        print("\n[INFO] label mapping:")
        for k, v in label2idx.items():
            print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()
