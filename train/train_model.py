import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from model.load_data import load_records, split_records
from model.gesture_ds_class import GestureDataset
from autotrimmer.autotrimmer import AutoTrimmer
from model.tcn import GestureTCN
from train.liveplot import LivePlot
from train.run_epoch import run_epoch
import model.config as cfg
import model.preprocessing as preprocessing
import model.utils as utils


def main():
    utils.seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_records(cfg.DATA_ROOT)

    train_records, val_records = split_records(
        records,
        leave_people=cfg.TEST_ON,
        val_size=cfg.VAL_SIZE,
        seed=cfg.SEED,
    )
    print(f"[INFO] train={len(train_records)} val={len(val_records)}")

    labels = sorted({r["gesture_id"] for r in records})
    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    people_train = sorted({r["subject_id"] for r in train_records})
    person2id_train = {person: i for i, person in enumerate(people_train)}

    trimmer = AutoTrimmer(cfg.TRIMMER_CFG, cfg.SENS_CFG)

    prep_kwargs = dict(
        trimmer=trimmer,
        n_hall=cfg.SENS_CFG["n_hall"],
        add_hall_diff=cfg.PREPROCESSING_CFG["add_hall_diff"],
    )
    train_samples = preprocessing.preprocess_data(train_records, **prep_kwargs)
    val_samples = preprocessing.preprocess_data(val_records, **prep_kwargs)

    # fit normalizer on TRAIN only
    mean, std = preprocessing.fit_normalizer_from_samples(train_samples)
    target_len = cfg.PREPROCESSING_CFG["static_target_len"]
    use_dynamic = cfg.PREPROCESSING_CFG["use_dynamic_target_len"]
    if use_dynamic:
        target_len = preprocessing.estimate_target_len_from_samples(
            np.concatenate((train_samples, val_samples)),
            cfg.PREPROCESSING_CFG["dynamic_target_len_quantile"],
            cfg.PREPROCESSING_CFG["min_target_len"],
        )
    target_type = "dynamic" if use_dynamic else "static"
    print(f"[INFO] target_type: {target_type}, target_len={target_len}")

    sampler = WeightedRandomSampler(
        weights=utils.compute_person_weights(train_samples, person2id_train),
        num_samples=len(train_samples),
        replacement=True,
    )

    gd_kwargs = dict(
        label2idx=label2idx,
        mean=mean,
        std=std,
        target_len=target_len,
        augment_cfg=cfg.AUGMENT_CFG,
        n_hall=cfg.SENS_CFG["n_hall"],
    )

    dl_kwargs = dict(
        batch_size=cfg.TCN_CFG["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    train_loader = DataLoader(
        GestureDataset(
            train_samples,
            train=True,
            augment=cfg.AUGMENT_CFG["augment_train"],
            **gd_kwargs,
        ),
        sampler=sampler,
        **dl_kwargs,
    )

    val_loader = DataLoader(
        GestureDataset(val_samples, train=False, augment=False, **gd_kwargs),
        shuffle=False,
        **dl_kwargs,
    )

    model = GestureTCN(cfg.TCN_CFG, num_classes=len(labels)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] total number of params={total_params:,}")
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TCN_CFG["lr"],
        weight_decay=cfg.TCN_CFG["weight_decay"],
    )
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.2,
    #     patience=10,
    # )

    plotter = LivePlot()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    epoch_train = 0
    for epoch in range(1, cfg.TCN_CFG["max_epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        # scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if best_val_loss - val_loss > 1e-4:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "mean": mean,
                "std": std,
                "target_len": target_len,
                "subjects_train": people_train,
                "sens_cfg": cfg.SENS_CFG,
                "add_hall_diff": cfg.PREPROCESSING_CFG["add_hall_diff"],
                "trimmer_cfg": cfg.TRIMMER_CFG,
                "tcn_cfg": cfg.TCN_CFG,
            }
            patience_counter = 0
            print("  [BEST] saved")
        else:
            patience_counter += 1

        if patience_counter >= cfg.TCN_CFG["patience"]:
            epoch_train = epoch
            print(f"[EARLY STOP] no improvement for {cfg.TCN_CFG["patience"]} epochs")
            break

    # save best checkpoint
    path = f"artifacts_{cfg.TCN_CFG['num_blocks']}"
    path = path + "_aug" if cfg.AUGMENT_CFG["augment_train"] else path
    out_dir = cfg.ARTIFACTS_PATH / path
    out_dir.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        torch.save(best_state, out_dir / "best_tcn.pt")
        np.save(out_dir / "mean.npy", mean)
        np.save(out_dir / "std.npy", std)
        with (out_dir / "label2idx.json").open("w", encoding="utf-8") as f:
            json.dump(label2idx, f, ensure_ascii=False, indent=2)
        with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_len": target_len,
                    "subjects_train": people_train,
                    "sens_cfg": cfg.SENS_CFG,
                    "add_hall_diff": cfg.PREPROCESSING_CFG["add_hall_diff"],
                    "trimmer_cfg": cfg.TRIMMER_CFG,
                    "augment_cfg": cfg.AUGMENT_CFG,
                    "tcn_cfg": cfg.TCN_CFG,
                    "epochs": epoch_train,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
