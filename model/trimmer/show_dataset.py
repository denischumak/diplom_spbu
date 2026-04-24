import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from autotrimmer import AutoTrimmer
import config as cfg


def load_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_raw_csv(raw_csv: Path) -> np.ndarray:
    data = np.genfromtxt(raw_csv, delimiter=",", skip_header=1, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"Empty CSV: {raw_csv}")
    if data.ndim == 1:
        data = data[None, :]

    return data


def plot_sample(
    data: np.ndarray, fs: int, title: str, gt_start=None, gt_end=None, trimmer_cfg=None
):
    sens_cfg = cfg.SENS_CFG
    trimmer = AutoTrimmer(trimmer_cfg, sens_cfg)
    pred_start, pred_end = trimmer.trim(data)
    hall = data[:, : sens_cfg["n_hall"]]
    n_hall_acc = sens_cfg["n_hall"] + sens_cfg["n_acc"]
    acc = data[:, sens_cfg["n_hall"] : n_hall_acc]
    gyro = data[:, n_hall_acc : n_hall_acc + sens_cfg["n_gyro"]]
    quat = data[:, n_hall_acc + sens_cfg["n_gyro"] : ]

    acc_mag = np.linalg.norm(acc, axis=1, ord=1)
    gyro_mag = np.linalg.norm(gyro, axis=1, ord=1)

    t = np.arange(len(data))

    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Hall
    for i in range(sens_cfg["n_hall"]):
        axes[0].plot(t, hall[:, i], label=f"hall_{i}")
    axes[0].set_ylabel("Hall")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # IMU activity
    axes[1].plot(t, gyro_mag, label="|gyro|")
    axes[1].set_ylabel("Gyro")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # Accel activity
    axes[2].plot(t, acc_mag, label="|acc|")
    axes[2].set_ylabel("Accel")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    if gt_start is not None:
        x = gt_start
        for ax in axes:
            ax.axvline(x, color="green", linestyle="-", linewidth=2)

    if gt_end is not None:
        x = gt_end
        for ax in axes:
            ax.axvline(x, color="red", linestyle="-", linewidth=2)
    if trimmer_cfg is not None:
        # axes[3].plot(t, score, label="score")
        # axes[3].set_ylabel("Score")
        # axes[3].grid(True)
        # axes[3].legend(loc="upper right")
        quat_angles = trimmer.get_quat_angles(quat)
        axes[3].plot(t, quat_angles, label="quat_angles")
        axes[3].set_ylabel("Quat Angles")
        axes[3].grid(True)
        axes[3].legend(loc="upper right")
        for ax in axes:
            ax.axvline(pred_start, color="blue", linestyle="--", linewidth=2)
            ax.axvline(pred_end, color="black", linestyle="--", linewidth=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    axes[1].set_xticks(np.arange(0, len(data), 20))
    plt.show(block=False)
    plt.pause(0.05)

    return fig


def parse_gt_input(user_text: str):
    text = user_text.strip().lower()

    if text in {"q", "quit", "exit"}:
        return "quit"
    elif text in {"n", "next"}:
        return "next"
    elif text in {"b", "back"}:
        return "back"
    else:
        return "invalid"


def show_dataset(root_dir: str, trimmer_cfg: dict = None):
    root = Path(root_dir)
    raw_files = sorted(root.rglob("*_raw.csv"))

    if not raw_files:
        print(f"[INFO] No *_raw.csv found in {root}")
        return

    print(f"[INFO] Found {len(raw_files)} samples")
    print(
        "[INFO] Controls in console: `gt_start gt_end (exclusive)`, `s` to skip, `q` to quit.\n"
    )

    plt.ion()

    idx = 0
    while idx < len(raw_files):
        raw_csv = raw_files[idx]
        sample_dir = raw_csv.parent
        sample_stem = raw_csv.stem.replace("_raw", "")
        meta_path = sample_dir / f"{sample_stem}_meta.json"
        gt_path = sample_dir / f"{sample_stem}_gt.json"

        try:
            data = load_raw_csv(raw_csv)
        except Exception as e:
            print(f"[WARN] {raw_csv}: {e}")
            idx += 1
            continue

        meta = load_meta(meta_path)
        fs = meta.get("sampling_rate_hz")
        label = meta.get("gesture_label")
        subject_id = meta.get("subject_id")
        session_id = meta.get("session_id")
        sample_id = meta.get("sample_id")

        current_gt_start = None
        current_gt_end = None
        if gt_path.exists():
            try:
                with gt_path.open("r", encoding="utf-8") as f:
                    gt = json.load(f)
                current_gt_start = gt.get("gt_start", None)
                current_gt_end = gt.get("gt_end", None)
            except Exception:
                pass

        title = f"[{idx + 1}/{len(raw_files)}] {sample_id} | label={label} | subject={subject_id} | session={session_id}"
        fig = plot_sample(
            data=data,
            fs=fs,
            title=title,
            gt_start=current_gt_start,
            gt_end=current_gt_end,
            trimmer_cfg=trimmer_cfg,
        )

        while True:
            prompt = f"{sample_id}"
            if current_gt_start is not None and current_gt_end is not None:
                prompt += f" [current: {current_gt_start} {current_gt_end}]"
            prompt += " (n=next, q=quit, b=back): "

            user_in = input(prompt)
            status = parse_gt_input(user_in)

            if status == "quit":
                plt.close(fig)
                print("[INFO] Quit.")
                return

            if status == "next":
                print(f"[NEXT] {sample_id}")
                plt.close(fig)
                idx += 1
                break

            if status == "back":
                print(f"[BACK] {sample_id}")
                plt.close(fig)
                idx = max(0, idx - 1)
                break

            if status == "invalid":
                print("[WARN] Invalid input. Example: 31 87")
                continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manual gt_start / gt_end annotation tool"
    )
    parser.add_argument(
        "--root",
        type=str,
        # default=r"C:\Users\User\Desktop\diplom\auto_trimmer_dataset_and_code\test",
        default=r"C:\Users\User\Desktop\diplom\dataset_collection\train",
        help="dataset_collection root folder",
    )
    args = parser.parse_args()

    cfg_path = Path(r"C:\Users\User\Desktop\diplom\model\trimmer\trimmer_cfg.json")
    with cfg_path.open("r", encoding="utf8") as f:
        trimmer_cfg = json.load(f)
    show_dataset(args.root, trimmer_cfg)
