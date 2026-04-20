# annotate_gt.py
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import argparse


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


def get_quat_angles(quat):
    baseline = quat[:10].mean(axis=0)
    q0 = baseline / np.linalg.norm(baseline)
    dots = np.abs(np.dot(quat, q0))  # (T,)
    dots = np.clip(dots, 0.0, 1.0)
    angle = np.rad2deg(2 * np.arccos(dots))
    return angle


def plot_sample(data: np.ndarray, fs: int, title: str, gt_start=None, gt_end=None):
    hall = data[:, :3]
    acc = data[:, 3:6]
    gyro = data[:, 6:9]
    quat = data[:, 9:13]

    acc_mag = np.linalg.norm(acc, axis=1, ord=1)
    gyro_mag = np.linalg.norm(gyro, axis=1, ord=1)
    quat_angles = get_quat_angles(quat)

    t = np.arange(len(data))

    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Hall
    axes[0].plot(t, hall[:, 0], label="hall1")
    axes[0].plot(t, hall[:, 1], label="hall2")
    axes[0].plot(t, hall[:, 2], label="hall3")
    axes[0].set_ylabel("Hall")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # gyro activity
    axes[1].plot(t, gyro_mag, label="|gyro|")
    axes[1].set_ylabel("Gyro")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # Accel activity
    axes[2].plot(t, acc_mag, label="|acc|")
    axes[2].set_ylabel("Accel")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    # Quat angles
    axes[3].plot(t, quat_angles, label="Quat Angles")
    axes[3].set_ylabel("Quat Angles (deg)")
    axes[3].grid(True)
    axes[3].legend(loc="upper right")

    if gt_start is not None:
        x = gt_start
        for ax in axes:
            ax.axvline(x, color="green", linestyle="-", linewidth=2)

    if gt_end is not None:
        x = gt_end
        for ax in axes:
            ax.axvline(x, color="red", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    axes[1].set_xticks(np.arange(0, len(data), 20))
    plt.show(block=False)
    plt.pause(0.05)

    return fig


def parse_gt_input(user_text: str, default_start=None, default_end=None):
    text = user_text.strip().lower()

    if text in {"q", "quit", "exit"}:
        return "quit", None, None
    if text in {"s", "skip"}:
        return "skip", None, None
    if text in {"b", "back"}:
        return "back", None, None

    if text == "":
        if default_start is not None and default_end is not None:
            return "ok", int(default_start), int(default_end)
        return "invalid", None, None

    parts = text.split()
    if len(parts) != 2:
        return "invalid", None, None

    try:
        gt_start = int(parts[0])
        gt_end = int(parts[1])
        return "ok", gt_start, gt_end
    except ValueError:
        return "invalid", None, None


def annotate_dataset(root_dir: str):
    root = Path(root_dir)
    rng = np.random.default_rng(seed=43)
    raw_files = sorted(root.rglob("*_raw.csv"))
    rng.shuffle(raw_files)

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
        )

        while True:
            prompt = f"{sample_id} -> enter gt_start gt_end"
            if current_gt_start is not None and current_gt_end is not None:
                prompt += f" [current: {current_gt_start} {current_gt_end}]"
            prompt += " (s=skip, q=quit, b=back): "

            user_in = input(prompt)
            status, gt_start, gt_end = parse_gt_input(
                user_in, current_gt_start, current_gt_end
            )

            if status == "quit":
                plt.close(fig)
                print("[INFO] Quit.")
                return

            if status == "skip":
                print(f"[SKIP] {sample_id}")
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

            if not (0 <= gt_start < gt_end <= len(data)):
                print(
                    f"[WARN] Bad range: gt_start={gt_start}, gt_end={gt_end}, len={len(data)}"
                )
                continue

            gt_obj = {
                "valid": True,
                "gt_start": gt_start,
                "gt_end": gt_end,
            }

            with gt_path.open("w", encoding="utf-8") as f:
                json.dump(gt_obj, f, ensure_ascii=False, indent=2)

            print(f"[SAVE] {gt_path} <- ({gt_start}, {gt_end})")
            plt.close(fig)
            idx += 1
            break

    print("[INFO] Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manual gt_start / gt_end annotation tool"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"C:\Users\User\Desktop\diplom\dataset_collection\label_thanks\subject_Denis",
        help="dataset_collection root folder",
    )
    args = parser.parse_args()
    annotate_dataset(args.root)
