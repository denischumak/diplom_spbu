#!/usr/bin/env python3
# collect_dataset.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import model.config as cfg

AXES = ["x", "y", "z"]
FEATURE_NAMES = [
    *[f"hall_{i + 1}" for i in range(cfg.SENS_CFG["n_hall"])],
    *[f"acc_{AXES[i]}" for i in range(cfg.SENS_CFG["n_acc"])],
    *[f"gyro_{AXES[i]}" for i in range(cfg.SENS_CFG["n_gyro"])],
    *[f"quat_{i + 1}" for i in range(cfg.SENS_CFG["n_quat"])],
]


def parse_fingers(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_line(line: str, expected_n: int = len(FEATURE_NAMES)):
    parts = line.strip().split()
    if len(parts) != expected_n:
        return None
    try:
        return np.array([float(x) for x in parts], dtype=np.float32)
    except ValueError:
        return None


class DatasetCollector:
    def __init__(self, args):
        self.args = args
        self.dataset_root = Path(args.dataset_root)

        self.session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = (
            self.dataset_root
            / f"label_{args.gesture_label}"
            / f"subject_{args.subject_id}"
            / f"session_{self.session_id}"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
        self.ser.set_buffer_size(rx_size=16384, tx_size=16384)
        self.lock = threading.Lock()
        
        self.recording = threading.Event()
        self.stop_event = threading.Event()

        self.buffer = []
        self.sample_idx = 0
        self.current_start_ts = None

        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)

    def _start_recording(self):
        with self.lock:
            self.buffer = []
            self.current_start_ts = time.time()
        self.recording.set()
        print("\n[REC] recording started. Press Enter again to stop.", flush=True)

    def _stop_recording_and_save(self):
        self.recording.clear()
        ts = 0.0
        with self.lock:
            data = np.array(self.buffer, dtype=np.float32)
            ts = time.time()

        if data.shape[0] < self.args.min_frames:
            print(
                f"[WARN] too few frames ({data.shape[0]}), sample discarded.",
                flush=True,
            )
            return

        self.sample_idx += 1
        sample_id = f"sample_{self.sample_idx:05d}"
        sample_dir = self.session_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        raw_csv = sample_dir / f"{sample_id}_raw.csv"
        meta_json = sample_dir / f"{sample_id}_meta.json"

        duration_sec = ts - self.current_start_ts

        # baseline_len = min(self.args.baseline_len, data.shape[0])
        # baseline_vector = data[:baseline_len].mean(axis=0).astype(np.float32)

        # Save raw CSV
        with raw_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES)
            for row in data:
                writer.writerow([f"{row[0]:.5f}"] + [f"{v:.5f}" for v in row[1:]])

        meta = {
            "sample_id": sample_id,
            "gesture_label": self.args.gesture_label,
            "subject_id": self.args.subject_id,
            "session_id": self.session_id,
            "hand_size": self.args.hand_size,
            "handedness": self.args.handedness,
            "working_fingers": self.args.working_fingers,
            "session_condition": self.args.session_condition,
            "sampling_rate_hz": self.args.sampling_rate_hz,
            "n_frames": int(data.shape[0]),
            "duration_sec": duration_sec,
            "raw_csv": str(raw_csv),
            # "baseline_len_used": int(baseline_len),
            # "baseline_vector": baseline_vector.tolist(),
            "notes": self.args.notes,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        with meta_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Helpful reminder for later preprocessing
        print(
            f"[SAVE] {sample_id} saved: {data.shape[0]} frames, {duration_sec}s",
            flush=True,
        )

    def _keyboard_loop(self):
        print("\nControls:")
        print("  Enter  -> start/stop recording")
        print("  q + Enter -> quit\n", flush=True)

        while not self.stop_event.is_set():
            try:
                cmd = input().strip().lower()
            except EOFError:
                self.stop_event.set()
                break
            except KeyboardInterrupt:
                self.stop_event.set()
                break

            if cmd == "q":
                self.stop_event.set()
                self.recording.clear()
                break

            if not self.recording.is_set():
                self._start_recording()
            else:
                self._stop_recording_and_save()
                print("[READY] press Enter for next recording.", flush=True)

    def _serial_loop(self):
        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                sample = parse_line(line, expected_n=len(FEATURE_NAMES))
                if sample is None:
                    print(line)
                    continue
                if self.recording.is_set():
                    with self.lock:
                        self.buffer.append(sample)

            except serial.SerialException as e:
                print(f"[SERIAL ERROR] {e}", flush=True)
                self.stop_event.set()
                break
            except Exception as e:
                print(f"[ERROR] {e}", flush=True)

    def run(self):
        print(f"[INFO] session dir: {self.session_dir}", flush=True)
        print("[INFO] waiting for Enter...", flush=True)

        self.keyboard_thread.start()

        try:
            self._serial_loop()
        finally:
            self.stop_event.set()
            try:
                self.ser.close()
            except Exception:
                pass


def build_argparser():
    p = argparse.ArgumentParser(
        description="Gesture dataset collector via Enter start/stop."
    )
    p.add_argument("--port", help="COM port, e.g. COM3 or /dev/ttyUSB0", default="COM3")
    p.add_argument("--baudrate", type=int, default=921600)
    p.add_argument("--timeout", type=float, default=0.1)

    p.add_argument(
        "--dataset-root",
        type=str,
        default="C:\\Users\\User\\Desktop\\diplom\\dataset_collection",
    )
    p.add_argument("--session-id", type=str, default="")
    p.add_argument(
        "--session-condition",
        type=str,
        default="standing",
        choices=["sitting", "standing"],
    )

    p.add_argument(
        "--gesture-label",
        type=str,
        required=True,
        choices=["hello", "bye", "thanks", "how_are_you"],
    )
    p.add_argument(
        "--subject-id",
        type=str,
        help="Subject id: S01, S02, [Name], ...",
        required=True,
    )
    p.add_argument(
        "--hand-size", type=str, required=True, choices=["small", "medium", "large"]
    )
    p.add_argument(
        "--handedness", type=str, default="right", choices=["left", "right", "unknown"]
    )
    p.add_argument("--working-fingers", type=parse_fingers, default="index,ring,pinky")

    p.add_argument("--sampling-rate-hz", type=int, default=100)
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--min-frames", type=str, default=50)
    return p


def main():
    args = build_argparser().parse_args()
    collector = DatasetCollector(args)
    collector.run()


if __name__ == "__main__":
    main()
