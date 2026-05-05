# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import serial
import torch

import config as cfg
import preprocessing
from gesture_ds_class import GestureDataset
from tcn import GestureTCN
from trimmer.autotrimmer import AutoTrimmer
try:
    import pygame
except ImportError:
    pygame = None


def parse_line(line: str, expected_n: int = 13):
    parts = line.strip().split()
    if len(parts) != expected_n:
        return None
    try:
        return np.array([float(x) for x in parts], dtype=np.float32)
    except ValueError:
        return None


class LiveInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[INFO] Loading model and configurations...")
        
        checkpoint_path = Path(cfg.ARTIFACTS_PATH) / "best_tcn.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Haven't found checkpoint {checkpoint_path}!")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.label2idx = checkpoint["label2idx"]
        self.idx2label = checkpoint["idx2label"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.target_len = checkpoint["target_len"]
        self.tcn_cfg = checkpoint["tcn_cfg"]
        self.sens_cfg = checkpoint["sens_cfg"]
        self.add_hall_diff = checkpoint["add_hall_diff"]
        self.total_sens_num = sum(self.sens_cfg.values())
        self.sampling_rate = args.sampling_rate_hz

        self.model = GestureTCN(self.tcn_cfg, num_classes=len(self.label2idx)).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.trimmer = AutoTrimmer(checkpoint["trimmer_cfg"], self.sens_cfg)
        self.prep_kwargs = dict(
            trimmer=self.trimmer,
            n_hall=self.sens_cfg["n_hall"],
            add_hall_diff=self.add_hall_diff,
        )

        self.gd_kwargs = dict(
            label2idx=self.label2idx,
            mean=self.mean,
            std=self.std,
            target_len=self.target_len,
            n_hall=self.sens_cfg["n_hall"],
        )

        
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
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.audio_dir = Path(cfg.ARTIFACTS_PATH).parent / "audio"
        if not self.audio_dir.exists():
            self.audio_dir = Path(__file__).resolve().parent.parent / "audio"

        self.audio_enabled = False
        self.audio_lock = threading.Lock()

        if pygame is None:
            print("[WARN] pygame is not installed, audio is disabled.")
        else:
            try:
                pygame.mixer.init()
                self.audio_enabled = True
                print(f"[INFO] Audio initialized, folder: {self.audio_dir}")
            except Exception as e:
                print(f"[WARN] Audio init failed: {e}")
        print("[INFO] Initialization ended.")

    def _audio_path_for_label(self, label: str) -> Path:
        safe = label.strip().lower().replace(" ", "_").replace("?", "")
        return self.audio_dir / f"{safe}.mp3"

    def _play_label_audio(self, label: str):
        if not self.audio_enabled:
            return

        audio_path = self._audio_path_for_label(label)
        if not audio_path.exists():
            print(f"[WARN] Audio file not found: {audio_path}")
            return

        def _worker():
            try:
                with self.audio_lock:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(str(audio_path))
                    pygame.mixer.music.play()
            except Exception as e:
                print(f"[WARN] Could not play audio for '{label}': {e}")

        threading.Thread(target=_worker, daemon=True).start()
    
    def _start_recording(self):
        with self.lock:
            self.buffer = []
        self.recording.set()
        print("\n[REC] recording started. Press Enter again to stop.", flush=True)

    def _stop_recording_and_infer(self):
        self.recording.clear()
        with self.lock:
            data = np.array(self.buffer, dtype=np.float32)

        if data.shape[0] < 50:
            print(
                f"[WARN] too few frames ({data.shape[0]}), sample discarded.",
                flush=True,
            )
            return
        duration = data.shape[0] / self.sampling_rate
        print(
            f"[INFO] Processing sample ({data.shape[0]} frames, {duration:.3f}s)...",
            flush=True,
        )

        t_start = time.time()
        prep_sample = [
            {"data": data, "gesture_id": self.idx2label[0], "subject_id": "u"}
        ]
        prep_sample = preprocessing.preprocess_data(prep_sample, self.sens_cfg["n_hall"], self.add_hall_diff, self.trimmer)

        ds = GestureDataset(prep_sample, train=False, augment=False, **self.gd_kwargs)

        x, length, _, _ = ds[0]
        x = x.unsqueeze(0).to(self.device)  # shape: [1, C, T]
        length = length.unsqueeze(0).to(self.device)  # shape: []

  
        with torch.no_grad():
            logits = self.model(x, length)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            confidence = probs[0][pred_idx].item() * 100

        pred_label = self.idx2label[pred_idx]
        t_end = time.time()
        inf_time_ms = (t_end - t_start) * 1000
        print("=" * 40)
        print(
            f"Predicted gesture: {pred_label.upper()} (Confidence: {confidence:.1f}%, inference time: {inf_time_ms:.2f} ms)"
        )
        print("=" * 40)
        self._play_label_audio(pred_label)

    def _keyboard_loop(self):
        print("\nControls:")
        print("  [Enter]  -> start/stop recording")
        print("  [q] + [Enter] -> quit\n", flush=True)

        while not self.stop_event.is_set():
            try:
                cmd = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.stop_event.set()
                break

            if cmd == "q":
                self.stop_event.set()
                self.recording.clear()
                break

            if not self.recording.is_set():
                self._start_recording()
            else:
                self._stop_recording_and_infer()
                print("[READY] press Enter for next recording.", flush=True)

    def _serial_loop(self):
        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                sample = parse_line(line, expected_n=self.total_sens_num)
                if sample is None:
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
        self.keyboard_thread.start()
        try:
            self._serial_loop()
        finally:
            self.stop_event.set()
            try:
                self.ser.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Inference for Gestures")
    parser.add_argument(
        "--port", type=str, default="COM3", help="COM port (e.g. COM3 or /dev/ttyUSB0)"
    )
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument("--sampling-rate-hz", type=int, default=100)

    args = parser.parse_args()

    app = LiveInference(args)
    app.run()
