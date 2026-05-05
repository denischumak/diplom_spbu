# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import threading
import time
from collections import deque

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
        self.target_len = int(checkpoint["target_len"])
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
        self.gd_kwargs = dict(
            label2idx=self.label2idx,
            mean=self.mean,
            std=self.std,
            target_len=self.target_len,
            n_hall=self.sens_cfg["n_hall"],
        )

        # Streaming state
        self.stream = deque(maxlen=self.target_len)
        self.stream_lock = threading.Lock()
        self.window_event = threading.Event()
        self.stop_event = threading.Event()

        self.sample_counter = 0
        self.latest_window = None
        self.latest_version = 0
        self.last_processed_version = -1

        # Every 100 ms by default at 100 Hz -> 10 samples
        self.hop_size = 10

        # FSM / anti-spam parameters
        self.min_confidence = cfg.ONL_INFERENCE_CFG["min_confidence"]
        self.stable_required = cfg.ONL_INFERENCE_CFG["stable_required"]
        self.rest_confidence = cfg.ONL_INFERENCE_CFG["rest_confidence"]
        self.rest_required = cfg.ONL_INFERENCE_CFG["rest_required"]
        self.min_emit_gap_sec = cfg.ONL_INFERENCE_CFG["min_emit_gap_sec"]

        self.rest_label = (
            "rest"
            if "rest" in self.label2idx
            else ("no_gesture" if "no_gesture" in self.label2idx else None)
        )

        self.active = threading.Event()  # armed / ready to run
        self.stream.clear()
        self.state = "PAUSED"
        self.candidate_label = None
        self.candidate_count = 0
        self.rest_count = 0
        self.last_emitted_label = None
        self.last_emit_time = 0.0

        # COM
        self.ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
        self.ser.set_buffer_size(rx_size=16384, tx_size=16384)

        # Audio
        self.audio_dir = Path(cfg.ARTIFACTS_PATH) / "audio"
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

        self.reader_thread = threading.Thread(target=self._serial_loop, daemon=True)
        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)

        print("[INFO] Initialization ended.")

    def _audio_path_for_label(self, label: str) -> Path:
        safe = label.strip().lower().replace(" ", "_").replace("?", "")
        safe = safe.replace("/", "_").replace("\\", "_")
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

    def _infer_window(self, data: np.ndarray):
        """
        data: raw sliding window [T, 13]
        returns: (pred_label, confidence) where confidence is [0..1]
        """
        prep_sample = [
            {"data": data, "gesture_id": self.idx2label[0], "subject_id": "u"}
        ]
        prep_sample = preprocessing.preprocess_data(
            prep_sample, self.sens_cfg["n_hall"], self.add_hall_diff
        )

        ds = GestureDataset(prep_sample, train=False, augment=False, **self.gd_kwargs)
        x, length, _, _ = ds[0]

        x = x.unsqueeze(0).to(self.device)  # [1, C, T]
        length = length.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x, length)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        pred_label = self.idx2label[pred_idx]
        return pred_label, confidence

    def _fsm_update(self, pred_label: str, confidence: float):
        now = time.time()

        # LOCKED: wait until rest/no_gesture re-arms the system
        if self.state == "LOCKED":
            if self.rest_label is not None:
                if pred_label == self.rest_label and confidence >= self.rest_confidence:
                    self.rest_count += 1
                    if self.rest_count >= self.rest_required:
                        self.state = "SEARCH"
                        self.rest_count = 0
                        self.candidate_label = None
                        self.candidate_count = 0
                        self.last_emitted_label = None
                        print("[FSM] re-armed by REST", flush=True)
                else:
                    self.rest_count = 0
            else:
                # Fallback if there is no rest/no_gesture class:
                # time-based cooldown only
                if now - self.last_emit_time >= self.min_emit_gap_sec:
                    self.state = "SEARCH"
                    self.candidate_label = None
                    self.candidate_count = 0
            return None

        # SEARCH: look for a stable non-rest class
        if confidence < self.min_confidence:
            self.candidate_label = None
            self.candidate_count = 0
            return None

        if self.rest_label is not None and pred_label == self.rest_label:
            self.candidate_label = None
            self.candidate_count = 0
            return None

        if pred_label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = pred_label
            self.candidate_count = 1

        if self.candidate_count >= self.stable_required:
            # suppress repeated emits of the same gesture until rest re-arms
            if pred_label != self.last_emitted_label:
                if now - self.last_emit_time >= self.min_emit_gap_sec:
                    self.last_emitted_label = pred_label
                    self.last_emit_time = now
                    self.state = "LOCKED"
                    self.candidate_label = None
                    self.candidate_count = 0
                    self.rest_count = 0

                    print("=" * 40)
                    print(
                        f"Predicted gesture: {pred_label.upper()} "
                        f"(Confidence: {confidence * 100:.1f}%)"
                    )
                    print("=" * 40)

                    self._play_label_audio(pred_label)
                    return pred_label

        return None

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
                if not self.active.is_set():
                    continue  # read-and-drop while paused

                with self.stream_lock:
                    self.stream.append(sample)
                    self.sample_counter += 1

                    if len(self.stream) < self.target_len:
                        continue

                    if self.sample_counter % self.hop_size != 0:
                        continue

                    # snapshot latest sliding window
                    self.latest_window = np.array(
                        self.stream, dtype=np.float32, copy=True
                    )
                    self.latest_version += 1
                    self.window_event.set()

            except serial.SerialException as e:
                print(f"[SERIAL ERROR] {e}", flush=True)
                self.stop_event.set()
                break
            except Exception as e:
                print(f"[ERROR] {e}", flush=True)

    def _inference_loop(self):
        while not self.stop_event.is_set():
            if not self.window_event.wait(timeout=0.2):
                continue

            if self.stop_event.is_set():
                break

            with self.stream_lock:
                if self.latest_window is None:
                    self.window_event.clear()
                    continue
                window = self.latest_window.copy()
                version = self.latest_version
                self.window_event.clear()

            if version == self.last_processed_version:
                continue

            self.last_processed_version = version
            if not self.active.is_set():
                continue
            pred_label, confidence = self._infer_window(window)
            if pred_label is None:
                continue

            self._fsm_update(pred_label, confidence)

    def _toggle_active(self):
        if self.active.is_set():
            self.active.clear()
            self.state = "PAUSED"
            print(
                "\n[PAUSE] stream & inference paused. Press Enter to resume.",
                flush=True,
            )
        else:
            with self.stream_lock:
                self.stream.clear()
                self.sample_counter = 0
                self.latest_window = None
                self.latest_version = 0
                self.last_processed_version = -1

            self.candidate_label = None
            self.candidate_count = 0
            self.rest_count = 0
            self.last_emitted_label = None

            self.active.set()
            self.state = "SEARCH"
            print(
                "\n[READY] stream & inference started. Press Enter again to pause.",
                flush=True,
            )

    def _keyboard_loop(self):
        print("\nControls:")
        print("  [Enter]  -> start/stop stream & inference")
        print("  [q] + [Enter] -> quit\n", flush=True)

        while not self.stop_event.is_set():
            try:
                cmd = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.stop_event.set()
                break

            if cmd == "q":
                self.stop_event.set()
                break

            # Enter pressed
            self._toggle_active()

    def run(self):
        self.reader_thread.start()
        self.infer_thread.start()
        self.keyboard_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        finally:
            self.stop_event.set()
            try:
                self.ser.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Online Inference for Gestures")
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="COM port (e.g. COM3 or /dev/ttyUSB0)",
    )
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument("--sampling-rate-hz", type=int, default=100)

    args = parser.parse_args()

    app = LiveInference(args)
    app.run()
