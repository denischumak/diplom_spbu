import argparse
import threading
import time
from pathlib import Path
import tkinter as tk  # Добавлен импорт tkinter

import numpy as np
import serial
import torch

import model.config as cfg
import model.preprocessing as preprocessing
from model.gesture_ds_class import GestureDataset
from model.tcn import GestureTCN
from autotrimmer.autotrimmer import AutoTrimmer

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
        
        checkpoint_path = cfg.ARTIFACTS_PATH / "artifacts_3_aug" / "best_tcn.pt"
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
        self.pred_label_rus = { 
                          "hello": "Привет", 
                          "bye": "Пока", 
                          "how_are_you": "Как дела?", 
                          "thanks": "Спасибо" 
                          }

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
        self.audio_dir = Path(__file__).resolve().parent / "audio"

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
                
        # --- Инициализация графического интерфейса ---
        self.root = tk.Tk()
        self.root.title("Распознавание жестов")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="white")

        # Центральный текст (жест)
        self.center_label = tk.Label(
            self.root, 
            text="Ожидание жеста", 
            font=("Helvetica", 100, "bold"), 
            bg="white", 
            fg="black"
        )
        self.center_label.place(relx=0.5, rely=0.5, anchor="center")

        # Текст в правом нижнем углу (статистика)
        self.info_label = tk.Label(
            self.root, 
            text="", 
            font=("Helvetica", 36), 
            bg="white", 
            fg="black",
            justify="right"
        )
        self.info_label.place(relx=0.98, rely=0.98, anchor="se")

        # Привязка клавиш
        self.root.bind("<Return>", self._toggle_recording)
        self.root.bind("<Escape>", self._quit_app)  # Выход по Escape
        
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
    
    def _toggle_recording(self, event=None):
        """Переключение записи по нажатию Enter в GUI"""
        if not self.recording.is_set():
            self._start_recording()
            self.center_label.config(text="Запись жеста")
            self.info_label.config(text="")
        else:
            self.center_label.config(text="Обработка...")
            self.root.update()  # Принудительно обновляем интерфейс перед инференсом
            self._stop_recording_and_infer()

    def _start_recording(self):
        with self.lock:
            self.buffer = []
        self.recording.set()
        print("\n[REC] recording started.", flush=True)

    def _stop_recording_and_infer(self):
        t_start = time.perf_counter()
        self.recording.clear()
        with self.lock:
            data = np.array(self.buffer, dtype=np.float32)

        if data.shape[0] < 50:
            print(f"[WARN] too few frames ({data.shape[0]}), sample discarded.", flush=True)
            self.center_label.config(text="Ожидание жеста")
            return
            
        duration = data.shape[0] / self.sampling_rate
        print(f"[INFO] Processing sample ({data.shape[0]} frames, {duration:.3f}s)...", flush=True)

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
        t_end = time.perf_counter()
        inf_time_ms = (t_end - t_start) * 1000
        
        print("=" * 40)
        print(f"Predicted gesture: {pred_label.upper()} (Confidence: {confidence:.1f}%, inference time: {inf_time_ms:.2f} ms)")
        print("=" * 40)
        
        # Обновление графического интерфейса
        pred_label_rus = self.pred_label_rus.get(pred_label, pred_label)
        self.center_label.config(text=pred_label_rus.upper())
        self.info_label.config(text=f"Уверенность: {confidence:.1f}%\nВремя: {inf_time_ms:.2f} мс")
        
        self._play_label_audio(pred_label)

    def _quit_app(self, event=None):
        """Закрытие приложения по Escape"""
        self.stop_event.set()
        self.recording.clear()
        self.root.destroy()

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
        # Запускаем чтение порта в фоновом потоке
        serial_thread = threading.Thread(target=self._serial_loop, daemon=True)
        serial_thread.start()
        
        try:
            # Запускаем основной цикл графического интерфейса в главном потоке
            self.root.mainloop()
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
