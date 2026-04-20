import optuna
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset_loader import load_trim_dataset
from autotrimmer import AutoTrimmer
from sklearn.model_selection import train_test_split

# # =========================
# # METRICS
# # =========================


# pred_start, gt_end - EXCLUSIVE
def compute_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)

    intersection = max(0, inter_end - inter_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)

    if union == 0:
        return 0.0
    return intersection / union


def compute_recall(pred_start, pred_end, gt_start, gt_end):
    # СЃРєРѕР»СЊРєРѕ С‡Р°СЃС‚Рё gt РїРѕРєСЂС‹С‚Рѕ
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)

    gt_len = gt_end - gt_start
    if gt_len <= 0:
        return 0.0
    return intersection / gt_len


def evaluate_config(dataset, cfg, max_err_ms=100, mean_by="subject_id"):
    trimmer = AutoTrimmer(cfg)
    groups = {}
    for sample in dataset:
        key = sample[mean_by]  # например, 'subject_01' или 'hello'
        if key not in groups:
            groups[key] = {"ious": [], "recalls": [], "start_errs": [], "end_errs": []}

        gt_start = sample["gt_start"]
        gt_end = sample["gt_end"]
        fs = sample["sampling_rate_hz"]

        pred_start, pred_end = trimmer.trim(sample["data"])
        iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
        recall = compute_recall(pred_start, pred_end, gt_start, gt_end)
        start_err = (abs(pred_start - gt_start) / fs) * 1000.0
        end_err = (abs(pred_end - gt_end) / fs) * 1000.0
        norm_start_err = min(start_err / max_err_ms, 1.0)
        norm_end_err = min(end_err / max_err_ms, 1.0)

        groups[key]["ious"].append(iou)
        groups[key]["recalls"].append(recall)
        groups[key]["start_errs"].append(norm_start_err)
        groups[key]["end_errs"].append(norm_end_err)

    # Для каждой группы вычисляем средние метрики
    group_mean_iou = []
    group_mean_recall = []
    group_mean_start_err = []
    group_mean_end_err = []
    for key, data in groups.items():
        group_mean_iou.append(np.mean(data["ious"]))
        group_mean_recall.append(np.mean(data["recalls"]))
        group_mean_start_err.append(np.mean(data["start_errs"]))
        group_mean_end_err.append(np.mean(data["end_errs"]))

    return {
        "mean_iou": np.mean(group_mean_iou),
        "mean_recall": np.mean(group_mean_recall),
        "norm_start_err": np.mean(group_mean_start_err),
        "norm_end_err": np.mean(group_mean_end_err),
        "std_start_err_sign": np.std(group_mean_start_err),  # разброс между группами
        "std_end_err_sign": np.std(group_mean_end_err),
    }


def getScore(metrics):
    recall = metrics["mean_recall"]
    if recall < 0.9:
        recall = 0.0
    score = (
        1.0 * metrics["mean_iou"]
        + 0.6 * recall
        - 0.3 * metrics["norm_start_err"]
        - 0.3 * metrics["norm_end_err"]
        # - 0.3 * np.abs(metrics["norm_start_err"])
        # - 0.5 * np.abs(metrics["norm_end_err"])
    )
    return score


def objective(trial, dataset, max_err_ms=100):
    """
    Целевая функция для Optuna: для предложенной конфигурации cfg
    вычисляет средний score по n_splits фолдам.
    """
    weight1 = trial.suggest_float("w1", 0.0, 1.0, step=0.02)
    weight2 = trial.suggest_float("w2", 0.0, 1.0, step=0.02)
    weight3 = trial.suggest_float("w3", 0.0, 1.0, step=0.02)
    total = weight1 + weight2 + weight3
    acc_weight, gyro_weight, quat_weight = (
        weight1 / total,
        weight2 / total,
        weight3 / total,
    )
    cfg = {
        "baseline_len_start": trial.suggest_int("baseline_len_start", 5, 20),
        "baseline_len_end": trial.suggest_int("baseline_len_end", 5, 25),
        "smooth_win_gyro": trial.suggest_int("smooth_win_gyro", 2, 13),
        "smooth_win_acc": trial.suggest_int("smooth_win_acc", 2, 13),
        "smooth_win_quat": trial.suggest_int("smooth_win_quat", 2, 10),
        "thr_start": trial.suggest_float("thr_start", 0.0, 1.0, step=0.02),
        "thr_end": trial.suggest_float("thr_end", 0.0, 1.0, step=0.02),
        "min_start_run": trial.suggest_int("min_start_run", 1, 25),
        "min_end_run": trial.suggest_int("min_end_run", 1, 25),
        "margin_start": trial.suggest_int("margin_start", 1, 15),
        "margin_end": trial.suggest_int("margin_end", 1, 15),
        "acc_weight": acc_weight,
        "gyro_weight": gyro_weight,
        "quat_weight": quat_weight,
    }
    metric = evaluate_config(dataset, cfg, max_err_ms=max_err_ms)
    return getScore(metric)


def optimize_hyperparameters(dataset, n_trials=500, seed=42):
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda trial: objective(trial, dataset),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def get_all_cfgs_and_scores(study, test_set):
    """Получаем все конфигурации из trials, отсортированные по убыванию score (value)"""
    cfgs = []
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    for trial in sorted_trials:
        weight1 = trial.params["w1"]
        weight2 = trial.params["w2"]
        weight3 = trial.params["w3"]
        total = weight1 + weight2 + weight3
        acc_weight, gyro_weight, quat_weight = (
            weight1 / total,
            weight2 / total,
            weight3 / total,
        )
        cfg = {
            "baseline_len_start": trial.params["baseline_len_start"],
            "baseline_len_end": trial.params["baseline_len_end"],
            "smooth_win_gyro": trial.params["smooth_win_gyro"],
            "smooth_win_acc": trial.params["smooth_win_acc"],
            "smooth_win_quat": trial.params["smooth_win_quat"],
            "thr_start": trial.params["thr_start"],
            "thr_end": trial.params["thr_end"],
            "min_start_run": trial.params["min_start_run"],
            "min_end_run": trial.params["min_end_run"],
            "margin_start": trial.params["margin_start"],
            "margin_end": trial.params["margin_end"],
            "acc_weight": acc_weight,
            "gyro_weight": gyro_weight,
            "quat_weight": quat_weight,
        }
        cfgs.append(
            {
                "cfg": cfg,
                "score_train": trial.value,
                "score_test": getScore(evaluate_config(test_set, cfg)),
            }
        )
    return cfgs


if __name__ == "__main__":
    dataset = load_trim_dataset(r"C:\Users\User\Desktop\diplom\dataset_collection")
    print("Starting Bayesian optimization...")
    strat = [f'{s["subject_id"]}__{s["gesture_label"]}' for s in dataset]
    train_set, test_set = train_test_split(
        dataset, test_size=0.2, random_state=48, shuffle=True, stratify=strat
    )
    study = optimize_hyperparameters(train_set, n_trials=500, seed=48)

    cfgs = get_all_cfgs_and_scores(study, test_set)
    csv_path = Path(
        r"C:\Users\User\Desktop\diplom\auto_trimmer_dataset_and_code\optuna_results_2.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = list(cfgs[0]["cfg"].keys()) + ["score_train", "score_test"]
        writer.writerow(row)
        for cfg in cfgs:
            writer.writerow(
                list(cfg["cfg"].values()) + [cfg["score_train"], cfg["score_test"]]
            )

    t = np.arange(1, len(cfgs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(t, [cfg["score_train"] for cfg in cfgs], label="train", color="blue")
    plt.plot(
        t,
        [cfg["score_test"] for cfg in cfgs],
        label="test",
        color="orange",
    )
    plt.legend()
    plt.xlabel("Configuration")
    plt.ylabel("Score")
    plt.grid(True)
    plt.title("Optimization Results")
    plt.show()
