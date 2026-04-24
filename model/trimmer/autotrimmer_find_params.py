import optuna
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).parent.parent))

from trimmer.dataset_find_params_loader import load_trim_dataset
from trimmer.autotrimmer import AutoTrimmer
from sklearn.model_selection import StratifiedKFold
from trimmer.load_trimmer_cfg import load_trim_cfg
from config import SENS_CFG



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
   
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)

    gt_len = gt_end - gt_start
    if gt_len <= 0:
        return 0.0
    return intersection / gt_len


def evaluate_config(dataset, cfg, max_err_ms=100, mean_by="subject_id"):
    trimmer = AutoTrimmer(cfg, SENS_CFG)
    groups = {}
    for sample in dataset:
        key = sample[mean_by]
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
        "std_start_err_sign": np.std(group_mean_start_err),
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
    )
    return score


def objective(trial, dataset, n_splits=5, max_err_ms=100):
    """
    Целевая функция для Optuna: для предложенной конфигурации cfg
    вычисляет средний score по n_splits фолдам.
    """
    weight1 = trial.suggest_float("w1", 0.0, 1.0, step=0.01)
    weight2 = trial.suggest_float("w2", 0.0, 1.0, step=0.01)
    weight3 = trial.suggest_float("w3", 0.0, 1.0, step=0.01)
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
        "thr_start": trial.suggest_float("thr_start", 0.0, 1.0, step=0.01),
        "thr_end": trial.suggest_float("thr_end", 0.0, 1.0, step=0.01),
        "min_start_run": trial.suggest_int("min_start_run", 1, 25),
        "min_end_run": trial.suggest_int("min_end_run", 1, 25),
        "margin_start": trial.suggest_int("margin_start", 1, 15),
        "margin_end": trial.suggest_int("margin_end", 1, 15),
        "acc_weight": acc_weight,
        "gyro_weight": gyro_weight,
        "quat_weight": quat_weight,
    }
    strat_labels = [f"{s['subject_id']}__{s['gesture_label']}" for s in dataset]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_scores = []
    for train_idx, val_idx in skf.split(dataset, strat_labels):
        val_fold = [dataset[i] for i in val_idx]
        metrics = evaluate_config(val_fold, cfg, max_err_ms=max_err_ms)
        fold_scores.append(getScore(metrics))

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    trial.set_user_attr("folds_std", std_score)
    return mean_score


def optimize_hyperparameters(dataset, n_trials=500, seed=42, n_splits=5):
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda trial: objective(trial, dataset, n_splits=n_splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def get_all_cfgs_and_scores(study, dataset, ret_values=100):
    """Получаем все конфигурации из trials, отсортированные по убыванию score (value)"""
    results = []
    for trial in study.trials:
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
        metrics_full = evaluate_config(dataset, cfg)
        score_cv_mean = trial.value
        score_cv_std = trial.user_attrs.get("folds_std", 0.0)
        results.append(
            {
                "cfg": cfg,
                "metrics_full": metrics_full,
                "score_cv_mean": score_cv_mean,
                "score_cv_std": score_cv_std,
            }
        )
    results.sort(key=lambda x: x["score_cv_mean"], reverse=True)
    return results[:ret_values]


import json

if __name__ == "__main__":
    dataset = load_trim_dataset(r"C:\Users\User\Desktop\diplom\dataset_collection")
    cfg = load_trim_cfg(r"C:\Users\User\Desktop\diplom\model\trimmer\trimmer_cfg.json")
    # metrics = evaluate_config(dataset, cfg)
    # print(metrics, f"score: {getScore(metrics)}")
    print("Starting Bayesian optimization with Stratified 5-Fold CV...")
    study = optimize_hyperparameters(dataset, n_trials=2000, seed=42, n_splits=5)

    cfgs = get_all_cfgs_and_scores(study, dataset, ret_values=100)

    csv_path = Path(
        r"C:\Users\User\Desktop\diplom\auto_trimmer_dataset_and_code\optuna_results_cv.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        first = cfgs[0]
        cfg_keys = list(first["cfg"].keys())
        metric_keys = list(first["metrics_full"].keys())
        header = (
            cfg_keys
            + [f"full_{k}" for k in metric_keys]
            + ["cv_score_mean", "cv_score_std"]
        )
        writer.writerow(header)
        for item in cfgs:
            row = (
                list(item["cfg"].values())
                + list(item["metrics_full"].values())
                + [item["score_cv_mean"], item["score_cv_std"]]
            )
            writer.writerow(row)

    
    t = np.arange(1, len(cfgs) + 1)
    plt.figure(figsize=(10, 6))
    means = [cfg["score_cv_mean"] for cfg in cfgs]
    stds = [cfg["score_cv_std"] for cfg in cfgs]
    plt.plot(t, means, label="CV mean score", color="blue")
    plt.fill_between(
        t,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.2,
        color="blue",
        label="±1 std",
    )
    plt.xlabel("Configuration (sorted by CV mean score)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.title("Optimization Results (Stratified K-Fold CV)")
    plt.legend()
    plt.show()
