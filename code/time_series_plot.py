import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import math


def read_adc_file(path):
    """Читает файл, возвращает numpy array с числовыми значениями (float)."""
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                # пропускаем строки, которые не парсятся (можно логировать при необходимости)
                continue
    if len(vals) == 0:
        raise ValueError(f"В файле {path} нет числовых данных.")
    return np.array(vals, dtype=float)


def plot_adc_timeseries(
    adc_values,
    coef,
    out_prefix="adc_timeseries",
    figsize=(10, 6),
    show=True,
):
    """
    Рисует временной ряд в вольтах (напряжение по номеру отсчёта),
    добавляет горизонтальные линии для mean и ±1σ и компактную аннотацию.
    Параметры интерфейса максимально сохранены в духе оригинального файла.
    """
    volts = adc_values * coef
    n = volts.size

    # статистики
    mean_v = np.mean(volts)

    std_v = np.std(volts, ddof=1)  # выборочное СКО (ddof=1)
    pp_v = np.max(volts) - np.min(volts)

    # Настройка внешнего вида (научная, нейтральная)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    # временной ряд
    x = np.arange(n)
    ax.plot(x, volts, lw=0.5, alpha=0.9)

    # подписи и аннотация статистики
    ax.set_xlabel("Отсчёт №", fontsize=25)
    ax.set_ylabel("Напряжение, В", fontsize=25)
    ax.set_title("Временной ряд показаний АЦП", fontsize=25)
    ax.tick_params(axis="both", labelsize=22)

    stats_text = (
        f"N = {n}\n"
        f"Среднее: {mean_v:.4f} В\n"
        f"СКО (σ): {std_v:.6f} В\n"
        f"Размах: {pp_v:.5f} В\n"
    )
    # разместим текст в правом верхнем углу графика (координаты осей)
    bbox_props = dict(
        boxstyle="round,pad=0.1", facecolor="white", alpha=0.8, edgecolor="0.7"
    )
    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=20,
        va="top",
        ha="left",
        bbox=bbox_props,
        linespacing=1,
    )

    plt.tight_layout()

    # Сохранение: PNG (как в оригинале)
    # out_prefix = Path(out_prefix)
    # png_path = out_prefix.with_suffix(".png")
    # fig.savefig(png_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    f_name = "ВЫХОД_SS494B_К_ADS1115_250SPS_60HZ_GAIN8_DIFF_RC"
    adc = read_adc_file("../text_files/" + f_name + ".txt")
    # coef = 3.3 / 4096
    # coef = 0.000125
    coef = 0.000015625
    plot_adc_timeseries(adc, coef=coef, out_prefix="../graphics/" + f_name + "_РЯД")


if __name__ == "__main__":
    main()
