import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import math


def read_adc_file(path):
    """Читает файл, возвращает numpy array с числовыми значениями (float)."""
    vals = []
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
                cnt += 1
            except ValueError:
                # пропускаем строки, которые не парсятся (можно логировать при необходимости)
                continue
    if len(vals) == 0:
        raise ValueError(f"В файле {path} нет числовых данных.")
    return np.array(vals, dtype=float)


def plot_adc_histogram(
    adc_values,
    coef,
    bins="auto",
    figsize=(10, 6),
    show=True,
):
    """
    adc_values: numpy array ADC counts
    coef: множитель для перехода в вольты (вольт/единица АЦП)
    bins: int or 'auto' или sequence
    out_prefix: имя файла (без расширения) для сохранения .png и .pdf
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
            "font.size": 30,
            "axes.labelsize": 25,
            "axes.titlesize": 25,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    # гистограмма (плотность)
    ax.hist(volts, bins=bins, alpha=0.85, edgecolor="k", linewidth=0.4)

    # вертикальные линии: mean и mean ± std
    ax.axvline(mean_v, color="black", linewidth=6.0, linestyle="-", label="Среднее")
    ax.axvline(mean_v + std_v, color="gray", linewidth=6, linestyle="--", label="±1σ")
    ax.axvline(mean_v - std_v, color="gray", linewidth=6, linestyle="--")

    # подписи и аннотация статистики
    ax.set_xlabel("Напряжение, В", fontsize=25)
    ax.set_ylabel("Количество отсчетов", fontsize=25)
    ax.set_title("Гистограмма показаний АЦП", fontsize=25)
    ax.tick_params(axis="both", labelsize=23)

    stats_text = (
        f"N = {n}\n"
        f"Среднее: {mean_v:.4f} В\n"
        f"СКО (σ): {std_v:.5f} В\n"
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

    ax.legend(fontsize=15, loc="upper right", framealpha=0.8)

    plt.tight_layout()

    # Сохранение: PDF (вектор) и PNG (растр для превью)
    # out_prefix = Path(out_prefix)
    # png_path = out_prefix.with_suffix(".png")
    # fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    f_name = "ВЫХОД_SS494B_К_ADS1115_250SPS_60HZ_GAIN1_SE_RC"
    adc = read_adc_file("../text_files/" + f_name + ".txt")
    # coef = 3.3 / 4096
    coef = 0.000125
    # coef = 0.000015625
    bins = "fd"

    plot_adc_histogram(adc, coef=coef, bins=bins)


if __name__ == "__main__":
    main()
