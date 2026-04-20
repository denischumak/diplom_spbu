import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    result = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        result.append(sum(data[start:end]) / (end - start))
    return np.array(result)


def median_filter(data, window_size):
    result = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        result.append(np.median(data[start:end]))
    return np.array(result)


def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)


# --- Основная программа ---
if __name__ == "__main__":
    # GAIN = 8
    coef = 1

    # Чтение исходных данных
    # input_filename = (
    #     "../text_files/ВЫХОД_SS494B_К_ADS1115_250SPS_60HZ_GAIN8_DIFF_RC.txt"
    # )
    input_filename = "../text_files/ПОКАЗАНИЯ_С_ДАТЧИКА_60ГЦ.txt"
    with open(input_filename, "r") as f:
        raw_data = np.array([float(line.strip()) for line in f if line.strip()]) * coef

    # with open(input_filename, "r") as f:
    #     raw_data = (
    #         np.array(
    #             [
    #                 float(line.split()[0].strip())
    #                 for line in f
    #                 if line.split()[0].strip()
    #             ]
    #         )
    #         * coef
    #     )
    #     # raw_data = (
    #     #     raw_data[0 : slice_num + 1] if isinstance(slice_num, int) else raw_data
    #     # )
    #     raw_data = raw_data[90:131]

    # Параметры фильтров
    window_ma = 4  # больше 4 - плохо
    window_median = 3  # для 5 уже задержка большая
    alpha = 0.5  # меньше 0.5 не стоит

    # Применяем фильтры
    ma = moving_average(raw_data, window_ma)
    med = median_filter(raw_data, window_median)
    ema = exponential_moving_average(raw_data, alpha)

    # --- Сохранение результатов в файлы ---
    # Скользящее среднее
    with open("../filtered_files/MA_" + f"N={window_ma}" + ".txt", "w") as f:
        for val in ma:
            f.write(f"{val:.6f}\n")  # 6 знаков после запятой, можно изменить

    # Медианный фильтр
    with open("../filtered_files/MEDIAN_" + f"N={window_median}" + ".txt", "w") as f:
        for val in med:
            f.write(f"{val:.6f}\n")

    # Экспоненциальное скользящее среднее
    with open("../filtered_files/EMA_" + f"N={alpha}" + ".txt", "w") as f:
        for val in ema:
            f.write(f"{val:.6f}\n")

    print(f"raw_data_std={np.std(raw_data, ddof=1)}")
    print(f"ma_std={np.std(ma, ddof=1)}")
    print(f"median_std={np.std(med, ddof=1)}")
    print(f"ema_std={np.std(ema, ddof=1)}")
    print()
    print(f"raw_data_p2p={np.max(raw_data) - np.min(raw_data)}")
    print(f"ma_p2p={np.max(ma) - np.min(ma)}")
    print(f"median_p2p={np.max(med) - np.min(med)}")
    print(f"ema_p2p={np.max(ema) - np.min(ema)}")

    # (Опционально) Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data, alpha=0.6, linewidth=1.5)
    # plt.plot(ma, label=f"Скользящее среднее (окно={window_ma})", linewidth=1)
    # plt.plot(med, label=f"Медианный фильтр (окно={window_median})", linewidth=1)
    # plt.plot(ema, label=f"EMA (alpha={alpha})", linewidth=5, linestyle="--")
    plt.xlabel("Отсчёт №", fontsize=20)
    plt.ylabel("Напряжение, мВ", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)
    plt.grid(True)
    plt.show()
