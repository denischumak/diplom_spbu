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
    coef = 0.000015625
    # slice_start = 350
    # slice_end = 419
    slice_start = 200 + 60
    slice_end = 356 - 50

    # Чтение исходных данных
    # input_filename = (
    #     "../text_files/ВЫХОД_SS494B_К_ADS1115_250SPS_60HZ_GAIN8_DIFF_RC.txt"
    # )
    input_filename = "../text_files/ОЧЕНЬ_БЫСТРЫЕ_КОЛЕБАНИЯ.txt"
    with open(input_filename, "r") as f:
        raw_data = np.array([float(line.strip()) for line in f if line.strip()]) * coef
        raw_data = (
            raw_data[slice_start : slice_end + 1]
            if isinstance(slice_end, int)
            else raw_data
        )

    input_filename2 = (
        "../text_files/ВЫХОД_SS494B_К_ADS1115_250SPS_60HZ_GAIN8_DIFF_RC.txt"
    )
    with open(input_filename2, "r") as f:
        raw_data_all = (
            np.array([float(line.strip()) for line in f if line.strip()]) * coef
        )

    # Параметры фильтров

    window_ma = [2, 4, 8]
    ma = []

    window_median = [3, 5, 7]
    median = []

    alpha = [0.5, 0.4, 0.3, 0.6]
    ema = []

    # Применяем фильтры
    for el in window_ma:
        ma.append(moving_average(raw_data, el))

    for el in window_median:
        median.append(median_filter(raw_data, el))

    for el in alpha:
        ema.append(exponential_moving_average(raw_data, el))

    # --- Сохранение результатов в файлы ---

    def saveToFile(data, filename, param_name, param_value):
        with open(
            f"../filtered_files/_{filename}_{param_name}={param_value}.txt", "w"
        ) as f:
            for val in data:
                f.write(f"{val:.6f}\n")  # 6 знаков после запятой, можно изменить

    for i in range(len(window_ma)):
        saveToFile(ma[i], "MA", "N", window_ma[i])

    for i in range(len(window_median)):
        saveToFile(median[i], "MEDIAN", "N", window_median[i])

    for i in range(len(window_median)):
        saveToFile(ema[i], "EMA", "alpha", alpha[i])

    def plot_data(data, filename, param_name, param_values, label_v):
        plt.figure(figsize=(12, 6))
        plt.plot(raw_data, label="Исходный сигнал", alpha=0.6)
        for i in range(len(param_values)):
            plt.plot(
                data[i],
                label=f"{label_v} ({param_name}={param_values[i]})",
                linewidth=1,
            )
        plt.xlabel("Отсчёт №")
        plt.ylabel("Напряжение, В")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(f"../graphics/{filename}.png", dpi=300)

    plot_data(ma, "MA", "N", window_ma, "Скользящее среднее")
    plot_data(median, "MEDIAN", "N", window_median, "Медианный фильтр")
    plot_data(ema, "EMA", "alpha", alpha, "Эксп. скользящее среднее")

    def calcRms(data):
        return np.std(data, ddof=1)

    def calcP2P(data):
        return np.max(raw_data_all) - np.min(raw_data_all)

    rms_all = calcRms(raw_data_all)
    p2p = calcP2P(raw_data_all)
    print(f"dataset_RMS = {rms_all}, dataset_p2p = {p2p}")

    rms_ma = []
    p2p_ma = []

    rms_med = []
    p2p_med = []

    rms_ema = []
    p2p_ema = []

    print("------------------------------------------------")
    for i in range(len(window_ma)):
        rms_ma.append(calcRms(moving_average(raw_data_all, window_ma[i])))
        p2p_ma.append(calcP2P(moving_average(raw_data_all, window_ma[i])))
        print(
            f"N = {window_ma[i]}, RMS = {rms_ma[i]}, P2P = {p2p_ma[i]}, RMS_better_% = {((rms_all / rms_ma[i] - 1) * 100):.4g}"
        )

    print("------------------------------------------------")
    for i in range(len(window_median)):
        rms_med.append(calcRms(median_filter(raw_data_all, window_median[i])))
        p2p_med.append(calcP2P(median_filter(raw_data_all, window_median[i])))
        print(
            f"N = {window_median[i]}, RMS = {rms_med[i]}, P2P = {p2p_med[i]}, RMS_better_% = {((rms_all / rms_med[i] - 1) * 100):.4g}"
        )

    print("------------------------------------------------")
    for i in range(len(alpha)):
        rms_ema.append(calcRms(exponential_moving_average(raw_data_all, alpha[i])))
        p2p_ema.append(calcP2P(exponential_moving_average(raw_data_all, alpha[i])))
        print(
            f"alpha = {alpha[i]}, RMS = {rms_ema[i]}, P2P = {p2p_ema[i]}, RMS_better_% = {((rms_all / rms_ema[i] - 1) * 100):.4g}"
        )
