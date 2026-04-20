import numpy as np

# 1. Загружаем данные в двумерный массив (строки — отсчёты, столбцы — датчики)
data = np.loadtxt("./text_files/ALL_DATA_OUTPUT.txt")

# data.shape -> (количество_строк, 5)
# Если нужно проверить размерность:
print(f"Загружено {data.shape[0]} отсчётов, {data.shape[1]} датчиков")

# 2. Для каждого датчика (столбца) вычисляем статистики
# axis=0 означает "по столбцам"

avg = np.mean(data, axis=0)  # среднее
median = np.median(data, axis=0)  # медиана
std = np.std(data, axis=0)  # стандартное отклонение
min_vals = np.min(data, axis=0)  # минимум
max_vals = np.max(data, axis=0)  # максимум
peak_to_peak = max_vals - min_vals  # размах

# 3. Выводим результаты для каждого датчика (от 1 до 5)
index = 0
for j in range(3):
    print(f"\HALL SENSOR {index+1}:")
    print(f"  avg:     {avg[index]:.9f}")
    print(f"  median:     {median[index]:.9f}")
    print(f"  std:   {std[index]:.9f}")
    print(f"  p2p:      {peak_to_peak[index]:.9f}")
    index += 1
for i in ["X", "Y", "Z"]:
    print(f"\ACCEL {i}:")
    print(f"  avg:     {avg[index]:.9f}")
    print(f"  median:     {median[index]:.9f}")
    print(f"  std:   {std[index]:.9f}")
    print(f"  p2p:      {peak_to_peak[index]:.9f}")
    index += 1
for i in ["X", "Y", "Z"]:
    print(f"\GYRO {i}:")
    print(f"  avg:     {avg[index]:.9f}")
    print(f"  median:     {median[index]:.9f}")
    print(f"  std:   {std[index]:.9f}")
    print(f"  p2p:      {peak_to_peak[index]:.9f}")
    index += 1
for i in ["RAW", "PITCH"]:
    print(f"{i}:")
    print(f"  avg:     {avg[index]:.9f}")
    print(f"  median:     {median[index]:.9f}")
    print(f"  std:   {std[index]:.9f}")
    print(f"  p2p:      {peak_to_peak[index]:.9f}")
    index += 1
