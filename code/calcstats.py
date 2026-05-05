import numpy as np

# 1. Загружаем данные в двумерный массив (строки — отсчёты, столбцы — датчики)
data = np.loadtxt(
    r"C:\Users\User\Desktop\diplom\text_files\ALL_DATA_OUTPUT_100hz.txt"
)

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

# 100 HZ
# std:   0.379137628
#   p2p:      2.099000000


# 62.5 HZ
#   std:   0.498702924
#   p2p:      2.692733333

# 3. Выводим результаты для каждого датчика (от 1 до 5)
index = 0
for j in range(3):
    print(f"\HALL SENSOR {index+1}:")
    print(f"  avg:     {avg[index]:.9f}")
    print(f"  median:     {median[index]:.9f}")
    print(f"  std:   {std[index]:.9f}")
    print(f"  p2p:      {peak_to_peak[index]:.9f}")
    index += 1
# print(f"  avg:     {avg.mean():.9f}")
# print(f"  median:     {median.mean():.9f}")
# print(f"  std:   {std.mean():.9f}")
# print(f"  p2p:      {peak_to_peak.mean():.9f}")

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

for i in ["q1", "q2", "q3", "q4"]:
    print(f"{i}:")
    print(f"  avg:     {avg[index]:.9f}")
    print(f"  median:     {median[index]:.9f}")
    print(f"  std:   {std[index]:.9f}")
    print(f"  p2p:      {peak_to_peak[index]:.9f}")
    index += 1


# \HALL SENSOR 1:
#   avg:     19.856340230
#   median:     19.875000000
#   std:   0.165499988
#   p2p:      1.218700000
# \HALL SENSOR 2:
#   avg:     41.792564500
#   median:     41.828100000
#   std:   0.309565033
#   p2p:      1.796900000
# \HALL SENSOR 3:
#   avg:     29.908187680
#   median:     29.921900000
#   std:   0.309880374
#   p2p:      1.781200000
# \ACCEL X:
#   avg:     -0.137254830
#   median:     -0.137300000
#   std:   0.000794692
#   p2p:      0.005600000
# \ACCEL Y:
#   avg:     0.180091120
#   median:     0.180100000
#   std:   0.000890061
#   p2p:      0.006500000
# \ACCEL Z:
#   avg:     0.987001410
#   median:     0.987000000
#   std:   0.000961809
#   p2p:      0.007800000
# \GYRO X:
#   avg:     0.007349410
#   median:     -0.006400000
#   std:   0.036872114
#   p2p:      0.244100000
# \GYRO Y:
#   avg:     0.011853590
#   median:     0.000300000
#   std:   0.030399398
#   p2p:      0.213600000
# \GYRO Z:
#   avg:     0.017678820
#   median:     0.018800000
#   std:   0.032821993
#   p2p:      0.274600000
# q1:
#   avg:     0.993650760
#   median:     0.993600000
#   std:   0.000499975
#   p2p:      0.006700000
# q2:
#   avg:     0.088445980
#   median:     0.088900000
#   std:   0.005417638
#   p2p:      0.090400000
# q3:
#   avg:     0.068535280
#   median:     0.068900000
#   std:   0.004254631
#   p2p:      0.071000000
# q4:
#   avg:     0.007883400
#   median:     0.007800000
#   std:   0.004565243
#   p2p:      0.016000000