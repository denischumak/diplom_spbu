import matplotlib.pyplot as plt
import numpy as np

vals = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18][::-1]
global_zero = 1.666

# Данные для SS494B
zero_494 = 1.640
v_494 = [
    1.645,
    1.645,
    1.647,
    1.648,
    1.650,
    1.654,
    1.659,
    1.666,
    1.677,
    1.697,
    1.732,
    1.800,
    1.949,
    2.337,
][::-1]
zero_shift_494 = global_zero - zero_494

# Данные для SS49E
zero_49e = 1.672
v_49e = [
    1.675,
    1.676,
    1.676,
    1.677,
    1.679,
    1.681,
    1.684,
    1.689,
    1.696,
    1.708,
    1.731,
    1.775,
    1.871,
    2.118,
][::-1]
zero_shift_49e = global_zero - zero_49e

# Данные для SS495A1
zero_495 = 1.650
v_495 = [
    1.652,
    1.653,
    1.654,
    1.655,
    1.656,
    1.658,
    1.661,
    1.666,
    1.673,
    1.685,
    1.706,
    1.749,
    1.845,
    2.107,
][::-1]
zero_shift_495 = global_zero - zero_495


v_494_centred = [val + zero_shift_494 for val in v_494]
v_49e_centred = [val + zero_shift_49e for val in v_49e]
v_495_centred = [val + zero_shift_495 for val in v_495]

# Преобразуем X в расстояние: dist = 20 - X
dist = [20 - val for val in vals]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(dist, v_494_centred, ":", label="SS494B", linewidth=6, markersize=3)
plt.plot(
    dist,
    v_49e_centred,
    "--",
    label="SS49E",
    linewidth=6,
    markersize=3,
)
plt.plot(dist, v_495_centred, "-", label="SS495A1", linewidth=3, markersize=3)

plt.xlabel("Расстояние до магнита, см", fontsize=20)
plt.ylabel("Выходное напряжение, В", fontsize=20)
plt.title("Зависимость напряжения датчиков Холла от расстояния", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(np.arange(2, 17, 1), fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(1, 17)
plt.tight_layout()
plt.show()
