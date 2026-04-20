import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1. Чтение данных из файла
# -------------------------------------------------


# Чтение данных из файла
input_filename = "ВЫХОД_ДИФФ+RC.txt"
with open(input_filename, "r") as f:
    lines = f.readlines()
    # Преобразуем строки в числа (float, можно заменить на int если нужно)
data = np.array([float(line.strip()) for line in lines if line.strip()]) * 0.000015625
data = data[0:201]

# Время — номер отсчёта (можно использовать индекс)
time = np.arange(len(data))


def median_filter(data, window_size):
    """
    Применяет медианный фильтр к списку data с окном window_size.
    Возвращает новый список отфильтрованных значений.
    """
    half = window_size // 2
    filtered = []
    for i in range(len(data)):
        # Определяем границы окна
        left = max(0, i - half)
        right = min(len(data) - 1, i + half)
        # Формируем окно и сортируем
        window = data[left : right + 1]
        window_sorted = sorted(window)
        # Медиана - средний элемент
        median = window_sorted[len(window_sorted) // 2]
        filtered.append(median)
    return filtered


# -------------------------------------------------
# 2. Построение графика
# -------------------------------------------------
plt.figure(figsize=(12, 6))

# Несглаженный сигнал — синий, полупрозрачный, тонкая линия
plt.plot(time, data, color="blue", alpha=0.5, linewidth=1.5, label="Несглаженный")

# Сглаженный сигнал — красный, плотная линия
plt.plot(
    time, median_filter(data, 3), color="red", linewidth=0.4, label="Сглаженный (M=3)"
)
plt.plot(
    time, median_filter(data, 5), color="black", linewidth=0.2, label="Сглаженный (M=5)"
)

# -------------------------------------------------
# 3. Оформление
# -------------------------------------------------
plt.xlabel("Номер отсчёта", fontsize=12)
plt.ylabel("Напряжение, ", fontsize=12)
plt.title("Сравнение несглаженного и сглаженного сигнала", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# -------------------------------------------------
# 4. Сохранение и отображение
# -------------------------------------------------

plt.show()

# -------------------------------------------------
# 5. Краткая статистика (опционально)
# -------------------------------------------------
print(f"Всего отсчётов: {len(data)}")
print(
    f"Несглаженный: min={np.min(data):.4f}, max={np.max(data):.4f}, RMS={np.std(data, ddof=1):.4f} В"
)
