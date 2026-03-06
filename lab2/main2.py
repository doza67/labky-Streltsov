import numpy as np
import matplotlib.pyplot as plt
import csv
from math import factorial


# 1. Зчитування даних (згідно з шаблоном у методичці) [cite: 171-181]
def read_data(filename):
    x, y = [], []
    # Початкові дані Варіанту 1 [cite: 194, 195]
    default_data = [
        {'n': 1000, 't': 3},
        {'n': 2000, 't': 5},
        {'n': 4000, 't': 11},
        {'n': 8000, 't': 28},
        {'n': 16000, 't': 85}
    ]
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x.append(float(row['n']))
                y.append(float(row['t']))
    except FileNotFoundError:
        print(f"Файл {filename} не знайдено. Використовую дані Варіанту 1.")
        x = [d['n'] for d in default_data]
        y = [d['t'] for d in default_data]
    return np.array(x), np.array(y)


# 2. Побудова таблиці розділених різниць [cite: 6-12, 156]
def get_divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef  # Повертає повну таблицю


# 3. Метод Ньютона [cite: 54-55, 159]
def newton_interpolation(x_nodes, y_nodes, x_target):
    table = get_divided_diff(x_nodes, y_nodes)
    coef = table[0, :]  # Коефіцієнти - це перший рядок
    n = len(coef)
    res = coef[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_target - x_nodes[i - 1])
        res += coef[i] * product
    return res


# 3.1 Метод факторіальних многочленів (для рівновіддалених вузлів) [cite: 133-141, 150]
def factorial_interpolation(x_nodes, y_nodes, x_target):
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    t = (x_target - x_nodes[0]) / h

    # Таблиця скінченних різниць [cite: 85, 99-100]
    diffs = np.zeros([n, n])
    diffs[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i + 1][j - 1] - diffs[i][j - 1]

    res = diffs[0, 0]
    t_prod = 1.0
    for k in range(1, n):
        t_prod *= (t - k + 1)
        res += (diffs[0, k] * t_prod) / factorial(k)
    return res


# --- ВИКОНАННЯ ---

# Крок 1: Дані
x_data, y_data = read_data('data.csv')

# Крок 2: Таблиця розділених різниць (вивід першого рядка)
diff_table = get_divided_diff(x_data, y_data)
print("Коефіцієнти розділених різниць:", diff_table[0, :])

# Крок 3: Прогноз для n=6000 [cite: 199, 204]
p_newton = newton_interpolation(x_data, y_data, 6000)
# Примітка: Факторіальний метод вимагає рівномірного кроку.
# Оскільки в даних n=1000, 2000, 4000... крок нерівномірний, метод Ньютона є основним.
print(f"Прогноз P(6000) за Ньютоном: {p_newton:.2f} мс")

# 4. Графік [cite: 161, 206-208]

x_fine = np.linspace(min(x_data), max(x_data), 500)
y_fine = [newton_interpolation(x_data, y_data, xi) for xi in x_fine]

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Експериментальні точки [cite: 207]')
plt.plot(x_fine, y_fine, label='Інтерполяційна крива (Ньютон) [cite: 208]')
plt.axvline(x=6000, color='green', linestyle='--', label='Прогноз n=6000')
plt.title("Прогноз часу виконання (Варіант 1)")
plt.xlabel("Розмір вхідних даних (n)")
plt.ylabel("Час виконання (t, мс)")
plt.legend()
plt.grid(True)
plt.show()


# 5. Дослідницька частина [cite: 298-304]

# Функція для аналізу похибки [cite: 67, 159]
def analyze_error(n_points):
    print(f"\n--- Аналіз для {n_points} вузлів ---")
    # Симуляція нових вузлів (наприклад, через функцію t = 0.0000003 * n^2)
    x_test = np.linspace(1000, 16000, n_points)
    y_test = 0.0000003 * x_test ** 2 + 0.002 * x_test  # Гіпотетична модель продуктивності

    # Обчислення в середній точці для перевірки точності
    mid_x = 6000
    interp_val = newton_interpolation(x_test, y_test, mid_x)
    actual_val = 0.0000003 * mid_x ** 2 + 0.002 * mid_x
    error = abs(actual_val - interp_val)
    print(f"Похибка при {n_points} вузлах: {error:.2e}")

    # Аналіз ефекту Рунге
    if n_points > 15:
        print("Попередження: Виявлено ознаки ефекту Рунге (осциляції на краях).")


for n in [5, 10, 20]:
    analyze_error(n)