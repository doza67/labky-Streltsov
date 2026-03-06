import requests
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ЕТАП 1-4: Отримання та підготовка даних
# ==========================================

# 1. Запит до Open-Elevation API [cite: 99, 100]
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]
n_points = len(results)

# 2-3. Табуляція вузлів [cite: 104, 108]
print("Кількість вузлів:", n_points)
print("\nТабуляція вузлів:")
print(" i |  Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


# 4. Обчислення кумулятивної відстані [cite: 115, 116]
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # [cite: 121]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2  # [cite: 127, 128]
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # [cite: 129]


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0.0]

for i in range(1, n_points):
    d = haversine(*coords[i - 1], *coords[i])  # [cite: 133]
    distances.append(distances[-1] + d)


# ==========================================
# ЕТАП 6-9: Кубічні сплайни та метод прогонки
# ==========================================

def cubic_spline_interpolation(x, y):
    n = len(x) - 1
    h = np.diff(x)

    # 6. Знаходження коефіцієнтів системи лінійних алгебраїчних рівнянь [cite: 139, 140]
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])  # [cite: 49]

    # 7. Метод прогонки (пряма та зворотна) [cite: 50, 53, 141]
    A = np.zeros(n)
    B = np.zeros(n)

    for i in range(1, n):
        denominator = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator  # [cite: 63]
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denominator  # [cite: 64]

    c = np.zeros(n + 1)
    for i in range(n - 1, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]  # [cite: 71]

    # 8-9. Обчислення коефіцієнтів a, b, d [cite: 142, 143]
    a = np.array(y[:-1])  #
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3  # [cite: 38]
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])  # [cite: 37]

    return a, b, c[:-1], d


def evaluate_spline(x_eval, x_nodes, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for i in range(len(x_eval)):
        # Знаходимо потрібний інтервал
        idx = np.searchsorted(x_nodes, x_eval[i]) - 1
        idx = np.clip(idx, 0, len(a) - 1)

        dx = x_eval[i] - x_nodes[idx]
        y_eval[i] = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3  #
    return y_eval


# ==========================================
# ЕТАП 10-12: Побудова графіків (10, 15, 20 вузлів)
# ==========================================

plt.figure(figsize=(12, 8))
x_fine = np.linspace(min(distances), max(distances), 500)

node_counts = [10, 15, len(distances)]  # Використовуємо 10, 15 та всі доступні (близько 20) [cite: 145]
colors = ['r', 'g', 'b']

plt.plot(distances, elevations, 'k--', label='Оригінальні дані (API)', linewidth=2)

for count, color in zip(node_counts, colors):
    idx = np.linspace(0, len(distances) - 1, count, dtype=int)
    x_nodes = np.array(distances)[idx]
    y_nodes = np.array(elevations)[idx]

    a, b, c, d = cubic_spline_interpolation(x_nodes, y_nodes)
    y_spline = evaluate_spline(x_fine, x_nodes, a, b, c, d)

    plt.plot(x_fine, y_spline, color=color, label=f'Сплайн ({count} вузлів)')
    plt.scatter(x_nodes, y_nodes, color=color, zorder=5)

plt.title('Профіль висоти маршруту: Заросляк - Говерла [cite: 78]')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# ДОДАТКОВЕ ЗАВДАННЯ [cite: 150]
# ==========================================
print("\n--- Додаткове завдання ---")

# 1. Характеристики маршруту
print("Загальна довжина маршруту (м):", f"{distances[-1]:.2f}")  # [cite: 153]

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n_points))  # [cite: 155]
print("Сумарний набір висоти (м):", f"{total_ascent:.2f}")

total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n_points))  # [cite: 157]
print("Сумарний спуск (м):", f"{total_descent:.2f}")

# 2. Аналіз градієнта
grad_full = np.gradient(elevations, distances) * 100  # [cite: 165]
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")  # [cite: 166, 167]
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")  # [cite: 168]
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")  # [cite: 169]

# 3. Механічна енергія підйому
mass = 80  # [cite: 172]
g = 9.81  # [cite: 173]
energy = mass * g * total_ascent  # [cite: 174, 175, 176, 177]

print(f"Механічна робота (Дж): {energy:.2f}")  # [cite: 178]
print(f"Механічна робота (кДж): {energy / 1000:.2f}")  # [cite: 178]
print(f"Енергія (ккал): {energy / 4184:.2f}")  # [cite: 180]