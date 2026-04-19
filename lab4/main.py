import numpy as np
import matplotlib.pyplot as plt

# --- 1. Аналітичне розв'язання [cite: 150] ---
# Функція вологості ґрунту [cite: 185]
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

# Явний вираз для першої похідної [cite: 185]
def dM_dt_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Точка для обчислень 
t0 = 1.0

# Точне значення похідної в точці t0 
exact_val = dM_dt_exact(t0)
print("--- 1. Аналітичне розв'язання ---")
print(f"Точне значення похідної в точці t0 = {t0}: y'(x0) = {exact_val}")

# Побудова графіка моделі вологості (як у прикладі) [cite: 138-148]
t_vals = np.linspace(0, 20, 500)
plt.figure(figsize=(8, 5))
plt.plot(t_vals, M(t_vals), color="#457b9d")
plt.title("Soil Moisture Model M(t)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.grid(True)
plt.show()

# --- 2. Дослідження залежності похибки від кроку h [cite: 186-191] ---
# Формула для апроксимації (центральна різниця) [cite: 186]
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Діапазон кроків h від 10^-20 до 10^3 [cite: 190]
# Використовуємо float64, хоча межа машинної точності ~10^-16
h_vals = np.logspace(-20, 3, 500)
errors = [abs(central_diff(M, t0, h) - exact_val) for h in h_vals]

# Знаходимо оптимальний крок h0 (мінімальна похибка) [cite: 190]
# Відфільтровуємо нульові похибки (виникають через обмеження float) для коректного пошуку
valid_indices = [i for i, e in enumerate(errors) if e > 0]
if valid_indices:
    min_error_idx = min(valid_indices, key=lambda i: errors[i])
    h0 = h_vals[min_error_idx]
    R0 = errors[min_error_idx]
else:
    h0 = 1e-7
    R0 = abs(central_diff(M, t0, h0) - exact_val)

print("\n--- 2. Дослідження кроку h ---")
print(f"Оптимальний крок h0: {h0:.1e}")
print(f"Найкраща досягнута точність R0: {R0}")

# Графік залежності похибки від кроку h (логарифмічний масштаб)
plt.figure(figsize=(8, 5))
plt.loglog(h_vals, errors, color="#e63946")
plt.axvline(h0, color='gray', linestyle='--', label=f'Оптимальне h0 ≈ {h0:.1e}')
plt.title("Залежність похибки чисельного диференціювання від кроку h")
plt.xlabel("Крок h")
plt.ylabel("Похибка R = |y'(h) - y'|")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# --- 3-6. Метод Рунге-Ромберга [cite: 192-197] ---
print("\n--- 3-6. Метод Рунге-Ромберга ---")
h_work = 10**-3  # [cite: 192]

# Значення похідної з кроками h та 2h [cite: 193, 194]
y_prime_h = central_diff(M, t0, h_work)
y_prime_2h = central_diff(M, t0, 2 * h_work)

# Похибка при кроці h [cite: 195]
R1 = abs(y_prime_h - exact_val)

# Уточнене значення за методом Рунге-Ромберга [cite: 196]
y_R_prime = y_prime_h + (y_prime_h - y_prime_2h) / 3

# Похибка R2 [cite: 197]
R2 = abs(y_R_prime - exact_val)

print(f"Робочий крок h = {h_work}")
print(f"y'(h) = {y_prime_h}, Похибка R1 = {R1}")
print(f"y'(2h) = {y_prime_2h}")
print(f"Уточнене значення Рунге-Ромберга y'_R = {y_R_prime}")
print(f"Похибка R2 = {R2}")
print(f"Характер зміни: похибка зменшилась у {R1/R2:.2f} разів.")

# --- 7. Метод Ейткена [cite: 198-208] ---
print("\n--- 7. Метод Ейткена ---")
# Значення похідної з кроком 4h [cite: 199, 203, 204]
y_prime_4h = central_diff(M, t0, 4 * h_work)

# Уточнене значення за методом Ейткена [cite: 206, 207]
numerator_E = (y_prime_2h)**2 - y_prime_4h * y_prime_h
denominator_E = 2 * y_prime_2h - (y_prime_4h + y_prime_h)
y_E_prime = numerator_E / denominator_E

# Порядок точності формули [cite: 206, 207]
ratio_p = abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h))
p = (1 / np.log(2)) * np.log(ratio_p)

# Похибка R3 
R3 = abs(y_E_prime - exact_val)

print(f"y'(4h) = {y_prime_4h}")
print(f"Уточнене значення Ейткена y'_E = {y_E_prime}")
print(f"Оцінка порядку точності p = {p:.2f}")
print(f"Похибка R3 = {R3}")