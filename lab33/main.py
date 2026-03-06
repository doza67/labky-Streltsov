import numpy as np
import matplotlib.pyplot as plt
import csv


# 1. Функції для методу найменших квадратів (МНК)
def form_matrix(x, m):
    """Формування матриці системи лінійних рівнянь [cite: 26, 134]"""
    matrix = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            matrix[i, j] = np.sum(x ** (i + j))
    return matrix


def form_vector(x, y, m):
    """Формування вектора вільних членів [cite: 27, 140]"""
    vector = np.zeros(m + 1)
    for i in range(m + 1):
        vector[i] = np.sum(y * (x ** i))
    return vector


def gauss_solve(A, b):
    """Розв'язання СЛАР методом Гаусса з вибором головного елемента [cite: 35, 145]"""
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    # Прямий хід [cite: 37, 146]
    for k in range(n):
        # Вибір головного елемента по стовпцю [cite: 44, 147]
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід [cite: 37, 150]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol


def calculate_polynomial(x, coef):
    """Обчислення значення многочлена [cite: 7, 155]"""
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x ** i)
    return y_poly


def calculate_variance(y_true, y_approx):
    """Обчислення середньоквадратичного відхилення (дисперсії) [cite: 30, 160]"""
    return np.sqrt(np.mean((y_true - y_approx) ** 2))


# 2. Основна частина роботи
# Дані з прикладу в описі [cite: 100-124]
months = np.arange(1, 25)
temps = np.array([-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0, -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3])

variances = []
max_m = 10  # Досліджуємо степені від 1 до 10 [cite: 82]

print("Степінь (m) | Дисперсія (δ)")
print("-" * 30)

for m in range(1, max_m + 1):
    A_mat = form_matrix(months, m)
    B_vec = form_vector(months, temps, m)
    coefficients = gauss_solve(A_mat, B_vec)

    y_pred = calculate_polynomial(months, coefficients)
    var = calculate_variance(temps, y_pred)
    variances.append(var)
    print(f"{m:10d} | {var:12.4f}")

# 3. Вибір оптимального степеня (мінімум дисперсії) [cite: 83, 170]
optimal_m = np.argmin(variances) + 1
print(f"\nОптимальний степінь многочлена: {optimal_m}")

# Фінальна апроксимація з оптимальним m
final_coef = gauss_solve(form_matrix(months, optimal_m), form_vector(months, temps, optimal_m))
y_approx = calculate_polynomial(months, final_coef)
error = np.abs(temps - y_approx)  # Похибка [cite: 81, 186]

# 4. Екстраполяція на 3 місяці [cite: 98, 181]
future_months = np.array([25, 26, 27])
future_temps = calculate_polynomial(future_months, final_coef)
print(f"Прогноз на місяці 25, 26, 27: {np.round(future_temps, 2)}")

# 5. Візуалізація результатів [cite: 96, 97, 193]
plt.figure(figsize=(12, 8))

# Графік апроксимації
plt.subplot(2, 1, 1)
plt.scatter(months, temps, color='red', label='Фактичні дані')
plt.plot(months, y_approx, label=f'Апроксимація (m={optimal_m})', color='blue')
plt.plot(future_months, future_temps, 'o--', color='green', label='Прогноз')
plt.title('Апроксимація температури методом найменших квадратів')
plt.legend()
plt.grid(True)

# Графік похибки
plt.subplot(2, 1, 2)
plt.bar(months, error, color='gray', alpha=0.7, label='Абсолютна похибка')
plt.title('Похибка апроксимації у вузлах')
plt.xlabel('Місяць')
plt.ylabel('Δ Temp')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Графік залежності дисперсії від степеня [cite: 82]
plt.figure()
plt.plot(range(1, max_m + 1), variances, marker='o')
plt.title('Залежність дисперсії від степеня многочлена')
plt.xlabel('Степінь m')
plt.ylabel('Дисперсія δ')
plt.grid(True)
plt.show()