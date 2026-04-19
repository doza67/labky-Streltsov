import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Глобальна змінна для підрахунку викликів функції (для адаптивного методу)
eval_count = 0


# 1. Задана функція навантаження на сервер
def f(x):
    global eval_count
    eval_count += 1
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24

# 2. Точне значення інтегралу (за допомогою вбудованої бібліотеки високої точності)
I0, _ = spi.quad(f, a, b)
print(f"2. Точне значення інтегралу I0: {I0:.12f}")


# 3. Складова формула Сімпсона
def simpson(func, a, b, N):
    if N % 2 != 0: N += 1  # N має бути парним
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])


# 4. Дослідження точності залежно від N
N_values = np.arange(10, 1002, 2)
errors = []
N_opt = None
eps_opt = None

for N in N_values:
    I_N = simpson(f, a, b, N)
    error = abs(I_N - I0)
    errors.append(error)
    # Шукаємо N, при якому точність досягає 1e-12
    if error <= 1e-12 and N_opt is None:
        N_opt = N
        eps_opt = error

print(f"4. Оптимальне розбиття N_opt (для eps <= 1e-12): {N_opt}")
print(f"   Похибка при N_opt (eps_opt): {eps_opt:.2e}")

# Побудова графіка залежності похибки від N
plt.figure(figsize=(10, 6))
plt.plot(N_values, errors, label=r'Похибка Сімпсона $\epsilon(N)$')
plt.axhline(1e-12, color='r', linestyle='--', label='Задана точність 1e-12')
if N_opt:
    plt.scatter([N_opt], [eps_opt], color='red')
plt.yscale('log')
plt.title('Залежність похибки інтегрування від кількості розбиттів N')
plt.xlabel('Кількість розбиттів (N)')
plt.ylabel(r'Похибка $\epsilon(N)$')
plt.grid(True)
plt.legend()
plt.show()

# 5. Обчислення похибки для N0
N0 = max(8, int(N_opt / 10))
if N0 % 8 != 0: N0 += (8 - N0 % 8)  # Робимо кратним 8

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"5. Вибрано N0 = {N0} (кратне 8)")
print(f"   Інтеграл при N0: {I_N0:.12f}")
print(f"   Похибка eps0: {eps0:.2e}")

# 6. Метод Рунге-Ромберга
I_N0_half = simpson(f, a, b, int(N0 / 2))
I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I0)
print(f"6. Уточнення за методом Рунге-Ромберга: {I_R:.12f}")
print(f"   Похибка epsR: {epsR:.2e}")

# 7. Метод Ейткена
I_N0_quarter = simpson(f, a, b, int(N0 / 4))
# Формула Ейткена
numerator = (I_N0_half) ** 2 - I_N0 * I_N0_quarter
denominator = 2 * I_N0_half - (I_N0 + I_N0_quarter)
I_E = numerator / denominator
epsE = abs(I_E - I0)

# Оцінка порядку точності
p = (1 / np.log(2)) * np.log(abs((I_N0_quarter - I_N0_half) / (I_N0_half - I_N0)))

print(f"7. Уточнення за методом Ейткена: {I_E:.12f}")
print(f"   Похибка epsE: {epsE:.2e}")
print(f"   Оцінений порядок методу p: {p:.4f}")


# 9. Адаптивний алгоритм
def adaptive_simpson_recursive(a, b, tol):
    c = (a + b) / 2
    h = b - a

    # Інтеграл на одному відрізку
    I1 = (h / 6) * (f(a) + 4 * f(c) + f(b))

    # Інтеграл на двох половинках
    d = (a + c) / 2
    e = (c + b) / 2
    I2 = (h / 12) * (f(a) + 4 * f(d) + 2 * f(c) + 4 * f(e) + f(b))

    # Умова збіжності
    if abs(I1 - I2) <= tol:
        return I2
    else:
        return adaptive_simpson_recursive(a, c, tol / 2) + adaptive_simpson_recursive(c, b, tol / 2)


print(f"9. Адаптивний алгоритм:")
tols = [1e-4, 1e-6, 1e-8]
for tol in tols:
    eval_count = 0  # Обнуляємо лічильник перед кожним запуском
    I_adapt = adaptive_simpson_recursive(a, b, tol)
    eps_adapt = abs(I_adapt - I0)
    print(f"   При tol = {tol}: Інтеграл = {I_adapt:.12f}, Похибка = {eps_adapt:.2e}, Викликів функції = {eval_count}")