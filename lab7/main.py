"""
Лабораторна робота №8
Ітераційні методи розв'язку систем лінійних алгебраїчних рівнянь.
Методи: проста ітерація, Якобі, Зейдель.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time

# ─────────────────────────────────────────────────────────────────
# ПАРАМЕТРИ
# ─────────────────────────────────────────────────────────────────
N          = 100
TRUE_X     = 2.5
EPS_TARGET = 1e-14
MAX_ITER   = 10_000
SEED       = 42
OUT        = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ
# ─────────────────────────────────────────────────────────────────
def mat_vec(A, x):
    """Добуток матриці на вектор."""
    return A @ x

def vec_norm(v):
    """Нескінченна норма вектора max|vi|."""
    return float(np.max(np.abs(v)))

def mat_norm(M):
    """Нескінченна норма матриці max_i Σ|mij|."""
    return float(np.max(np.sum(np.abs(M), axis=1)))

def save_matrix(path, M): np.savetxt(path, M, fmt="%.10f")
def load_matrix(path):    return np.loadtxt(path)
def save_vector(path, v): np.savetxt(path, v, fmt="%.10f")
def load_vector(path):    return np.loadtxt(path)

# ─────────────────────────────────────────────────────────────────
# КРОК 1 – Генерація матриці A та вектора B
# ─────────────────────────────────────────────────────────────────
def generate_diag_dominant(n, seed=42):
    """
    Генерація матриці з діагональним переважанням:
    |a_ii| > Σ_{j≠i} |a_ij|
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-5, 5, (n, n))
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        A[i, i] = row_sum + rng.uniform(1, 3)   # гарантоване переважання
    return A

print("=" * 65)
print("КРОК 1: Генерація матриці A (діагональне переважання) та вектора B")
print("=" * 65)

A = generate_diag_dominant(N, SEED)
x_true = np.full(N, TRUE_X)
B = mat_vec(A, x_true)

save_matrix(os.path.join(OUT, "lab8_A.txt"), A)
save_vector(os.path.join(OUT, "lab8_B.txt"), B)

# Перевірка діагонального переважання
diag_dom_ok = all(
    abs(A[i, i]) > sum(abs(A[i, j]) for j in range(N) if j != i)
    for i in range(N)
)
print(f"Матриця A ({N}×{N}) збережена → lab8_A.txt")
print(f"Вектор B ({N}) збережений   → lab8_B.txt")
print(f"Діагональне переважання виконується: {diag_dom_ok}")
print(f"Точний розв'язок: всі x_i = {TRUE_X}")

# ─────────────────────────────────────────────────────────────────
# КРОК 3 – Початкове наближення x_i^(0) = 1.0/(i+1)
# ─────────────────────────────────────────────────────────────────
X0 = np.array([1.0 / (i + 1) for i in range(N)])
print(f"\nПочаткове наближення: x_i^(0) = 1/(i+1),  перші 5: {X0[:5]}")

# ─────────────────────────────────────────────────────────────────
# МЕТОД ПРОСТОЇ ІТЕРАЦІЇ
# ─────────────────────────────────────────────────────────────────
def simple_iteration(A, b, x0, eps=1e-14, max_iter=10_000):
    """
    Метод простої ітерації:  X^(k+1) = X^(k) - τ*(A·X^(k) - f)
    τ вибирається як 2/(λ_min + λ_max).
    C = E - τ*A,  d = τ*f.
    """
    n = len(b)
    # Оптимальний τ: 2 / (λ_min + λ_max) через власні значення
    eigvals = np.linalg.eigvalsh(A)
    lam_min, lam_max = eigvals.min(), eigvals.max()
    tau = 2.0 / (lam_min + lam_max)

    C = np.eye(n) - tau * A
    d = tau * b
    print(f"\n  τ = {tau:.6e},  ||C||∞ = {mat_norm(C):.6f}")

    x = x0.copy()
    hist_err, hist_res = [], []

    for k in range(1, max_iter + 1):
        x_new = mat_vec(C, x) + d          # X^(k+1) = C·X^(k) + d

        err = vec_norm(x_new - x)
        res = vec_norm(mat_vec(A, x_new) - b)
        true_err = vec_norm(x_new - x_true)

        hist_err.append(err)
        hist_res.append(res)

        x = x_new

        # Зупинка: будь-яка з умов виконується (методичка: "або")
        if err <= eps or res <= eps:
            print(f"  ✓ Збіжність на ітерації {k}: "
                  f"||ΔX||∞={err:.2e}, ||AX-b||∞={res:.2e}, "
                  f"||X-X_true||∞={true_err:.2e}")
            break
    else:
        print(f"  ⚠ Досягнуто MAX_ITER={max_iter}: "
              f"||ΔX||∞={err:.2e}, ||AX-b||∞={res:.2e}")

    return x, k, hist_err, hist_res

# ─────────────────────────────────────────────────────────────────
# МЕТОД ЯКОБІ
# ─────────────────────────────────────────────────────────────────
def jacobi(A, b, x0, eps=1e-14, max_iter=10_000):
    """
    Метод Якобі:
    x_i^(k+1) = (f_i - Σ_{j≠i} a_ij * x_j^(k)) / a_ii
    """
    n = len(b)
    D_inv = 1.0 / np.diag(A)
    C = np.eye(n) - D_inv[:, None] * A
    d = D_inv * b

    x = x0.copy()
    hist_err, hist_res = [], []

    for k in range(1, max_iter + 1):
        x_new = mat_vec(C, x) + d

        err = vec_norm(x_new - x)
        res = vec_norm(mat_vec(A, x_new) - b)
        true_err = vec_norm(x_new - x_true)

        hist_err.append(err)
        hist_res.append(res)
        x = x_new

        if err <= eps or res <= eps:
            print(f"  ✓ Збіжність на ітерації {k}: "
                  f"||ΔX||∞={err:.2e}, ||AX-b||∞={res:.2e}, "
                  f"||X-X_true||∞={true_err:.2e}")
            break
    else:
        print(f"  ⚠ Досягнуто MAX_ITER={max_iter}: "
              f"||ΔX||∞={err:.2e}, ||AX-b||∞={res:.2e}")

    return x, k, hist_err, hist_res

# ─────────────────────────────────────────────────────────────────
# МЕТОД ГАУСА-ЗЕЙДЕЛЯ (векторизований)
# ─────────────────────────────────────────────────────────────────
def seidel(A, b, x0, eps=1e-14, max_iter=10_000):
    """
    Метод Гауса-Зейделя:
    x_i^(k+1) = (f_i - Σ_{j<i} a_ij*x_j^(k+1) - Σ_{j>i} a_ij*x_j^(k)) / a_ii
    """
    n = len(b)
    # Розкладаємо A = L_low + D + U_up один раз
    D_inv = 1.0 / np.diag(A)
    L_low = np.tril(A, -1)   # нижня трикутна (без діагоналі)
    U_up  = np.triu(A,  1)   # верхня трикутна (без діагоналі)

    x = x0.copy()
    hist_err, hist_res = [], []

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        # Рядок i: x_i = (b_i - Σ_{j<i}a_ij*x_j^new - Σ_{j>i}a_ij*x_j^old) / a_ii
        for i in range(n):
            x[i] = D_inv[i] * (b[i]
                                - L_low[i, :i] @ x[:i]
                                - U_up[i, i+1:] @ x_old[i+1:])

        err = vec_norm(x - x_old)
        res = vec_norm(mat_vec(A, x) - b)
        true_err = vec_norm(x - x_true)

        hist_err.append(err)
        hist_res.append(res)

        if err <= eps or res <= eps:
            print(f"  ✓ Збіжність на ітерації {k}: "
                  f"||ΔX||∞={err:.2e}, ||AX-b||∞={res:.2e}, "
                  f"||X-X_true||∞={true_err:.2e}")
            break
    else:
        print(f"  ⚠ Досягнуто MAX_ITER={max_iter}: "
              f"||ΔX||∞={err:.2e}, ||AX-b||∞={res:.2e}")

    return x, k, hist_err, hist_res

# ─────────────────────────────────────────────────────────────────
# КРОК 4 – Запуск всіх методів
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("КРОК 4: Розв'язок СЛАР ітераційними методами (eps = 1e-14)")
print("=" * 65)

results = {}

print("\n── Метод простої ітерації ──────────────────────────────────")
t0 = time.perf_counter()
x_si, k_si, he_si, hr_si = simple_iteration(A, B, X0, EPS_TARGET, MAX_ITER)
t_si = time.perf_counter() - t0
results["Проста ітерація"] = dict(x=x_si, iters=k_si, time=t_si,
                                   hist_err=he_si, hist_res=hr_si)
print(f"  Час: {t_si:.3f} с  |  Ітерацій: {k_si}")

print("\n── Метод Якобі ─────────────────────────────────────────────")
t0 = time.perf_counter()
x_jac, k_jac, he_jac, hr_jac = jacobi(A, B, X0, EPS_TARGET, MAX_ITER)
t_jac = time.perf_counter() - t0
results["Якобі"] = dict(x=x_jac, iters=k_jac, time=t_jac,
                         hist_err=he_jac, hist_res=hr_jac)
print(f"  Час: {t_jac:.3f} с  |  Ітерацій: {k_jac}")

print("\n── Метод Гауса-Зейделя ─────────────────────────────────────")
t0 = time.perf_counter()
x_sei, k_sei, he_sei, hr_sei = seidel(A, B, X0, EPS_TARGET, MAX_ITER)
t_sei = time.perf_counter() - t0
results["Зейдель"] = dict(x=x_sei, iters=k_sei, time=t_sei,
                           hist_err=he_sei, hist_res=hr_sei)
print(f"  Час: {t_sei:.3f} с  |  Ітерацій: {k_sei}")

# ─────────────────────────────────────────────────────────────────
# ПІДСУМКОВА ТАБЛИЦЯ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ПІДСУМОК")
print("=" * 65)
header = f"{'Метод':<22} {'Ітерацій':>10} {'Час, с':>9} "  \
         f"{'||AX-b||∞':>13} {'||X-Xtrue||∞':>14}"
print(header)
print("-" * 70)
for name, r in results.items():
    res = vec_norm(mat_vec(A, r["x"]) - B)
    err = vec_norm(r["x"] - x_true)
    print(f"{name:<22} {r['iters']:>10} {r['time']:>9.3f} "
          f"{res:>13.4e} {err:>14.4e}")
print("=" * 65)

# Матриця C методу Якобі та її норми
D_inv = 1.0 / np.diag(A)
C_jac = np.eye(N) - D_inv[:, None] * A
print(f"\nНорми матриці C (Якобі):")
print(f"  ||C||∞  = {mat_norm(C_jac):.6f}  (має бути < 1)")
print(f"  ||C||_1 = {np.max(np.sum(np.abs(C_jac), axis=0)):.6f}")
print(f"  ||C||_F = {np.linalg.norm(C_jac, 'fro'):.6f}")

# ─────────────────────────────────────────────────────────────────
# ГРАФІКИ
# ─────────────────────────────────────────────────────────────────
print("\nПобудова графіків...")

colors  = {"Проста ітерація": "#e74c3c",
           "Якобі":           "#2980b9",
           "Зейдель":         "#27ae60"}
markers = {"Проста ітерація": "o", "Якобі": "s", "Зейдель": "^"}
lim_plot = 200   # показуємо перші 200 ітерацій на детальних графіках

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Лабораторна робота №8 — Ітераційні методи розв'язку СЛАР",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.33)

# ── 1. Теплова карта матриці A ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(A, cmap="RdBu", aspect="auto")
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title("Матриця A (діаг. переважання)", fontsize=11, fontweight="bold")
ax1.set_xlabel("j"); ax1.set_ylabel("i")

# ── 2. Початкове наближення vs точний розв'язок ───────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(np.arange(N), X0, "k.", markersize=3, label="X₀ = 1/(i+1)")
ax2.axhline(TRUE_X, color="red", linewidth=1.5, linestyle="--",
            label=f"X_true = {TRUE_X}")
ax2.set_title("Початкове наближення", fontsize=11, fontweight="bold")
ax2.set_xlabel("i"); ax2.set_ylabel("xᵢ⁽⁰⁾")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)

# ── 3. Гістограма діагональних vs позадіагональних елементів ──────
ax3 = fig.add_subplot(gs[0, 2])
diag_vals = np.abs(np.diag(A))
offdiag_sum = np.sum(np.abs(A), axis=1) - diag_vals
idx50 = np.arange(50)
ax3.bar(idx50 - 0.2, diag_vals[:50], 0.4,
        color="#2ecc71", label="|a_ii|")
ax3.bar(idx50 + 0.2, offdiag_sum[:50], 0.4,
        color="#e74c3c", alpha=0.7, label="Σ|a_ij|, j≠i")
ax3.set_title("Діагональне переважання (перші 50)", fontsize=11, fontweight="bold")
ax3.set_xlabel("i"); ax3.set_ylabel("Значення")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

# ── 4. Збіжність ||ΔX||∞ (всі ітерації) ─────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for name, r in results.items():
    ax4.semilogy(range(1, len(r["hist_err"]) + 1), r["hist_err"],
                 color=colors[name], linewidth=1.5, label=name)
ax4.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1.5,
            label=f"eps={EPS_TARGET:.0e}")
ax4.set_title("Збіжність ||X^(k+1)−X^(k)||∞", fontsize=11, fontweight="bold")
ax4.set_xlabel("Ітерація"); ax4.set_ylabel("Норма (log)")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.4)

# ── 5. Збіжність ||AX−b||∞ (всі ітерації) ────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
for name, r in results.items():
    ax5.semilogy(range(1, len(r["hist_res"]) + 1), r["hist_res"],
                 color=colors[name], linewidth=1.5, label=name)
ax5.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1.5,
            label=f"eps={EPS_TARGET:.0e}")
ax5.set_title("Збіжність нев'язки ||AX−b||∞", fontsize=11, fontweight="bold")
ax5.set_xlabel("Ітерація"); ax5.set_ylabel("Норма (log)")
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.4)

# ── 6. Перші lim_plot ітерацій (деталь) ──────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
for name, r in results.items():
    data = r["hist_err"][:lim_plot]
    ax6.semilogy(range(1, len(data) + 1), data,
                 color=colors[name], linewidth=2, label=name,
                 marker=markers[name], markevery=max(1, len(data)//10),
                 markersize=5)
ax6.set_title(f"Збіжність ||ΔX||∞ (перші {lim_plot} ітер.)",
              fontsize=11, fontweight="bold")
ax6.set_xlabel("Ітерація"); ax6.set_ylabel("Норма (log)")
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.4)

# ── 7. Компоненти фінальних розв'язків ───────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
idx = np.arange(N)
for name, r in results.items():
    ax7.plot(idx, r["x"], color=colors[name], linewidth=1,
             alpha=0.8, label=name)
ax7.axhline(TRUE_X, color="k", linestyle="--", linewidth=1.5,
            label=f"X_true={TRUE_X}")
ax7.set_title("Компоненти розв'язку xᵢ", fontsize=11, fontweight="bold")
ax7.set_xlabel("i"); ax7.set_ylabel("xᵢ")
ax7.legend(fontsize=9); ax7.grid(True, alpha=0.4)

# ── 8. Вектор нев'язки AX−b кожного методу ───────────────────────
ax8 = fig.add_subplot(gs[2, 1])
w = 0.25
for k_idx, (name, r) in enumerate(results.items()):
    res_vec = mat_vec(A, r["x"]) - B
    ax8.bar(idx + (k_idx - 1) * w, res_vec, w,
            color=colors[name], alpha=0.8, label=name)
ax8.set_title("Нев'язка (AX−b)ᵢ", fontsize=11, fontweight="bold")
ax8.set_xlabel("i"); ax8.set_ylabel("(AX−b)ᵢ")
ax8.legend(fontsize=9); ax8.grid(True, alpha=0.3)

# ── 9. Порівняльна діаграма: ітерацій та часу ────────────────────
ax9 = fig.add_subplot(gs[2, 2])
names  = list(results.keys())
iters  = [results[n]["iters"] for n in names]
times  = [results[n]["time"]  for n in names]
bar_colors = [colors[n] for n in names]
x_pos = np.arange(len(names))

bars = ax9.bar(x_pos, iters, color=bar_colors, alpha=0.85, width=0.4)
for bar, val in zip(bars, iters):
    ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax9.set_xticks(x_pos); ax9.set_xticklabels(names, fontsize=10)
ax9.set_title("Кількість ітерацій", fontsize=11, fontweight="bold")
ax9.set_ylabel("Ітерацій"); ax9.grid(True, alpha=0.3, axis="y")

# Час як текст поверх
ax9b = ax9.twinx()
ax9b.plot(x_pos, times, "kD--", linewidth=1.5, markersize=7, label="Час, с")
ax9b.set_ylabel("Час, с")
ax9b.legend(loc="upper right", fontsize=9)

plt.savefig(os.path.join(OUT, "lab8_results.png"),
            dpi=150, bbox_inches="tight")
print("Графіки збережені → lab8_results.png")
plt.show()