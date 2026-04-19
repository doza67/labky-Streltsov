"""
Лабораторна робота №7
LU-розклад. Ітераційні методи уточнення розв'язку СЛАР.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────────────────────────
# ПАРАМЕТРИ
# ─────────────────────────────────────────────────────────────────
N = 100
TRUE_X = 2.5            # всі xi = 2.5
EPS_TARGET = 1e-14
SEED = 42
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────
# КРОК 1 – Генерація матриці A та вектора B
# ─────────────────────────────────────────────────────────────────
def generate_diag_dominant_matrix(n, seed=42):
    """Генеруємо діагонально-домінантну матрицю (гарантує LU без pivoting)."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(-10, 10, (n, n))
    # Діагональне домінування
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + rng.uniform(1, 5)
    return A

def save_matrix(filename, M):
    np.savetxt(filename, M, fmt="%.10f")

def load_matrix(filename):
    return np.loadtxt(filename)

def save_vector(filename, v):
    np.savetxt(filename, v, fmt="%.10f")

def load_vector(filename):
    return np.loadtxt(filename)

print("=" * 60)
print("КРОК 1: Генерація матриці A та вектора B")
print("=" * 60)

A = generate_diag_dominant_matrix(N, SEED)
x_true = np.full(N, TRUE_X)
B = A @ x_true       # b_i = Σ a_ij * x_j

save_matrix(os.path.join(OUTPUT_DIR, "matrix_A.txt"), A)
save_vector(os.path.join(OUTPUT_DIR, "vector_B.txt"), B)

print(f"Матриця A ({N}×{N}) згенерована і збережена → matrix_A.txt")
print(f"Вектор B ({N}×1) обчислений і збережений   → vector_B.txt")
print(f"Точний розв'язок: всі x_i = {TRUE_X}")

# ─────────────────────────────────────────────────────────────────
# КРОК 2 – LU-розклад (власна реалізація)
# ─────────────────────────────────────────────────────────────────
def lu_decompose(A):
    """
    LU-розклад матриці A = L·U
    L – нижня трикутна, U – верхня трикутна з одиницями на діагоналі.
    Повертає L, U.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    np.fill_diagonal(U, 1.0)          # u_ii = 1

    for k in range(n):
        # k-й стовпець L
        for i in range(k, n):
            s = sum(L[i, j] * U[j, k] for j in range(k))
            L[i, k] = A[i, k] - s

        # k-й рядок U
        for i in range(k + 1, n):
            s = sum(L[k, j] * U[j, i] for j in range(k))
            U[k, i] = (A[k, i] - s) / L[k, k]

    return L, U

def solve_lu(L, U, b):
    """Розв'язок LZ=b, потім UX=Z."""
    n = len(b)
    # Пряма підстановка: LZ = b
    Z = np.zeros(n)
    Z[0] = b[0] / L[0, 0]
    for k in range(1, n):
        s = sum(L[k, j] * Z[j] for j in range(k))
        Z[k] = (b[k] - s) / L[k, k]

    # Зворотна підстановка: UX = Z
    X = np.zeros(n)
    X[n - 1] = Z[n - 1]
    for k in range(n - 2, -1, -1):
        s = sum(U[k, j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - s

    return X

def mat_vec(A, x):
    """Добуток матриці на вектор."""
    return A @ x

def vec_norm(v):
    """Нескінченна норма вектора (max|v_i|)."""
    return np.max(np.abs(v))

def residual_norm(A, x, b):
    """Норма нев'язки max|Σ a_ij*x_j - b_i|."""
    return vec_norm(mat_vec(A, x) - b)

# ─────────────────────────────────────────────────────────────────
# КРОК 3 – Розв'язок СЛАР через LU
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 2-3: LU-розклад та розв'язок системи")
print("=" * 60)

# Використовуємо vectorized версію для N=100 (власний цикл надто повільний)
# Але зберігаємо структуру відповідно до методички
def lu_decompose_fast(A):
    """Векторизована LU без pivoting."""
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)
    for k in range(n):
        L[k:, k] = A[k:, k] - L[k:, :k] @ U[:k, k]
        if k + 1 < n:
            U[k, k+1:] = (A[k, k+1:] - L[k, :k] @ U[:k, k+1:]) / L[k, k]
    return L, U

def solve_lu_fast(L, U, b):
    n = len(b)
    Z = np.zeros(n)
    Z[0] = b[0] / L[0, 0]
    for k in range(1, n):
        Z[k] = (b[k] - L[k, :k] @ Z[:k]) / L[k, k]
    X = np.zeros(n)
    X[n-1] = Z[n-1]
    for k in range(n-2, -1, -1):
        X[k] = Z[k] - U[k, k+1:] @ X[k+1:]
    return X

import time

t0 = time.time()
L, U = lu_decompose_fast(A)
t_lu = time.time() - t0
print(f"LU-розклад виконано за {t_lu:.4f} с")

# Перевірка: ||A - L·U|| ≤ ε
lu_err = vec_norm(A - L @ U)
print(f"Перевірка ||A - L·U||∞ = {lu_err:.2e}")

# Збереження LU
save_matrix(os.path.join(OUTPUT_DIR, "matrix_L.txt"), L)
save_matrix(os.path.join(OUTPUT_DIR, "matrix_U.txt"), U)
print("L та U збережені → matrix_L.txt, matrix_U.txt")

t0 = time.time()
X0 = solve_lu_fast(L, U, B)
t_solve = time.time() - t0
print(f"\nРозв'язок системи AX=B знайдено за {t_solve:.4f} с")

# ─────────────────────────────────────────────────────────────────
# КРОК 4 – Оцінка точності
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 4: Оцінка точності розв'язку")
print("=" * 60)

eps_lu = residual_norm(A, X0, B)
err_x_lu = vec_norm(X0 - x_true)

print(f"Норма нев'язки ||AX - B||∞        = {eps_lu:.6e}")
print(f"Похибка розв'язку ||X - X_true||∞ = {err_x_lu:.6e}")
print(f"Перші 5 компонент X0: {X0[:5]}")

# ─────────────────────────────────────────────────────────────────
# КРОК 5 – Ітераційне уточнення
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 5: Ітераційне уточнення розв'язку")
print("=" * 60)

def iterative_refinement(A, L, U, b, x0, eps=1e-14, max_iter=20):
    """
    Ітераційне уточнення розв'язку.
    Зупинка: ||ΔX||∞ ≤ eps  АБО  стагнація (поліпшення < 0.1%).
    Повертає: уточнений розв'язок, кількість ітерацій,
              історію норми похибки та нев'язки.
    """
    x = x0.copy()
    history_res = []
    history_err = []
    prev_err = None

    for it in range(1, max_iter + 1):
        R = b - mat_vec(A, x)
        res_norm = vec_norm(R)
        err_norm = vec_norm(x - x_true)

        history_res.append(res_norm)
        history_err.append(err_norm)

        print(f"  Ітерація {it:2d}: ||R||∞ = {res_norm:.4e},  "
              f"||X−X_true||∞ = {err_norm:.4e}")

        # Умова 1: досягнуто цільової точності розв'язку
        if err_norm <= eps:
            print(f"\n  ✓ Точність ||X−X_true||∞ ≤ {eps:.0e} "
                  f"досягнута на ітерації {it}")
            break

        # Умова 2: стагнація (поліпшення < 1%)
        if prev_err is not None and prev_err > 0:
            improvement = abs(prev_err - err_norm) / prev_err
            if improvement < 0.01:
                print(f"\n  ⚠ Стагнація (поліпшення < 1%) на ітерації {it}.")
                print(f"  Машинна точність досягнута: ||R||∞ ≈ {res_norm:.2e}")
                break

        prev_err = err_norm
        dX = solve_lu_fast(L, U, R)
        x = x + dX

    return x, it, history_res, history_err

X_refined, n_iters, hist_res, hist_err = iterative_refinement(
    A, L, U, B, X0, eps=EPS_TARGET
)

# Також демонструємо уточнення з навмисно зашумленим початковим розв'язком
print("\n--- Демонстрація з пертурбованим початковим розв'язком ---")
rng_perturb = np.random.default_rng(99)
X0_perturbed = X0 + rng_perturb.uniform(-0.1, 0.1, N)   # додаємо шум ±0.1
X_refined2, n_iters2, hist_res2, hist_err2 = iterative_refinement(
    A, L, U, B, X0_perturbed, eps=EPS_TARGET
)
print(f"З пертурбацією ±0.1: збіжність за {n_iters2} ітерацій")

eps_final = residual_norm(A, X_refined, B)
err_final = vec_norm(X_refined - x_true)

print(f"\nФінальна норма нев'язки ||AX - B||∞       = {eps_final:.4e}")
print(f"Фінальна похибка ||X_refined - X_true||∞  = {err_final:.4e}")
print(f"Кількість ітерацій: {n_iters}")
print(f"Перші 5 компонент X_refined: {X_refined[:5]}")

# ─────────────────────────────────────────────────────────────────
# ГРАФІКИ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Побудова графіків...")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Лабораторна робота №7 — LU-розклад і ітераційне уточнення СЛАР",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

iters_range = np.arange(1, len(hist_res2) + 1)

# ── 1. Теплова карта матриці A (лівий верх) ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(A, cmap="RdBu", aspect="auto")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title("Матриця A", fontsize=11, fontweight="bold")
ax1.set_xlabel("j"); ax1.set_ylabel("i")

# ── 2. Теплова карта матриці L ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(L, cmap="YlOrRd", aspect="auto")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title("Нижня трикутна матриця L", fontsize=11, fontweight="bold")
ax2.set_xlabel("j"); ax2.set_ylabel("i")

# ── 3. Теплова карта матриці U ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(U, cmap="Blues", aspect="auto")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title("Верхня трикутна матриця U", fontsize=11, fontweight="bold")
ax3.set_xlabel("j"); ax3.set_ylabel("i")

# ── 4. Норма нев'язки по ітераціях (пертурб.) ────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.semilogy(iters_range, hist_res2, "ro-", linewidth=2,
             markersize=6, label="||R||∞ (з пертурб.)")
ax4.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1.5,
            label=f"ціль {EPS_TARGET:.0e}")
ax4.set_title("Збіжність нев'язки", fontsize=11, fontweight="bold")
ax4.set_xlabel("Ітерація"); ax4.set_ylabel("||AX − B||∞ (log)")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.4)
ax4.set_xticks(iters_range)

# ── 5. Похибка по ітераціях (пертурб.) ───────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.semilogy(iters_range, hist_err2, "bs-", linewidth=2,
             markersize=6, label="||ΔX||∞ (з пертурб.)")
ax5.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1.5,
            label=f"ціль {EPS_TARGET:.0e}")
ax5.set_title("Збіжність похибки розв'язку", fontsize=11, fontweight="bold")
ax5.set_xlabel("Ітерація"); ax5.set_ylabel("||X − X_true||∞ (log)")
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.4)
ax5.set_xticks(iters_range)

# ── 6. Порівняння: LU vs LU+Ітерація ─────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.semilogy(iters_range, hist_res2, "ro-", linewidth=2,
             markersize=5, label="Нев'язка ||R||∞")
ax6.semilogy(iters_range, hist_err2, "bs--", linewidth=2,
             markersize=5, label="Похибка ||ΔX||∞")
ax6.axhline(EPS_TARGET, color="gray", linestyle=":", linewidth=1.5,
            label=f"eps = {EPS_TARGET:.0e}")
ax6.set_title("Нев'язка vs Похибка (пертурб.)", fontsize=11, fontweight="bold")
ax6.set_xlabel("Ітерація"); ax6.set_ylabel("Норма (log)")
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.4)
ax6.set_xticks(iters_range)

# ── 7. Компоненти X0 та X_refined ────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
idx = np.arange(N)
ax7.plot(idx, X0_perturbed, "g.", markersize=3, alpha=0.5, label="X₀ (пертурб.)")
ax7.plot(idx, X_refined2, "r.", markersize=3, alpha=0.8, label="X уточнений")
ax7.axhline(TRUE_X, color="k", linestyle="--", linewidth=1.5,
            label=f"X_true = {TRUE_X}")
ax7.set_title("Компоненти розв'язку", fontsize=11, fontweight="bold")
ax7.set_xlabel("i"); ax7.set_ylabel("xᵢ")
ax7.legend(fontsize=9); ax7.grid(True, alpha=0.4)

# ── 8. Вектор нев'язки R = AX0_pert − B ─────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
R_initial = A @ X0_perturbed - B
ax8.bar(np.arange(N), R_initial, color="steelblue", alpha=0.7)
ax8.set_title("Нев'язка після LU+пертурбації", fontsize=11, fontweight="bold")
ax8.set_xlabel("i"); ax8.set_ylabel("(AX₀ − B)ᵢ")
ax8.grid(True, alpha=0.4)

# ── 9. Вектор похибки X_refined − X_true ─────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
err_vec = X_refined2 - x_true
ax9.bar(np.arange(N), err_vec, color="tomato", alpha=0.7)
ax9.set_title("Похибка уточненого розв'язку", fontsize=11, fontweight="bold")
ax9.set_xlabel("i"); ax9.set_ylabel("(X_refined − X_true)ᵢ")
ax9.grid(True, alpha=0.4)

plt.savefig(os.path.join(OUTPUT_DIR, "lab7_results.png"),
            dpi=150, bbox_inches="tight")
print("Графіки збережені → lab7_results.png")

# ─────────────────────────────────────────────────────────────────
# ПІДСУМОК
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ПІДСУМОК")
print("=" * 60)
print(f"{'Розмір матриці A:':<45} {N}×{N}")
label_lu_time   = "Час LU-розкладу:"
label_sol_time  = "Час розв'язку LU:"
label_lu_check  = "Перевірка ||A - L*U||inf:"
label_res_lu    = "Норма нев'язки після LU:"
label_err_lu    = "Похибка після LU:"
label_iters     = "Ітерацій уточнення:"
label_res_ref   = "Норма нев'язки після уточнення:"
label_err_ref   = "Похибка після уточнення:"
print(f"{label_lu_time:<45} {t_lu:.4f} с")
print(f"{label_sol_time:<45} {t_solve:.4f} с")
print(f"{label_lu_check:<45} {lu_err:.2e}")
print(f"{label_res_lu:<45} {eps_lu:.4e}")
print(f"{label_err_lu:<45} {err_x_lu:.4e}")
print(f"{label_iters:<45} {n_iters}")
print(f"{label_res_ref:<45} {eps_final:.4e}")
print(f"{label_err_ref:<45} {err_final:.4e}")
print("=" * 60)

plt.show()