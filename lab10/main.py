"""
Лабораторна робота №10
Чисельне розв'язання задачі Коші для ЗДР першого порядку

Рівняння: y' = -y + x + 1,  y(0) = 1,  x ∈ [0, 1]
Точний розв'язок: y(x) = e^(-x) + x
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# Параметри задачі
# ──────────────────────────────────────────────────────────────────────────────
def f(x, y):
    """Права частина ОДР: y' = -y + x + 1"""
    return -y + x + 1

def exact(x):
    """Точний розв'язок: y(x) = e^(-x) + x"""
    return np.exp(-x) + x

X0, X1 = 0.0, 1.0   # відрізок
Y0      = 1.0        # початкова умова
H       = 0.1        # крок за замовчуванням
EPS     = 1e-5       # задана точність
C       = 0.8        # константа для автовибору кроку (0 < C < 1)

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 2 – Метод Рунге-Кутта 4-го порядку
# ══════════════════════════════════════════════════════════════════════════════

def rk4_step(f, x, y, h):
    """Один крок методу Рунге-Кутта 4-го порядку."""
    k1 = h * f(x,           y)
    k2 = h * f(x + h/2,     y + k1/2)
    k3 = h * f(x + h/2,     y + k2/2)
    k4 = h * f(x + h,       y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


def rk4_solve(f, x0, x1, y0, h):
    """RK4 на рівномірній сітці; повертає (xs, ys)."""
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    while x + h <= x1 + 1e-12:
        y = rk4_step(f, x, y, h)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def rk4_local_error_exact(f, x0, x1, y0, h, exact):
    """Локальна похибка RK4 відносно точного розв'язку."""
    xs, ys = rk4_solve(f, x0, x1, y0, h)
    err = np.abs(ys - exact(xs))
    return xs, err


def rk4_runge_error(f, x0, x1, y0, h):
    """
    Оцінка локальної похибки RK4 методом Рунге.
    На кожному вузлі порівнюємо крок h та h/2.
    """
    xs_h,  ys_h  = rk4_solve(f, x0, x1, y0, h)
    xs_h2, ys_h2 = rk4_solve(f, x0, x1, y0, h/2)

    # вузли сітки h присутні і в сітці h/2 (кожен другий)
    err = []
    for i in range(len(xs_h)):
        j = 2 * i          # відповідний індекс у сітці h/2
        runge = abs(ys_h2[j] - ys_h[i]) / (2**4 - 1)
        err.append(runge)
    return xs_h, np.array(err)


def rk4_auto_step(f, x0, x1, y0, eps, h0=None, c=C):
    """
    Автоматичний вибір кроку для RK4 (оцінка похибки методом Рунге).
    Повертає (xs, ys, hs) – координати, значення та використані кроки.
    """
    if h0 is None:
        h0 = (x1 - x0) / 10
    x, y, h = x0, y0, h0
    xs, ys, hs = [x], [y], [h]

    while x < x1 - 1e-12:
        h = min(h, x1 - x)
        # один крок з h, два кроки з h/2
        y1_h  = rk4_step(f, x, y, h)
        y_mid = rk4_step(f, x, y, h/2)
        y1_h2 = rk4_step(f, x + h/2, y_mid, h/2)

        runge = abs(y1_h2 - y1_h) / (2**4 - 1)

        if runge > eps:
            h /= 2          # зменшуємо крок
            continue
        else:
            x += h
            y  = y1_h2      # використовуємо точніше значення
            xs.append(x)
            ys.append(y)
            hs.append(h)
            if runge < c * eps:
                h *= 2      # збільшуємо крок
    return np.array(xs), np.array(ys), np.array(hs)


# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 1 – Метод прогнозу та корекції Адамса 2-го порядку
# ══════════════════════════════════════════════════════════════════════════════

def adams2_solve(f, x0, x1, y0, h):
    """
    Метод прогнозу та корекції Адамса 2-го порядку.
    Перший крок – RK4 для ініціалізації.
    Повертає (xs, ys).
    """
    xs = [x0]
    ys = [y0]

    # стартовий крок через RK4
    x1_node = x0 + h
    y1_node = rk4_step(f, x0, y0, h)
    xs.append(x1_node)
    ys.append(y1_node)

    x = x1_node
    y = y1_node
    f_prev = f(x0, y0)
    f_curr = f(x, y)

    while x + h <= x1 + 1e-12:
        # Прогноз (екстраполяція Адамса 2-го порядку):
        # y*_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})
        y_pred = y + h/2 * (3*f_curr - f_prev)
        x_next = x + h
        f_pred = f(x_next, y_pred)

        # Корекція (інтерполяція Адамса 2-го порядку):
        # y_{n+1} = y_n + h/2 * (f_{n+1}* + f_n)
        y_corr = y + h/2 * (f_pred + f_curr)

        # Одна ітерація уточнення
        f_corr = f(x_next, y_corr)
        y_corr = y + h/2 * (f_corr + f_curr)

        f_prev = f_curr
        f_curr = f(x_next, y_corr)
        y = y_corr
        x = x_next
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def adams2_local_error_exact(f, x0, x1, y0, h, exact):
    """Локальна похибка Адамса-2 відносно точного розв'язку."""
    xs, ys = adams2_solve(f, x0, x1, y0, h)
    err = np.abs(ys - exact(xs))
    return xs, err


def adams2_runge_error(f, x0, x1, y0, h):
    """
    Оцінка похибки через різницю прогнозованого та скорегованого значень
    (аналог оцінки Рунге для Адамса).
    """
    xs = [x0]
    errs = [0.0]
    ys = [y0]

    x1_node = x0 + h
    y1_node = rk4_step(f, x0, y0, h)
    xs.append(x1_node)
    ys.append(y1_node)
    errs.append(0.0)

    x = x1_node
    y = y1_node
    f_prev = f(x0, y0)
    f_curr = f(x, y)

    while x + h <= x1 + 1e-12:
        y_pred = y + h/2 * (3*f_curr - f_prev)
        x_next = x + h
        f_pred = f(x_next, y_pred)
        y_corr = y + h/2 * (f_pred + f_curr)

        # оцінка похибки: |y_corr - y_pred| / 3  (коеф. для 2-го пор.)
        err = abs(y_corr - y_pred) / 3

        f_corr = f(x_next, y_corr)
        y_corr = y + h/2 * (f_corr + f_curr)

        f_prev = f_curr
        f_curr = f(x_next, y_corr)
        y = y_corr
        x = x_next
        xs.append(x)
        ys.append(y)
        errs.append(err)

    return np.array(xs), np.array(errs)


def adams2_auto_step(f, x0, x1, y0, eps, h0=None, c=C):
    """
    Автоматичний вибір кроку для Адамса-2.
    Якщо |y_corr - y_pred|/3 > eps  → h /= 2
    Якщо |y_corr - y_pred|/3 < c*eps → h *= 2
    При зміні кроку перезапускаємо ініціалізацію RK4.
    """
    if h0 is None:
        h0 = (x1 - x0) / 10

    h = h0
    x, y = x0, y0
    xs, ys, hs = [x], [y], [h]

    while x < x1 - 1e-12:
        h = min(h, x1 - x)

        # ініціалізація через RK4
        y1 = rk4_step(f, x, y, h)
        x1_node = x + h

        f_prev = f(x, y)
        f_curr = f(x1_node, y1)

        h2 = min(h, x1 - x1_node)
        if h2 <= 0:
            xs.append(x1_node); ys.append(y1); hs.append(h)
            break

        y_pred = y1 + h2/2 * (3*f_curr - f_prev)
        x_next = x1_node + h2
        f_pred = f(x_next, y_pred)
        y_corr = y1 + h2/2 * (f_pred + f_curr)
        err = abs(y_corr - y_pred) / 3

        if err > eps:
            h /= 2
            continue

        # приймаємо перший внутрішній вузол
        xs.append(x1_node); ys.append(y1); hs.append(h)

        # уточнення та прийняття другого вузла
        f_corr = f(x_next, y_corr)
        y_corr = y1 + h2/2 * (f_corr + f_curr)
        xs.append(x_next); ys.append(y_corr); hs.append(h2)
        x, y = x_next, y_corr

        if err < c * eps:
            h *= 2

    return np.array(xs), np.array(ys), np.array(hs)


# ══════════════════════════════════════════════════════════════════════════════
# Побудова графіків
# ══════════════════════════════════════════════════════════════════════════════

def make_plots():
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        "Лабораторна робота №10\n"
        r"$y' = -y + x + 1,\quad y(0)=1,\quad y(x)=e^{-x}+x,\quad x\in[0,1]$",
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.35)

    x_fine = np.linspace(X0, X1, 500)

    # ── Графік 0: Порівняння розв'язків ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    xs_rk,  ys_rk  = rk4_solve(f, X0, X1, Y0, H)
    xs_ad,  ys_ad  = adams2_solve(f, X0, X1, Y0, H)
    ax0.plot(x_fine, exact(x_fine), 'k-',  lw=2,   label='Точний розв\'язок')
    ax0.plot(xs_rk,  ys_rk,  'bo--', ms=5, lw=1.2, label=f'RK4 (h={H})')
    ax0.plot(xs_ad,  ys_ad,  'rs:',  ms=5, lw=1.2, label=f'Адамс-2 (h={H})')
    ax0.set_title('Порівняння чисельних розв\'язків з точним')
    ax0.set_xlabel('x'); ax0.set_ylabel('y(x)')
    ax0.legend(); ax0.grid(True, alpha=0.4)

    # ── Графік 1: Похибка RK4 (точна) ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    xs_e, err_e = rk4_local_error_exact(f, X0, X1, Y0, H, exact)
    ax1.semilogy(xs_e, err_e + 1e-20, 'b-o', ms=4, lw=1.3)
    ax1.set_title(f'Ч.2 – Похибка RK4 (точна), h={H}')
    ax1.set_xlabel('x'); ax1.set_ylabel('|y_exact – y_num|')
    ax1.grid(True, which='both', alpha=0.4)

    # ── Графік 2: Похибка RK4 (Рунге) ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    xs_r, err_r = rk4_runge_error(f, X0, X1, Y0, H)
    ax2.semilogy(xs_r, err_r + 1e-20, 'g-s', ms=4, lw=1.3)
    ax2.set_title(f'Ч.2 – Оцінка похибки RK4 (Рунге), h={H}')
    ax2.set_xlabel('x'); ax2.set_ylabel('Оцінка Рунге')
    ax2.grid(True, which='both', alpha=0.4)

    # ── Графік 3: Порівняння похибок RK4 для різних кроків ───────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    for hi, col in zip([0.2, 0.1, 0.05, 0.025], ['#e74c3c','#3498db','#2ecc71','#9b59b6']):
        xs_i, err_i = rk4_local_error_exact(f, X0, X1, Y0, hi, exact)
        ax3.semilogy(xs_i, err_i + 1e-20, color=col, marker='.', lw=1.2, label=f'h={hi}')
    ax3.set_title('Ч.2 – Залежність похибки RK4 від кроку')
    ax3.set_xlabel('x'); ax3.set_ylabel('|похибка|')
    ax3.legend(fontsize=8); ax3.grid(True, which='both', alpha=0.4)

    # ── Графік 4: Автовибір кроку RK4 ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    xs_a, ys_a, hs_a = rk4_auto_step(f, X0, X1, Y0, eps=EPS, h0=H)
    ax4.step(xs_a, hs_a, 'm-', where='post', lw=1.4)
    ax4.set_title(f'Ч.2 – Автовибір кроку RK4, ε={EPS}')
    ax4.set_xlabel('x'); ax4.set_ylabel('h(x)')
    ax4.grid(True, alpha=0.4)

    # ── Графік 5: Похибка Адамса-2 (точна) ───────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    xs_ae, err_ae = adams2_local_error_exact(f, X0, X1, Y0, H, exact)
    ax5.semilogy(xs_ae, err_ae + 1e-20, 'r-o', ms=4, lw=1.3, label='Точна похибка')
    xs_ar, err_ar = adams2_runge_error(f, X0, X1, Y0, H)
    ax5.semilogy(xs_ar, err_ar + 1e-20, 'b--s', ms=4, lw=1.2, label='Оцінка похибки')
    ax5.set_title(f'Ч.1 – Похибка Адамса-2, h={H}')
    ax5.set_xlabel('x'); ax5.set_ylabel('|похибка|')
    ax5.legend(fontsize=8); ax5.grid(True, which='both', alpha=0.4)

    # ── Графік 6: Автовибір кроку Адамса-2 ───────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    xs_aa, ys_aa, hs_aa = adams2_auto_step(f, X0, X1, Y0, eps=EPS, h0=H)
    ax6.step(xs_aa, hs_aa, 'c-', where='post', lw=1.4)
    ax6.set_title(f'Ч.1 – Автовибір кроку Адамса-2, ε={EPS}')
    ax6.set_xlabel('x'); ax6.set_ylabel('h(x)')
    ax6.grid(True, alpha=0.4)

    plt.savefig('/mnt/user-data/outputs/lab10_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Графіки збережено: lab10_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# Текстовий звіт у консоль
# ══════════════════════════════════════════════════════════════════════════════

def print_report():
    sep = "=" * 72

    print(sep)
    print("ЛАБОРАТОРНА РОБОТА №10")
    print("Задача Коші:  y' = -y + x + 1,  y(0) = 1")
    print("Точний розв'язок: y(x) = e^(-x) + x,  x ∈ [0, 1]")
    print(sep)

    # ── ЧАСТИНА 1 ─────────────────────────────────────────────────────────────
    print("\n─── ЧАСТИНА 1 – Метод прогнозу та корекції Адамса 2-го порядку ───")
    print(f"{'x':>8} {'y_num':>14} {'y_exact':>14} {'err_exact':>14} {'err_runge':>14}")
    print("-" * 72)

    xs_ad, ys_ad   = adams2_solve(f, X0, X1, Y0, H)
    xs_ar, err_ar  = adams2_runge_error(f, X0, X1, Y0, H)
    y_ex = exact(xs_ad)
    err_ex = np.abs(ys_ad - y_ex)

    for i in range(len(xs_ad)):
        print(f"{xs_ad[i]:8.4f} {ys_ad[i]:14.8f} {y_ex[i]:14.8f} "
              f"{err_ex[i]:14.2e} {err_ar[i]:14.2e}")

    print(f"\nМаксимальна похибка Адамса-2: {err_ex.max():.4e}")
    print(f"Рекомендований крок (ε={EPS}): h ≈ {H * (EPS/err_ex.max())**(1/2):.4e}")

    xs_aa, ys_aa, hs_aa = adams2_auto_step(f, X0, X1, Y0, eps=EPS, h0=H)
    print(f"\nАвтовибір кроку Адамса-2 (ε={EPS}):")
    print(f"  Вузлів: {len(xs_aa)},  h_min={hs_aa.min():.4e},  h_max={hs_aa.max():.4e}")
    err_aa = np.abs(ys_aa - exact(xs_aa))
    print(f"  Максимальна похибка: {err_aa.max():.4e}")

    # ── ЧАСТИНА 2 ─────────────────────────────────────────────────────────────
    print(f"\n─── ЧАСТИНА 2 – Метод Рунге-Кутта 4-го порядку ───")
    print(f"{'x':>8} {'y_num':>14} {'y_exact':>14} {'err_exact':>14} {'err_runge':>14}")
    print("-" * 72)

    xs_rk,  ys_rk   = rk4_solve(f, X0, X1, Y0, H)
    xs_rr,  err_rr  = rk4_runge_error(f, X0, X1, Y0, H)
    y_ex_rk = exact(xs_rk)
    err_ex_rk = np.abs(ys_rk - y_ex_rk)

    for i in range(len(xs_rk)):
        print(f"{xs_rk[i]:8.4f} {ys_rk[i]:14.8f} {y_ex_rk[i]:14.8f} "
              f"{err_ex_rk[i]:14.2e} {err_rr[i]:14.2e}")

    print(f"\nМаксимальна похибка RK4: {err_ex_rk.max():.4e}")
    print(f"Рекомендований крок (ε={EPS}): h ≈ {H * (EPS/max(err_ex_rk.max(),1e-20))**(1/4):.4e}")

    print(f"\nДослідження залежності похибки RK4 від кроку:")
    print(f"{'h':>10} {'max_err':>14} {'порядок':>10}")
    prev_err = None
    prev_h = None
    for hi in [0.2, 0.1, 0.05, 0.025]:
        _, err_i = rk4_local_error_exact(f, X0, X1, Y0, hi, exact)
        me = err_i.max()
        if prev_err is not None:
            order = np.log(prev_err / me) / np.log(prev_h / hi)
            print(f"{hi:10.4f} {me:14.4e} {order:10.2f}")
        else:
            print(f"{hi:10.4f} {me:14.4e} {'—':>10}")
        prev_err, prev_h = me, hi

    xs_ra, ys_ra, hs_ra = rk4_auto_step(f, X0, X1, Y0, eps=EPS, h0=H)
    print(f"\nАвтовибір кроку RK4 (ε={EPS}):")
    print(f"  Вузлів: {len(xs_ra)},  h_min={hs_ra.min():.4e},  h_max={hs_ra.max():.4e}")
    err_ra = np.abs(ys_ra - exact(xs_ra))
    print(f"  Максимальна похибка: {err_ra.max():.4e}")

    print(f"\n{sep}")
    print("Графіки збережені у файл lab10_plots.png")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Точка входу
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_report()
    make_plots()