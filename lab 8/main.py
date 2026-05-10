#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ЛАБОРАТОРНА РОБОТА №8                                               ║
║         Чисельні методи розв'язування нелінійних рівнянь                   ║
║                                                                              ║
║  Рівняння:  f(x) = x·sin(x) − 0.6 = 0  на [0, 4]                          ║
║  Алгебраїчне рівняння: P(x) = x³ − 3x² + 5x − 3 = 0                       ║
║     Корені: x₁=1 (дійсний), x₂,₃=1±i√2 (комплексні)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sin, cos, sqrt, pi
import os

# ──────────────────────────────────────────────────────────────────────────────
# ГЛОБАЛЬНІ ПАРАМЕТРИ
# ──────────────────────────────────────────────────────────────────────────────
EPS      = 1e-6       # точність
MAX_ITER = 10_000     # максимум ітерацій
A, B, H  = 0.0, 4.0, 0.1   # інтервал і крок табуляції

POLY          = [1.0, -3.0, 5.0, -3.0]   # x³ − 3x² + 5x − 3
POLY_FILENAME = "coefficients.txt"
TAB_FILENAME  = "tabulation.txt"

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 1: ТРАНСЦЕНДЕНТНЕ РІВНЯННЯ
#   f(x) = x·sin(x) − 0.6 = 0
# ══════════════════════════════════════════════════════════════════════════════

def f(x):
    """Трансцендентна функція."""
    return x * sin(x) - 0.6

def df(x):
    """Перша похідна f′(x) = sin(x) + x·cos(x)."""
    return sin(x) + x * cos(x)

def d2f(x):
    """Друга похідна f″(x) = 2cos(x) − x·sin(x)."""
    return 2.0 * cos(x) - x * sin(x)

def stop(x_new, x_old):
    """Критерій зупинки: |xₙ₊₁ − xₙ| < ε  І  |f(xₙ₊₁)| < ε."""
    return abs(x_new - x_old) < EPS and abs(f(x_new)) < EPS

# ─────────────────────────── Завдання 1: Табуляція ───────────────────────────

def tabulate(a=A, b=B, h=H, filename=TAB_FILENAME):
    """
    Табулює f(x) на [a, b] з кроком h.
    Записує результати у текстовий файл.
    Повертає (xs, ys).
    """
    xs, ys = [], []
    x = a
    while x <= b + 1e-12:
        xs.append(round(x, 10))
        ys.append(f(x))
        x = round(x + h, 10)

    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(f"Табуляція f(x) = x·sin(x) − 0.6  на [{a}, {b}], крок h={h}\n")
        fh.write(f"{'x':>10} | {'f(x)':>18}\n")
        fh.write("─" * 33 + "\n")
        for xi, yi in zip(xs, ys):
            sign = "+" if yi >= 0 else ""
            fh.write(f"{xi:>10.4f} | {sign}{yi:>17.8f}\n")

    return xs, ys


def find_approx_roots(xs, ys):
    """
    Знаходить наближені корені за зміною знаку функції.
    Повертає список наближених значень абсцис.
    """
    roots = []
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            # Лінійна інтерполяція
            r = xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            roots.append(round(r, 8))
    return roots

# ─────────────── Завдання 2: Методи уточнення коренів ───────────────────────

def method_simple_iteration(x0, lam):
    """
    Метод простої ітерації (релаксація).
    φ(x) = x − λ·f(x)
    Збіжність: |φ′(x*)| = |1 − λ·f′(x*)| < 1.
    """
    x = x0
    for n in range(1, MAX_ITER + 1):
        x_new = x - lam * f(x)
        if stop(x_new, x):
            return x_new, n
        x = x_new
    return x, MAX_ITER


def method_newton(x0):
    """
    Метод Ньютона (дотичних).
    x_{n+1} = x_n − f(x_n) / f′(x_n)
    Порядок збіжності: 2.
    """
    x = x0
    for n in range(1, MAX_ITER + 1):
        dv = df(x)
        if abs(dv) < 1e-15:
            break
        x_new = x - f(x) / dv
        if stop(x_new, x):
            return x_new, n
        x = x_new
    return x, MAX_ITER


def method_chebyshev(x0):
    """
    Метод Чебишева.
    x_{n+1} = x_n − f/f′ − f²·f″/(2·f′³)
    Порядок збіжності: 3.
    """
    x = x0
    for n in range(1, MAX_ITER + 1):
        fv, dv, d2v = f(x), df(x), d2f(x)
        if abs(dv) < 1e-15:
            break
        x_new = x - fv / dv - (fv ** 2 * d2v) / (2.0 * dv ** 3)
        if stop(x_new, x):
            return x_new, n
        x = x_new
    return x, MAX_ITER


def method_chord(x0, x1):
    """
    Метод хорд (secant / метод хорди).
    x_{n+1} = x_n − f(x_n)·(x_n − x_{n-1}) / (f(x_n) − f(x_{n-1}))
    Порядок збіжності: ≈1.618 (золотий перетин).
    """
    xp, xc = x0, x1
    for n in range(1, MAX_ITER + 1):
        fp, fc = f(xp), f(xc)
        d = fc - fp
        if abs(d) < 1e-15:
            break
        x_new = xc - fc * (xc - xp) / d
        if stop(x_new, xc):
            return x_new, n
        xp, xc = xc, x_new
    return xc, MAX_ITER


def method_parabola(x0, x1, x2):
    """
    Метод парабол (Мюллера).
    Апроксимує f(x) параболою через три останні точки,
    корінь параболи — нове наближення.
    Порядок збіжності: ≈1.839.
    """
    xa, xb, xc = x0, x1, x2
    for n in range(1, MAX_ITER + 1):
        fa, fb, fc = f(xa), f(xb), f(xc)
        h0 = xb - xa
        h1 = xc - xb
        if abs(h0) < 1e-15 or abs(h1) < 1e-15:
            break
        d0    = (fb - fa) / h0
        d1    = (fc - fb) / h1
        s_sum = h0 + h1
        if abs(s_sum) < 1e-15:
            break
        d2    = (d1 - d0) / s_sum
        b_coef = d1 + h1 * d2
        disc   = b_coef ** 2 - 4.0 * fc * d2
        if disc < 0.0:
            disc = 0.0        # залишаємося в дійсній площині
        sq  = sqrt(disc)
        # вибираємо більший за модулем знаменник
        den = (b_coef + sq) if abs(b_coef + sq) >= abs(b_coef - sq) else (b_coef - sq)
        if abs(den) < 1e-15:
            break
        x_new = xc - 2.0 * fc / den
        if stop(x_new, xc):
            return x_new, n
        xa, xb, xc = xb, xc, x_new
    return xc, MAX_ITER


def method_inverse_interp(x0, x1, x2):
    """
    Метод зворотної інтерполяції (Лагранж).
    Будуємо інтерполяційний многочлен x(F) через точки
    (f(x₀),x₀), (f(x₁),x₁), (f(x₂),x₂) і знаходимо x(0).

    x(0) = x₀·f₁f₂/((f₀−f₁)(f₀−f₂))
          + x₁·f₀f₂/((f₁−f₀)(f₁−f₂))
          + x₂·f₀f₁/((f₂−f₀)(f₂−f₁))
    """
    xa, xb, xc = x0, x1, x2
    for n in range(1, MAX_ITER + 1):
        fa, fb, fc = f(xa), f(xb), f(xc)
        dab = fa - fb
        dac = fa - fc
        dbc = fb - fc
        if abs(dab) < 1e-15 or abs(dac) < 1e-15 or abs(dbc) < 1e-15:
            break
        # Lagrange inverse at F=0
        x_new = (xa * fb * fc / (dab * dac)
               + xb * fa * fc / (-dab * dbc)
               + xc * fa * fb / (dac * dbc))
        if stop(x_new, xc):
            return x_new, n
        xa, xb, xc = xb, xc, x_new
    return xc, MAX_ITER


# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 2: АЛГЕБРАЇЧНЕ РІВНЯННЯ
#   P(x) = x³ − 3x² + 5x − 3 = 0
# ══════════════════════════════════════════════════════════════════════════════

# ─────────── Завдання 6–7: Файл коефіцієнтів + обчислення ───────────────────

def save_coefficients(coeffs, filename=POLY_FILENAME):
    """
    Завдання 6.
    Записує коефіцієнти довільного алгебраїчного рівняння у текстовий файл.
    """
    deg = len(coeffs) - 1
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write("# Коефіцієнти алгебраїчного многочлена (від старшого степеня)\n")
        fh.write(f"# Степінь: {deg}\n")
        fh.write(f"# Рівняння: {''.join(_poly_str(coeffs))}\n")
        for c in coeffs:
            fh.write(f"{c}\n")


def load_coefficients(filename=POLY_FILENAME):
    """
    Завдання 7.
    Зчитує коефіцієнти довільного алгебраїчного многочлена з текстового файлу.
    Повертає список коефіцієнтів [aₙ, aₙ₋₁, …, a₀].
    """
    coeffs = []
    with open(filename, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith('#'):
                coeffs.append(float(s))
    return coeffs


def poly_horner(coeffs, x):
    """
    Завдання 7.
    Обчислює значення довільного алгебраїчного многочлена P(x)
    за схемою Горнера.
    """
    val = coeffs[0]
    for c in coeffs[1:]:
        val = val * x + c
    return val


# ─────────── Завдання 8: Метод Ньютона зі схемою Горнера ─────────────────────

def horner_val_and_deriv(coeffs, x):
    """
    Обчислює P(x) та P′(x) одночасно за схемою Горнера.

    b₀ = aₙ,  bₖ = bₖ₋₁·x + aₙ₋ₖ  →  bₙ = P(x)
    c₀ = bₙ,  cₖ = cₖ₋₁·x + bₙ₋ₖ  →  cₙ₋₁ = P′(x)
    """
    n   = len(coeffs) - 1
    b   = coeffs[0]   # накопичувач P(x)
    c   = coeffs[0]   # накопичувач P′(x)
    for i in range(1, n):
        b = b * x + coeffs[i]
        c = c * x + b
    b = b * x + coeffs[n]
    return b, c     # P(x), P′(x)


def newton_horner(coeffs, x0):
    """
    Завдання 8.
    Метод Ньютона зі схемою Горнера для знаходження дійсного кореня
    алгебраїчного рівняння.
    Повертає (корінь, кількість ітерацій).
    """
    x = x0
    for n in range(1, MAX_ITER + 1):
        px, dpx = horner_val_and_deriv(coeffs, x)
        if abs(dpx) < 1e-15:
            break
        x_new = x - px / dpx
        if abs(x_new - x) < EPS and abs(poly_horner(coeffs, x_new)) < EPS:
            return x_new, n
        x = x_new
    return x, MAX_ITER


# ─────────── Завдання 9: Метод Ліна (Bairstow) ──────────────────────────────

def lin_bairstow(coeffs, p0, q0):
    """
    Завдання 9.
    Метод Ліна (Bairstow) — знаходить квадратний дільник x² + p·x + q
    алгебраїчного рівняння.  Комплексні корені — корені цього дільника.

    Алгоритм:
      1. Ділення P(x) на (x²+p·x+q) за схемою Горнера → b[k], залишок α,β
      2. Ділення b(x) на (x²+p·x+q) → c[k]
      3. Розв'язок лінійної системи Bairstow для поправок Δp, Δq
      4. Ітерація до збіжності

    Повертає (p, q, z₁, z₂, iter_count).
    """
    n = len(coeffs) - 1
    a = list(coeffs)
    p, q = p0, q0

    for itr in range(1, MAX_ITER + 1):

        # ── Крок 1: b[k] = a[k] − p·b[k-1] − q·b[k-2] ──────────────────
        b    = [0.0] * (n + 1)
        b[0] = a[0]
        if n >= 1:
            b[1] = a[1] - p * b[0]
        for k in range(2, n + 1):
            b[k] = a[k] - p * b[k - 1] - q * b[k - 2]

        alpha = b[n - 1]   # коефіцієнт x у залишку
        beta  = b[n]       # вільний член залишку

        if abs(alpha) < EPS and abs(beta) < EPS:
            break

        # ── Крок 2: c[k] = b[k] − p·c[k-1] − q·c[k-2] ──────────────────
        c    = [0.0] * n
        c[0] = b[0]
        if n >= 2:
            c[1] = b[1] - p * c[0]
        for k in range(2, n):
            c[k] = b[k] - p * c[k - 1] - q * c[k - 2]

        # ── Крок 3: Bairstow linear system ───────────────────────────────
        #   c[n-2]·Δp + c[n-3]·Δq = alpha
        #   c[n-1]·Δp + c[n-2]·Δq = beta
        cnm2 = c[n - 2]
        cnm3 = c[n - 3] if n >= 3 else 0.0
        cnm1 = c[n - 1]

        det = cnm2 ** 2 - cnm1 * cnm3
        if abs(det) < 1e-15:
            print(f"    [УВАГА] Вироджений якобіан на ітерації {itr}")
            break

        dp = (alpha * cnm2 - beta  * cnm3) / det
        dq = (beta  * cnm2 - alpha * cnm1) / det
        p += dp
        q += dq

        if abs(dp) < EPS and abs(dq) < EPS:
            break

    # Корені квадратного рівняння x² + p·x + q = 0
    disc = p ** 2 - 4.0 * q
    if disc >= 0.0:
        z1 = (-p + sqrt(disc)) / 2.0
        z2 = (-p - sqrt(disc)) / 2.0
    else:
        re = -p / 2.0
        im = sqrt(-disc) / 2.0
        z1 = complex(re,  im)
        z2 = complex(re, -im)

    return p, q, z1, z2, itr


# ══════════════════════════════════════════════════════════════════════════════
# ДОПОМІЖНІ ФУНКЦІЇ
# ══════════════════════════════════════════════════════════════════════════════

def _poly_str(coeffs):
    """Повертає рядкове представлення многочлена."""
    deg = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        pw = deg - i
        if abs(c) < 1e-14:
            continue
        if c > 0 and parts:
            parts.append(f"+ {c}x^{pw}" if pw > 1 else (f"+ {c}x" if pw == 1 else f"+ {c}"))
        else:
            parts.append(f"{c}x^{pw}" if pw > 1 else (f"{c}x" if pw == 1 else f"{c}"))
    return parts


def _sep(char="═", n=72):
    return char * n


def _hdr(title):
    print(f"\n{_sep()}")
    print(f"  {title}")
    print(_sep())


# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІКИ
# ══════════════════════════════════════════════════════════════════════════════

def plot_transcendental(xs, ys, root1, root2, filename="plot_transcendental.png"):
    """Графік трансцендентної функції."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Лівий: загальний вигляд ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(xs, ys, 'steelblue', lw=2.2, label=r'$f(x)=x\sin x - 0.6$')
    ax.axhline(0, color='k', lw=0.9)
    ax.axvline(0, color='k', lw=0.9)
    for r, clr, lbl in [(root1, '#e74c3c', f'Корінь 1: $x_1≈{root1:.6f}$'),
                        (root2, '#2ecc71', f'Корінь 2: $x_2≈{root2:.6f}$')]:
        ax.axvline(r, color=clr, ls='--', alpha=0.7)
        ax.plot(r, f(r), 'o', color=clr, ms=9, label=lbl)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(r'$f(x)=x\sin x - 0.6 = 0$', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Правий: збіжність різних методів (корінь 1) ──────────────────────
    ax2 = axes[1]
    styles   = ['-o', '-s', '-^', '-D', '-v', '-p']
    colors_m = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
    x0 = root1 - 0.15   # початкове наближення
    iters_data = []

    run_funcs = [
        ("Проста ітерація",   lambda: method_simple_iteration(x0, 0.5)),
        ("Ньютон",            lambda: method_newton(x0)),
        ("Чебишев",           lambda: method_chebyshev(x0)),
    ]
    for label, fn in run_funcs:
        _, n_it = fn()
        iters_data.append((label, n_it))

    bar_names  = [l for l, _ in iters_data]
    bar_values = [v for _, v in iters_data]
    bars = ax2.bar(bar_names, bar_values, color=colors_m[:len(bar_names)], alpha=0.8, edgecolor='k')
    for bar, val in zip(bars, bar_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Кількість ітерацій', fontsize=12)
    ax2.set_title(f'Порівняння методів (корінь 1, ε={EPS})', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Чисельні методи розв\'язування нелінійних рівнянь', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def plot_iteration_comparison(results, filename="plot_iterations.png"):
    """Зведена порівняльна таблиця ітерацій для всіх методів і обох коренів."""
    methods = ['Проста ітерація', 'Ньютон', 'Чебишев', 'Хорди', 'Параболи', 'Зворот. інтерп.']
    n1 = [results['r1'][m][1] for m in methods]
    n2 = [results['r2'][m][1] for m in methods]

    x_pos = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x_pos - width/2, n1, width, label='Корінь 1', color='#3498db', alpha=0.85, edgecolor='k')
    bars2 = ax.bar(x_pos + width/2, n2, width, label='Корінь 2', color='#e74c3c', alpha=0.85, edgecolor='k')

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, str(int(h)),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=18, ha='right', fontsize=10)
    ax.set_ylabel('Кількість ітерацій', fontsize=12)
    ax.set_title(f'Порівняння методів: f(x)=x·sin(x)−0.6=0,  ε={EPS}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def plot_polynomial(coeffs, x_real, z1, z2, filename="plot_polynomial.png"):
    """Графік алгебраїчного многочлена."""
    xs = np.linspace(-0.3, 3.5, 700)
    ys = [poly_horner(coeffs, xi) for xi in xs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(xs, ys, 'steelblue', lw=2.2, label='$P(x) = x^3 - 3x^2 + 5x - 3$')
    ax.axhline(0, color='k', lw=0.9)
    ax.axvline(0, color='k', lw=0.9)
    ax.plot(x_real, poly_horner(coeffs, x_real), 'ro', ms=10,
            label=f'Дійсний корінь: $x_1 = {x_real:.8f}$')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('P(x)', fontsize=12)
    ax.set_title('$P(x)=x^3 - 3x^2 + 5x - 3 = 0$', fontsize=13)
    ax.set_ylim(-6, 12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Комплексна площина
    ax2 = axes[1]
    ax2.axhline(0, color='k', lw=0.9)
    ax2.axvline(0, color='k', lw=0.9)
    ax2.plot(x_real, 0, 'ro', ms=11, zorder=5, label=f'$x_1 = {x_real:.4f}$ (дійсний)')
    if isinstance(z1, complex):
        ax2.plot(z1.real,  z1.imag, 'b^', ms=11, zorder=5,
                 label=f'$z_{{2,3}} = {z1.real:.4f} \\pm {abs(z1.imag):.4f}i$')
        ax2.plot(z2.real,  z2.imag, 'b^', ms=11, zorder=5)
        # Показуємо модуль
        theta = np.linspace(0, 2*pi, 200)
        r_mod = abs(z1)
        ax2.plot(r_mod * np.cos(theta), r_mod * np.sin(theta),
                 'b--', alpha=0.3, label=f'|z| = {r_mod:.4f}')
    ax2.set_xlabel('Re', fontsize=12)
    ax2.set_ylabel('Im', fontsize=12)
    ax2.set_title('Корені в комплексній площині', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.suptitle('Алгебраїчне рівняння: $P(x) = x^3 - 3x^2 + 5x - 3 = 0$',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


# ══════════════════════════════════════════════════════════════════════════════
# ГОЛОВНА ПРОГРАМА
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(_sep())
    print("  ЛАБОРАТОРНА РОБОТА №8")
    print("  Чисельні методи розв'язування нелінійних рівнянь")
    print(f"  Рівняння: f(x) = x·sin(x) − 0.6 = 0   на [{A}, {B}]")
    print(f"  Точність: ε = {EPS}")
    print(_sep())

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 1: ТАБУЛЯЦІЯ
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 1. Табуляція f(x) = x·sin(x) − 0.6  на [0, 4], h = 0.1")

    xs, ys = tabulate()
    print(f"  Результати записано у '{TAB_FILENAME}'  ({len(xs)} вузлів)")

    approx = find_approx_roots(xs, ys)
    print(f"\n  Знайдено наближених коренів: {len(approx)}")
    for i, r in enumerate(approx, 1):
        print(f"    Корінь {i}: x ≈ {r:.6f},  f(x) = {f(r):+.6f}")

    if len(approx) < 2:
        print("  [ПОМИЛКА] Потрібно ≥2 коренів!")
        return

    x0_1 = approx[0]   # ≈ 0.82  (f′ > 0, зростання)
    x0_2 = approx[1]   # ≈ 2.93  (f′ < 0, спадання)

    print(f"\n  Обрано початкові наближення:")
    print(f"    Корінь 1 (f зростає): x₀ = {x0_1:.6f},  f′(x₀) = {df(x0_1):+.4f}")
    print(f"    Корінь 2 (f спадає):  x₀ = {x0_2:.6f},  f′(x₀) = {df(x0_2):+.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 2–4: МЕТОДИ УТОЧНЕННЯ
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 2–4. Методи уточнення коренів  (ε = 1e-6)")

    # Параметри λ для простої ітерації
    # λ = 0.5 (Корінь 1): |φ′(x*)| ≈ |1 − 0.5·1.29| ≈ 0.36 < 1  ✓
    # λ = -0.3 (Корінь 2): |φ′(x*)| ≈ |1 − (-0.3)·(-2.65)| ≈ 0.21 < 1  ✓
    LAM1 =  0.5
    LAM2 = -0.3

    # Три початкові точки для багатокрокових методів
    dx = H
    xa1, xb1, xc1 = x0_1 - dx, x0_1, x0_1 + dx
    xa2, xb2, xc2 = x0_2 - dx, x0_2, x0_2 + dx

    results = {}

    for tag, x0, lam, xA, xB, xC in [
        ('r1', x0_1, LAM1, xa1, xb1, xc1),
        ('r2', x0_2, LAM2, xa2, xb2, xc2),
    ]:
        label = "Корінь 1" if tag == 'r1' else "Корінь 2"
        print(f"\n  ▸ {label}  |  x₀ = {x0:.6f}  |  λ = {lam:.3f}")
        print(f"  {'─'*66}")
        print(f"  {'Метод':<28} {'x*':<20} {'f(x*)':<14} {'iter':>5}")
        print(f"  {'─'*66}")

        r = {}
        for mname, mfn in [
            ('Проста ітерація',  lambda: method_simple_iteration(x0, lam)),
            ('Ньютон',           lambda: method_newton(x0)),
            ('Чебишев',          lambda: method_chebyshev(x0)),
            ('Хорди',            lambda: method_chord(xA, xB)),
            ('Параболи',         lambda: method_parabola(xA, xB, xC)),
            ('Зворот. інтерп.',  lambda: method_inverse_interp(xA, xB, xC)),
        ]:
            xr, n = mfn()
            r[mname] = (xr, n, f(xr))
            print(f"  {mname:<28} {xr:<20.10f} {f(xr):<14.2e} {n:>5}")

        results[tag] = r
        print(f"  {'─'*66}")

    # ── Підсумкова таблиця ───────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  {'Метод':<28} | {'Корінь 1':>10} | {'Корінь 2':>10}")
    print(f"  {'─'*55}")
    for mn in ['Проста ітерація', 'Ньютон', 'Чебишев',
               'Хорди', 'Параболи', 'Зворот. інтерп.']:
        n1 = results['r1'][mn][1]
        n2 = results['r2'][mn][1]
        print(f"  {mn:<28} | {n1:>10} | {n2:>10}")
    print(f"  {'─'*55}")

    # ── Графіки ──────────────────────────────────────────────────────────────
    r1_precise = results['r1']['Ньютон'][0]
    r2_precise = results['r2']['Ньютон'][0]
    f1 = plot_transcendental(xs, ys, r1_precise, r2_precise)
    f2 = plot_iteration_comparison(results)
    print(f"\n  Графіки: '{f1}',  '{f2}'")

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 5: АЛГЕБРАЇЧНЕ РІВНЯННЯ
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 5. Алгебраїчне рівняння: P(x) = x³ − 3x² + 5x − 3 = 0")
    print("  Обгрунтування вибору (одне дійсне + два комплексно-спряжені корені):")
    print("  P(x) = (x − 1)·(x² − 2x + 3)")
    print("  Дискримінант x² − 2x + 3:  D = 4 − 12 = −8 < 0  →  комплексні корені ✓")
    print("  Правило Декарта: 3 зміни знаків у [1, −3, 5, −3] → ≤3 додатних кореня")
    print(f"\n  Перевірка коефіцієнтів (рівняння Вієта):")
    print(f"    x₁ + (z₂ + z₃) = 1 + 2·Re(z) = 1 + 2 = 3 = −a₁/a₀ = 3  ✓")
    print(f"    x₁·z₂·z₃ = 1·|z|² = 1·3 = 3 = −a₃/a₀ = 3  ✓")

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 6: ЗАПИС КОЕФІЦІЄНТІВ
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 6. Запис коефіцієнтів у файл")
    save_coefficients(POLY, POLY_FILENAME)
    print(f"  Коефіцієнти {POLY} записано у '{POLY_FILENAME}'")

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 7: ЗЧИТУВАННЯ + ОБЧИСЛЕННЯ МНОГОЧЛЕНА
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 7. Зчитування коефіцієнтів, обчислення P(x) за схемою Горнера")
    loaded = load_coefficients(POLY_FILENAME)
    print(f"  Зчитані коефіцієнти: {loaded}")
    print(f"\n  Контрольні обчислення poly_horner(coeffs, x):")
    for xp, expected in [(0.0, -3.0), (1.0, 0.0), (2.0, 3.0), (-1.0, -12.0)]:
        pv = poly_horner(loaded, xp)
        print(f"    P({xp:+.1f}) = {pv:>12.6f}  (очікується {expected})")

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 8: НЬЮТОН + ГОРНЕР (ДІЙСНИЙ КОРІНЬ)
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 8. Метод Ньютона зі схемою Горнера — дійсний корінь")
    x_real, n_real = newton_horner(loaded, 0.5)
    print(f"  Початкове наближення: x₀ = 0.5")
    print(f"  Знайдений корінь:     x₁ = {x_real:.14f}")
    print(f"  P(x₁) = {poly_horner(loaded, x_real):.4e}   (має → 0)")
    print(f"  Кількість ітерацій:   {n_real}")

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАВДАННЯ 9: МЕТОД ЛІНА — КОМПЛЕКСНІ КОРЕНІ
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ЗАВДАННЯ 9. Метод Ліна (Bairstow) — комплексні корені")
    print("  Шукаємо квадратний множник x² + p·x + q  (точний: p=−2, q=3)")
    print("  Початкові наближення: p₀ = −1.5,  q₀ = 2.5")

    p_res, q_res, z1, z2, n_lin = lin_bairstow(loaded, -1.5, 2.5)
    print(f"\n  Збіжність за {n_lin} ітерацій")
    print(f"  Квадратний множник: x² + ({p_res:.10f})x + ({q_res:.10f})")

    if isinstance(z1, complex):
        print(f"\n  Комплексні корені:")
        print(f"    z₁ = {z1.real:.10f} + {z1.imag:.10f}·i")
        print(f"    z₂ = {z2.real:.10f} + {z2.imag:.10f}·i")
        print(f"\n  Перевірка (точні: 1 ± i√2 ≈ 1 ± 1.41421356i):")
        print(f"    Re = {z1.real:.10f}  (має бути 1.0)")
        print(f"    Im = {z1.imag:.10f}  (має бути ±{sqrt(2):.10f})")
        print(f"    |z₁|² = {abs(z1)**2:.10f}  (має бути = q = {q_res:.6f})")
    else:
        print(f"    z₁ = {z1:.10f},  z₂ = {z2:.10f}  (дійсні)")

    # Незалежна перевірка numpy
    np_roots = np.roots(loaded)
    print(f"\n  Незалежна перевірка (numpy.roots):")
    for r in sorted(np_roots, key=lambda z: abs(z.imag)):
        if abs(r.imag) < 1e-10:
            print(f"    x = {r.real:.12f}  (дійсний)")
        else:
            print(f"    z = {r.real:.12f} {r.imag:+.12f}i  (комплексний)")

    # Графіки многочлена
    f3 = plot_polynomial(loaded, x_real, z1, z2)
    print(f"\n  Графік: '{f3}'")

    # ══════════════════════════════════════════════════════════════════════════
    # ПІДСУМОК
    # ══════════════════════════════════════════════════════════════════════════
    _hdr("ПІДСУМОК — ЗВІТ")

    print(f"  Рівняння 1: f(x) = x·sin(x) − 0.6 = 0")
    print(f"    Корінь 1 (зростання): x₁ = {r1_precise:.10f}")
    print(f"    Корінь 2 (спадання):  x₂ = {r2_precise:.10f}")
    print()
    print(f"  Рівняння 2: P(x) = x³ − 3x² + 5x − 3 = 0")
    print(f"    Дійсний корінь:       x₁ = {x_real:.10f}")
    if isinstance(z1, complex):
        print(f"    Комплексні корені:    z₂ = {z1.real:.6f} + {z1.imag:.6f}i")
        print(f"                          z₃ = {z2.real:.6f} + {z2.imag:.6f}i")

    print()
    all_files = [TAB_FILENAME, POLY_FILENAME,
                 "plot_transcendental.png", "plot_iterations.png", "plot_polynomial.png"]
    print("  Вихідні файли:")
    for fn in all_files:
        tag = "✓" if os.path.exists(fn) else "✗"
        print(f"    [{tag}] {fn}")

    print(f"\n{_sep()}\n  Готово!\n{_sep()}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()