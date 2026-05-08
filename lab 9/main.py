"""
Лабораторна робота №9
Метод Хука-Дживса багатовимірної оптимізації
для розв'язку системи нелінійних рівнянь
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════
# 1. АЛГОРИТМ МЕТОДУ ХУКА-ДЖИВСА
# ══════════════════════════════════════════════════════════

def _explore(f, start, h, n):
    x = start.copy(); f_x = f(x); improved = False
    for i in range(n):
        xp = x.copy(); xp[i] += h
        if f(xp) < f_x: x, f_x = xp, f(xp); improved = True; continue
        xm = x.copy(); xm[i] -= h
        if f(xm) < f_x: x, f_x = xm, f(xm); improved = True
    return x, improved

def hooke_jeeves(f, x0, step=0.5, beta=0.5, eps1=1e-10, eps2=1e-10, max_iter=100000):
    n = len(x0); h = float(step)
    b = np.array(x0, dtype=float)
    trajectory = [b.copy()]

    for _ in range(max_iter):
        x, improved = _explore(f, b, h, n)
        if not improved:
            h *= beta
            if h < eps1: break
            continue
        if np.linalg.norm(x - b) < eps2 and abs(f(x) - f(b)) < eps2:
            b = x.copy(); trajectory.append(b.copy()); break
        # Пошук по зразку
        p = 2.0 * x - b
        p2, _ = _explore(f, p, h, n)
        b = x.copy(); trajectory.append(b.copy())
        if f(p2) < f(b):
            b = p2.copy(); trajectory.append(b.copy())

    return b, f(b), trajectory, len(trajectory)

# ══════════════════════════════════════════════════════════
# 2. ЦІЛЬОВІ ФУНКЦІЇ
# ══════════════════════════════════════════════════════════

def rosenbrock(x):
    return 100.0*(x[1]-x[0]**2)**2 + (1.0-x[0])**2

def himmelblau(x):
    return (x[0]**2+x[1]-11.0)**2 + (x[0]+x[1]**2-7.0)**2

def wood(x):
    return (100*(x[1]-x[0]**2)**2 + (1-x[0])**2 +
            90*(x[3]-x[2]**2)**2 + (1-x[2])**2 +
            10*(x[1]+x[3]-2)**2 + 0.1*(x[1]-x[3])**2)

# ══════════════════════════════════════════════════════════
# 3. СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ
#    f1(x,y) = x^2 + y^2 - 4 = 0  (коло R=2)
#    f2(x,y) = x*y - 1       = 0  (гіпербола)
# ══════════════════════════════════════════════════════════

def eq1(xy): return xy[0]**2 + xy[1]**2 - 4.0
def eq2(xy): return xy[0]*xy[1] - 1.0
def obj_system(xy): return eq1(xy)**2 + eq2(xy)**2

# ══════════════════════════════════════════════════════════
# 4. ЗБЕРЕЖЕННЯ ТРАЄКТОРІЇ
# ══════════════════════════════════════════════════════════

def save_trajectory(trajectory, obj_func, filename="trajectory.txt"):
    with open(filename, "w", encoding="utf-8") as out:
        out.write("Траєкторія спуску методу Хука-Дживса\n")
        out.write("Система: x²+y²=4,  x·y=1\n")
        out.write("="*64+"\n")
        out.write(f"{'Крок':>6}  {'x':>16}  {'y':>16}  {'F(x,y)':>16}\n")
        out.write("-"*64+"\n")
        for k, pt in enumerate(trajectory):
            fval = obj_func(pt)
            out.write(f"{k:6d}  {pt[0]:16.10f}  {pt[1]:16.10f}  {fval:16.10e}\n")
    print(f"  [OK] Траєкторію збережено: {filename}")

# ══════════════════════════════════════════════════════════
# 5. ГРАФІКИ
# ══════════════════════════════════════════════════════════

DARK="#0d1117"; TEXT="#c9d1d9"; GRID="#21262d"; SPINE="#30363d"
BLUE="#58a6ff"; RED="#ff7b72"; GOLD="#ffd700"; GREEN="#3fb950"

def _dark_ax(ax):
    ax.set_facecolor(DARK); ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5)

def plot_system(fname="system_equations.png"):
    fig, ax = plt.subplots(figsize=(7,7)); fig.patch.set_facecolor(DARK); _dark_ax(ax)
    t = np.linspace(0, 2*np.pi, 500)
    ax.plot(2*np.cos(t), 2*np.sin(t), color=BLUE, lw=2.5, label=r"$x^2+y^2=4$")
    xh = np.linspace(0.18, 4, 400)
    ax.plot(xh, 1/xh, color=RED, lw=2.5, label=r"$xy=1$")
    ax.plot(-xh, -1/xh, color=RED, lw=2.5)
    for sign in (+1,-1):
        for xv in [np.sqrt(2+np.sqrt(3)), np.sqrt(2-np.sqrt(3))]:
            xs = sign*xv; ys = 1.0/xs
            ax.scatter(xs, ys, color=GOLD, s=90, zorder=6)
            ax.annotate(f"({xs:.3f},{ys:.3f})", (xs,ys),
                        xytext=(6,5), textcoords="offset points", color=GOLD, fontsize=7.5)
    ax.axhline(0,color=SPINE,lw=0.8); ax.axvline(0,color=SPINE,lw=0.8)
    ax.set_xlim(-3.5,3.5); ax.set_ylim(-3.5,3.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Графіки рівнянь системи", color="#f0f6fc", fontsize=13, pad=10)
    ax.legend(facecolor="#161b22", edgecolor=SPINE, labelcolor=TEXT, fontsize=11)
    plt.tight_layout(); plt.savefig(fname, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [OK] {fname}")

def plot_rosenbrock(trajectory, fname="rosenbrock_trajectory.png"):
    fig, ax = plt.subplots(figsize=(8,7)); fig.patch.set_facecolor(DARK); _dark_ax(ax)
    xg=np.linspace(-2.5,1.8,400); yg=np.linspace(-0.5,3.5,400)
    X,Y=np.meshgrid(xg,yg); Z=100*(Y-X**2)**2+(1-X)**2
    levels=np.logspace(0,4.5,50)
    ax.contourf(X,Y,Z,levels=levels,cmap="inferno",alpha=0.75)
    ax.contour(X,Y,Z,levels=levels,colors="white",alpha=0.1,linewidths=0.3)
    tx=[p[0] for p in trajectory]; ty=[p[1] for p in trajectory]
    ax.plot(tx,ty,color=BLUE,lw=1.8,marker="o",markersize=3,label="Траєкторія",zorder=4)
    ax.scatter(tx[0],ty[0],color=GREEN,s=100,zorder=6,label="Старт")
    ax.scatter(1,1,color=GOLD,s=180,zorder=7,marker="*",label="Мінімум (1,1)")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_title("Функція Розенброка – траєкторія Хука-Дживса",color="#f0f6fc",fontsize=12,pad=10)
    ax.legend(facecolor="#161b22",edgecolor=SPINE,labelcolor=TEXT,fontsize=10)
    plt.tight_layout(); plt.savefig(fname,dpi=130,bbox_inches="tight"); plt.close()
    print(f"  [OK] {fname}")

def plot_objective(trajectory=None, fname="objective_function.png"):
    fig=plt.figure(figsize=(15,6)); fig.patch.set_facecolor(DARK)
    gs=GridSpec(1,2,figure=fig,wspace=0.38)
    xg=np.linspace(-2.8,2.8,250); yg=np.linspace(-2.8,2.8,250)
    X,Y=np.meshgrid(xg,yg); Z=(X**2+Y**2-4)**2+(X*Y-1)**2
    ax3=fig.add_subplot(gs[0],projection="3d"); ax3.set_facecolor(DARK)
    ax3.plot_surface(X,Y,Z,cmap="plasma",alpha=0.88,linewidth=0,antialiased=True)
    ax3.set_xlabel("x",color=TEXT); ax3.set_ylabel("y",color=TEXT)
    ax3.set_zlabel("F",color=TEXT); ax3.tick_params(colors="#8b949e",labelsize=7)
    ax3.set_title("Цільова функція F(x,y)",color="#f0f6fc",pad=6)
    ax2=fig.add_subplot(gs[1]); ax2.set_facecolor(DARK)
    levels=np.logspace(-5,2.5,40)
    cp=ax2.contourf(X,Y,Z,levels=levels,cmap="plasma",alpha=0.75)
    ax2.contour(X,Y,Z,levels=levels,colors="white",alpha=0.12,linewidths=0.35)
    fig.colorbar(cp,ax=ax2,pad=0.02)
    if trajectory and len(trajectory)>1:
        tx=[p[0] for p in trajectory]; ty=[p[1] for p in trajectory]
        ax2.plot(tx,ty,color=BLUE,lw=1.8,marker="o",markersize=3.5,label="Траєкторія",zorder=4)
        ax2.scatter(tx[0],ty[0],color=GREEN,s=90,zorder=6,label="Старт")
        ax2.scatter(tx[-1],ty[-1],color=GOLD,s=150,zorder=7,marker="*",label="Мінімум")
    _dark_ax(ax2); ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title("Контури F(x,y) та траєкторія спуску",color="#f0f6fc",fontsize=11,pad=8)
    ax2.legend(facecolor="#161b22",edgecolor=SPINE,labelcolor=TEXT,fontsize=9)
    plt.savefig(fname,dpi=130,bbox_inches="tight"); plt.close()
    print(f"  [OK] {fname}")

# ══════════════════════════════════════════════════════════
# ГОЛОВНА ПРОГРАМА
# ══════════════════════════════════════════════════════════

def sep(title=""):
    w=62
    if title:
        pad=max(0,(w-len(title)-2)//2)
        print("─"*pad+f" {title} "+"─"*max(0,w-pad-len(title)-2))
    else:
        print("─"*w)

def main():
    print("\n"+"═"*62)
    print("   ЛАБОРАТОРНА РОБОТА №9")
    print("   Метод Хука-Дживса багатовимірної оптимізації")
    print("═"*62)

    sep("ПУНКТ 1. Система рівнянь")
    print("""
  Задана система нелінійних рівнянь (m = 2):

      f₁(x, y)  =  x² + y² − 4  =  0        (коло R=2)
      f₂(x, y)  =  x · y − 1    =  0        (гіпербола)

  Аналітичні розв'язки:
      (x, y) ≈ (±1.9319, ±0.5176)  та  (±0.5176, ±1.9319)
""")
    plot_system("system_equations.png")

    sep("ПУНКТ 2. Алгоритм методу Хука-Дживса")
    print("""
  Два етапи:
  ① Досліджуючий пошук (_explore):
     Для i = 1..n послідовно пробуємо x[i] ± h.
     Якщо f(x±h) < f(x) – приймаємо крок.
     Якщо жодного покращення: h ← β·h; зупинка при h < ε₁.

  ② Пошук по зразку (hooke_jeeves):
     p = 2·x_current − b_base  (λ = 1)
     З точки p виконуємо досліджуючий пошук.
     Базисна точка: b ← x_current.
""")

    sep("ПУНКТ 3. Тестування – функція Розенброка")
    print("  F(x₁,x₂) = 100(x₂−x₁²)² + (1−x₁)²")
    print("  Теоретичний мінімум: (1, 1), F = 0\n")

    x0_ros = np.array([-1.2, 1.0])
    opt_r, f_r, traj_r, n_r = hooke_jeeves(
        rosenbrock, x0_ros, step=0.1, beta=0.5, eps1=1e-10, eps2=1e-10)
    print(f"  Початкове наближення : {x0_ros}")
    print(f"  Параметри            : h₀=0.1, β=0.5, ε₁=ε₂=1e-10")
    print(f"  x*   = ({opt_r[0]:.8f}, {opt_r[1]:.8f})")
    print(f"  F(x*)= {f_r:.4e}   (еталон 0.0)")
    print(f"  Точок на траєкторії  : {n_r}\n")
    plot_rosenbrock(traj_r)

    sep("  Додаткові тести")
    print()
    tests = [
        ("Химмельблау (2D)", himmelblau, np.array([0.0, 0.0]),      0.1, 1e-10),
        ("Вуд (4D)",         wood,       np.array([-3.,-1.,-3.,-1.]),0.1, 1e-8),
    ]
    for name, func, x0t, h0, eps in tests:
        opt_t, f_t, _, n_t = hooke_jeeves(func, x0t, step=h0, eps1=eps, eps2=eps)
        coords=", ".join(f"{v:.5f}" for v in opt_t)
        print(f"  {name:<22}: x* = ({coords})")
        print(f"  {'':22}  F = {f_t:.2e},  кроків = {n_t}\n")

    sep("ПУНКТ 4. Розв'язок системи нелінійних рівнянь")
    print("  Параметри: h₀ = 0.05, β = 0.5, ε₁ = ε₂ = 1e-12\n")

    starts = [
        np.array([ 1.8,  0.4]),
        np.array([-1.8, -0.4]),
        np.array([ 0.4,  1.8]),
        np.array([-0.4, -1.8]),
    ]
    print(f"  {'№':>2}  {'Початок':^18}  {'x*':^28}  {'F(x*)':^13}  {'Кроки':>6}")
    print("  "+"─"*75)
    all_solutions = []
    for k, x0 in enumerate(starts):
        opt, fval, traj, nsteps = hooke_jeeves(
            obj_system, x0, step=0.05, beta=0.5, eps1=1e-12, eps2=1e-12)
        all_solutions.append((opt, fval, traj, nsteps))
        print(f"  {k+1:>2}  ({x0[0]:5.2f},{x0[1]:5.2f})        "
              f"({opt[0]:10.6f},{opt[1]:10.6f})  {fval:13.4e}  {nsteps:>6}")

    best = min(all_solutions, key=lambda s: s[1])
    opt_best, fval_best = best[0], best[1]
    print(f"\n  Найточніший розв'язок: ({opt_best[0]:.8f}, {opt_best[1]:.8f})")
    print(f"  f₁ = {eq1(opt_best):+.2e},   f₂ = {eq2(opt_best):+.2e}")
    print(f"  F(x*) = {fval_best:.4e}\n")
    traj_main = all_solutions[0][2]

    sep("ПУНКТ 5. Траєкторія у файл")
    save_trajectory(traj_main, obj_system, "trajectory.txt")
    print(f"  Точок на траєкторії (пошук №1): {len(traj_main)}\n")
    plot_objective(traj_main)

    sep("ПУНКТ 6. Підсумковий звіт")
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║              ВИСНОВОК ЛАБОРАТОРНОЇ РОБОТИ №9             ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Система:  x² + y² = 4  (коло)                          ║
  ║            x · y   = 1  (гіпербола)                     ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Тест (Розенброк):                                       ║
  ║    x* = ({opt_r[0]:.6f}, {opt_r[1]:.6f})                ║
  ║    F  = {f_r:.2e}  (еталон 0)                       ║
  ║                                                          ║
  ║  Розв'язок системи:                                      ║
  ║    x* = ({opt_best[0]:.6f}, {opt_best[1]:.6f})           ║
  ║    f₁ = {eq1(opt_best):+.2e},  f₂ = {eq2(opt_best):+.2e}              ║
  ║    F  = {fval_best:.2e}  (практично нуль)              ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Файли виведено:                                         ║
  ║    system_equations.png      – графіки рівнянь           ║
  ║    rosenbrock_trajectory.png – траєкторія тест           ║
  ║    objective_function.png    – 3D + контури              ║
  ║    trajectory.txt            – траєкторія спуску         ║
  ╚══════════════════════════════════════════════════════════╝
""")
    sep()

if __name__ == "__main__":
    main()