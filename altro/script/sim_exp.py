#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulazione degli algoritmi per la tesi "Selezione Dinamica della
Dimensione del Campione in Metodi di Ottimizzazione per il Machine Learning".

Genera:
  - figure/convergenza.pdf     : J(w_k)-J(w*) vs k (scala log) per diversi theta
  - figure/batch_size.pdf      : n_k vs k (dinamico vs fisso)
  - figure/bar_comparison.pdf  : valutazioni di gradiente totali per metodo
  - figure/cono_discesa.pdf    : schema geometrico della condizione di discesa
  - tabella6_1.txt / tabella6_2.txt : righe LaTeX per le tabelle di risultati
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N = 1000          # numero di esempi
m = 10            # dimensionalita'

np.set_printoptions(precision=4, suppress=True)


# ----------------------------------------------------------------------
# Problema 1: regressione lineare (loss quadratica), X ~ N(0,I)
# ----------------------------------------------------------------------
def build_ls(kappa_goal=None):
    X = rng.standard_normal((N, m))
    if kappa_goal is not None:
        # Introduce correlazione per alzare il condizionamento di H = X'X/N
        for j in range(1, m):
            X[:, j] = X[:, 0] * (0.9 - 0.05 * j) + 0.5 * X[:, j]
    w_true = rng.standard_normal(m)
    y = X @ w_true + 0.1 * rng.standard_normal(N)
    H = X.T @ X / N
    L = np.linalg.eigvalsh(H).max()
    lam = np.linalg.eigvalsh(H).min()
    w_star = np.linalg.solve(X.T @ X, X.T @ y)

    def J(w):
        return 0.5 * np.mean((X @ w - y) ** 2)

    def grad_full(w):
        return X.T @ (X @ w - y) / N

    def hess_full(w):
        return H

    def loss_i(w, i):
        return 0.5 * (X[i] @ w - y[i]) ** 2

    def grad_i(w, i):
        return X[i] * (X[i] @ w - y[i])

    def hessvec_i(w, i, v):
        return (X[i] @ v) * X[i]

    return dict(J=J, grad_full=grad_full, hess_full=hess_full, loss_i=loss_i,
                grad_i=grad_i, hessvec_i=hessvec_i, w_star=w_star,
                L=L, lam=lam, kappa=L / lam, label="Quadratica")


# ----------------------------------------------------------------------
# Problema 2: Rosenbrock con dataset "centrato" (media nulla delle perturbazioni)
# ----------------------------------------------------------------------
def build_rosen():
    # d_i ~ tale che sum_i d_i = 0   (trucco "dataset centrato" della tesi)
    d = rng.standard_normal((N, m))
    d -= d.mean(axis=0)
    w_star = np.ones(m)

    def J(w):
        wm = w[:-1]
        return np.sum(100.0 * (w[1:] - wm ** 2) ** 2 + (1.0 - wm) ** 2)

    def grad_full(w):
        g = np.zeros_like(w)
        g[:-1] += -400.0 * w[:-1] * (w[1:] - w[:-1] ** 2) - 2.0 * (1.0 - w[:-1])
        g[1:] += 200.0 * (w[1:] - w[:-1] ** 2)
        return g

    def hess_full(w):
        n = len(w)
        H = np.zeros((n, n))
        for i in range(n - 1):
            H[i, i] += 1200.0 * w[i] ** 2 - 400.0 * w[i + 1] + 2.0
            H[i, i + 1] += -400.0 * w[i]
            H[i + 1, i] += -400.0 * w[i]
            H[i + 1, i + 1] += 200.0
        return H

    def loss_i(w, i):
        return J(w) + d[i] @ w

    def grad_i(w, i):
        return grad_full(w) + d[i]

    def hessvec_i(w, i, v):
        return hess_full(w) @ v

    return dict(J=J, grad_full=grad_full, hess_full=hess_full, loss_i=loss_i,
                grad_i=grad_i, hessvec_i=hessvec_i, w_star=w_star,
                L=400.0, lam=1.0, kappa=400.0, label="Rosenbrock")


# ----------------------------------------------------------------------
# Algoritmi
# ----------------------------------------------------------------------
def dynamic_gd(prob, w0, theta, max_iter=300, alpha=None, batch0=16,
               tol=1e-6):
    """Gradiente a campione dinamico con line search di Wolfe sul batch."""
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    if alpha is None:
        alpha = 1.0 / prob["L"]
    hist, batch_sizes, evals = [], [], 0
    for k in range(max_iter):
        inds = rng.choice(N, size=n, replace=False)
        grads = np.array([prob["grad_i"](w, i) for i in inds])
        g = grads.mean(axis=0)
        evals += n
        # CCV
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
            V_norm1 = var_vec.sum()
            gg = g @ g
            if gg > 1e-16 and V_norm1 / n > theta ** 2 * gg:
                n = min(int(np.ceil(V_norm1 / (theta ** 2 * gg))) + 1, N)
        # line search Wolfe (sul batch corrente)
        Jc = np.mean([prob["loss_i"](w, i) for i in inds])
        c1, c2 = 1e-4, 0.9
        step = alpha
        gd = - (g @ g)
        if g @ g > 1e-16:
            for _ in range(30):
                w_new = w + step * (-g)
                if np.mean([prob["loss_i"](w_new, i) for i in inds]) <= Jc + c1 * step * gd:
                    g_new = np.mean([prob["grad_i"](w_new, i) for i in inds], axis=0)
                    if g_new @ (-g) >= c2 * gd:
                        break
                step *= 0.5
            else:
                step = 0.0
        w = w + step * (-g)
        hist.append(w.copy())
        batch_sizes.append(n)
        if np.linalg.norm(prob["grad_full"](w)) < tol:
            break
    return w, hist, batch_sizes, evals


def sgd(prob, w0, max_iter=300, batch=1, tol=1e-6):
    """SGD con passo 1/(lambda*k) limitato superiormente da 1/L (stile classico)."""
    w = np.array(w0, dtype=float)
    hist, evals = [], 0
    for k in range(max_iter):
        inds = rng.choice(N, size=batch, replace=False)
        grads = np.array([prob["grad_i"](w, i) for i in inds])
        g = grads.mean(axis=0)
        step = min(1.0 / (prob["lam"] * (k + 1)), 1.0 / prob["L"])
        w = w - step * g
        evals += batch
        hist.append(w.copy())
        if np.linalg.norm(prob["grad_full"](w)) < tol:
            break
    return w, hist, None, evals


def batch_gd(prob, w0, max_iter=300, tol=1e-6):
    w = np.array(w0, dtype=float)
    hist, evals = [], 0
    for k in range(max_iter):
        g = prob["grad_full"](w)
        step = 1.0 / prob["L"]
        w = w - step * g
        evals += N
        hist.append(w.copy())
        if np.linalg.norm(g) < tol:
            break
    return w, hist, None, evals


def newton_cg(prob, w0, theta, max_iter=300, batch0=16, R=0.1, maxcg=60,
              tol=1e-6):
    """Newton-CG con campionamento dinamico e test di arresto adattivo."""
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    hist, batch_sizes, cg_iters, evals = [], [], [], 0

    def cg(A, b, gamma, maxcg):
        x = np.zeros_like(b)
        r = b - A(x)
        p = r.copy()
        rr = r @ r
        niter = 0
        for _ in range(maxcg):
            Ap = A(p)
            pHp = p @ Ap
            if pHp <= 1e-14:
                break
            alpha = rr / pHp
            x = x + alpha * p
            r_new = r - alpha * Ap
            rr_new = r_new @ r_new
            niter += 1
            if rr_new <= gamma * (x @ x) + 1e-16:
                return x, niter
            beta = rr_new / rr
            p = r_new + beta * p
            r, rr = r_new, rr_new
        return x, niter

    for k in range(max_iter):
        inds_S = rng.choice(N, size=n, replace=False)
        g = np.mean([prob["grad_i"](w, i) for i in inds_S], axis=0)
        evals += n
        n_h = min(max(1, int(round(R * n))), N)
        inds_H = rng.choice(inds_S, size=n_h, replace=False)
        Hv = lambda v: np.mean([prob["hessvec_i"](w, i, v) for i in inds_H], axis=0)
        p0 = -g
        gamma = 0.0
        if p0 @ p0 > 1e-16 and n_h > 1:
            Hp0 = np.array([prob["hessvec_i"](w, i, p0) for i in inds_H])
            gamma = np.sum(np.var(Hp0, axis=0, ddof=1)) / (n_h * (p0 @ p0))
        d, n_cg = cg(Hv, -g, gamma, maxcg)
        evals += n_cg * n_h
        cg_iters.append(n_cg)
        Jb = lambda wc: np.mean([prob["loss_i"](wc, i) for i in inds_S])
        c1, c2 = 1e-4, 0.9
        step, J_w = 1.0 / prob["L"], Jb(w)
        gd = g @ d
        if gd >= 0:
            d = -g
            gd = - (g @ g)
        for _ in range(30):
            w_new = w + step * d
            if Jb(w_new) <= J_w + c1 * step * gd:
                g_new = np.mean([prob["grad_i"](w_new, i) for i in inds_S], axis=0)
                if g_new @ d >= c2 * gd:
                    break
            step *= 0.5
        else:
            step = 0.0
        w = w + step * d
        # CCV
        inds_new = rng.choice(N, size=n, replace=False)
        g_new = np.mean([prob["grad_i"](w, i) for i in inds_new], axis=0)
        evals += n
        if n > 1:
            var_vec = np.var([prob["grad_i"](w, i) for i in inds_new], axis=0, ddof=1)
            V_norm1 = var_vec.sum()
            if g_new @ g_new > 1e-16 and V_norm1 / n > theta ** 2 * (g_new @ g_new):
                n = min(int(np.ceil(V_norm1 / (theta ** 2 * (g_new @ g_new)))) + 1, N)
        hist.append(w.copy())
        batch_sizes.append(n)
        if np.linalg.norm(prob["grad_full"](w)) < tol:
            break
    return w, hist, batch_sizes, evals, cg_iters



# ----------------------------------------------------------------------
# Esecuzione
# ----------------------------------------------------------------------
w0 = np.full(m, 2.0)
probs = [build_ls(), build_ls(kappa_goal=1), build_rosen()]
probs[1]["label"] = "Quadratica mal condiz."

res = {"tab6_1": [], "tab6_2": []}

print("=" * 78)
for prob in probs:
    ws = prob["w_star"]
    print(f"\nProblema: {prob['label']:22s}  kappa={prob['kappa']:.2f}  "
          f"L={prob['L']:.3f}  lambda={prob['lam']:.3f}")

    rows1 = []
    # Dynamic GD per tre theta
    for th in (0.1, 0.5, 0.9):
        t0 = time.time()
        wf, hist, sizes, ev = dynamic_gd(prob, w0, th)
        dt = time.time() - t0
        err = float(np.linalg.norm(wf - ws))
        rows1.append(("Dynamic GD", f"$\\theta={th}$", len(hist),
                      f"{dt:.2f}", f"{err:.2e}", max(sizes)))
        print(f"  Dynamic GD theta={th}: it={len(hist):3d} "
              f"tempo={dt:6.2f}s err={err:.2e} max_n={max(sizes)}")
    # SGD
    t0 = time.time()
    wf, hist, _, ev = sgd(prob, w0)
    dt = time.time() - t0
    err = float(np.linalg.norm(wf - ws))
    rows1.append(("SGD", "$n=1$", len(hist), f"{dt:.2f}", f"{err:.2e}", 1))
    print(f"  SGD               : it={len(hist):3d} tempo={dt:6.2f}s err={err:.2e}")
    # Batch GD
    t0 = time.time()
    wf, hist, _, ev = batch_gd(prob, w0)
    dt = time.time() - t0
    err = float(np.linalg.norm(wf - ws))
    rows1.append(("Batch GD", "$n=N$", len(hist), f"{dt:.2f}", f"{err:.2e}", N))
    print(f"  Batch GD          : it={len(hist):3d} tempo={dt:6.2f}s err={err:.2e}")

    rows2 = []
    t0 = time.time()
    wf, hist, sizes, ev, cg_it = newton_cg(prob, w0, 0.5)
    dt = time.time() - t0
    err = float(np.linalg.norm(wf - ws))
    cg_avg = float(np.mean(cg_it)) if cg_it else 0.0
    rows2.append(("Newton-CG", f"$R=0.1$", len(hist), f"{cg_avg:.1f}",
                  f"{dt:.2f}", f"{err:.2e}"))
    print(f"  Newton-CG R=0.1   : it={len(hist):3d} cg_avg={cg_avg:4.1f} "
          f"tempo={dt:6.2f}s err={err:.2e}")

    # Le tabelle riportano i problemi quadratici; Rosenbrock resta per la
    # discussione qualitativa (i metodi del primo ordine vi falliscono).
    if "Quadratica" in prob["label"]:
        res["tab6_1"].append((prob["label"], rows1))
        res["tab6_2"].append((prob["label"], rows2))

print("\nGenerazione figure...")

# --- Figura 6.1: convergenza per diversi theta (scala log) ---------------
prob = probs[0]
J0 = prob["J"](w0)
Jstar = prob["J"](prob["w_star"])
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for th, col in zip((0.1, 0.5, 0.9), ("#1f77b4", "#d62728", "#2ca02c")):
    w = w0.copy()
    n = 16
    hist = [prob["J"](w) - Jstar]
    for k in range(250):
        inds = rng.choice(N, size=n, replace=False)
        grads = np.array([prob["grad_i"](w, i) for i in inds])
        g = grads.mean(axis=0)
        if n > 1:
            V = np.var(grads, axis=0, ddof=1).sum()
            gg = g @ g
            if gg > 1e-16 and V / n > th ** 2 * gg:
                n = min(int(np.ceil(V / (th ** 2 * gg))) + 1, N)
        step = 1.0 / prob["L"]
        Jc = np.mean([prob["loss_i"](w, i) for i in inds])
        gd = - (g @ g)
        for _ in range(30):
            wn = w - step * g
            if np.mean([prob["loss_i"](wn, i) for i in inds]) <= Jc + 1e-4 * step * gd:
                break
            step *= 0.5
        w = wn
        hist.append(prob["J"](w) - Jstar)
        if prob["grad_full"](w) @ prob["grad_full"](w) < 1e-12:
            break
    hist = np.maximum(np.asarray(hist), 1e-14)
    ax.semilogy(range(len(hist)), hist, label=rf"$\theta={th}$", color=col,
                lw=1.6)
ax.set_xlabel(r"Iterazioni $k$")
ax.set_ylabel(r"$J(w_k) - J(w_*)$")
ax.set_title("Convergenza del gradiente a campione dinamico")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figure/convergenza.pdf")
fig.savefig("figure/convergenza.png", dpi=150)
plt.close(fig)



# --- Figura 6.2 / 5.2: dimensione del batch n_k vs k ---------------------
w = w0.copy()
n = 16
sizes = [n]
for k in range(250):
    inds = rng.choice(N, size=n, replace=False)
    grads = np.array([prob["grad_i"](w, i) for i in inds])
    g = grads.mean(axis=0)
    th = 0.5
    if n > 1:
        V = np.var(grads, axis=0, ddof=1).sum()
        gg = g @ g
        if gg > 1e-16 and V / n > th ** 2 * gg:
            n = min(int(np.ceil(V / (th ** 2 * gg))) + 1, N)
    step = 1.0 / prob["L"]
    Jc = np.mean([prob["loss_i"](w, i) for i in inds])
    gd = - (g @ g)
    for _ in range(30):
        wn = w - step * g
        if np.mean([prob["loss_i"](wn, i) for i in inds]) <= Jc + 1e-4 * step * gd:
            break
        step *= 0.5
    w = wn
    sizes.append(n)
    if np.linalg.norm(prob["grad_full"](w)) < 1e-6:
        break

fig, ax = plt.subplots(figsize=(6.2, 4.0))
ks = np.arange(len(sizes))
ax.step(ks, sizes, where="mid", color="#1f77b4", lw=1.8,
        label="Dinamico (CCV)")
ax.axhline(16, color="#d62728", ls="--", lw=1.4, label="Batch fisso $n=16$")
ax.set_xlabel(r"Iterazione $k$")
ax.set_ylabel(r"Dimensione del batch $n_k$")
ax.set_title("Evoluzione dinamica della dimensione del batch")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figure/batch_size.pdf")
fig.savefig("figure/batch_size.png", dpi=150)
plt.close(fig)

# --- Figura 6.3: barre valutazioni di gradiente totali -------------------
labels = ["Dynamic GD\n($\\theta=0.5$)", "SGD", "Batch GD", "Newton-CG"]
vals, conv = [], []
for name in ("dynamic", "sgd", "batch", "newton"):
    if name == "dynamic":
        _, _, _, ev = dynamic_gd(prob, w0, 0.5)
        conv.append(True)
    elif name == "sgd":
        wf, _, _, ev = sgd(prob, w0)
        conv.append(np.linalg.norm(prob["grad_full"](wf)) < 1e-6)
    elif name == "batch":
        _, _, _, ev = batch_gd(prob, w0)
        conv.append(True)
    else:
        _, _, _, ev, _ = newton_cg(prob, w0, 0.5)
        conv.append(True)
    vals.append(ev)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="black",
              alpha=0.85)
for i, v in enumerate(vals):
    ax.text(i, v * 1.02, f"{int(v):,}".replace(",", "."),
            ha="center", va="bottom", fontsize=9)
ax.text(1, vals[1] * 0.9, "*", ha="center", va="top", fontsize=14,
        color="black")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel("Valutazioni di gradiente")
ax.set_title("Costo totale (budget 300 iterazioni)")
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.94, "* SGD non raggiunge $\\|\\nabla J\\|<10^{-6}$ entro il budget",
        transform=ax.transAxes, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig("figure/bar_comparison.pdf")
fig.savefig("figure/bar_comparison.png", dpi=150)
plt.close(fig)

# --- Figura 5.3: cono di discesa -----------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.set_aspect("equal")
grad = np.array([0.95, 0.31])
g_est = np.array([0.78, 0.10])
err = grad - g_est
r = np.linalg.norm(grad)
ax.annotate("", xy=grad, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=g_est, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
ang = np.linspace(-0.45, 0.45, 100)
ray = 1.25 * r
xs = ray * np.cos(ang)
ys = ray * np.sin(ang)
ax.fill_between(xs, ys, 0, color="#ffd700", alpha=0.25)
ax.plot(xs, ys, color="gold", lw=1.2)
ax.annotate("", xy=err, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6, ls="--"))
ax.text(grad[0] + 0.05, grad[1] + 0.02, r"$\nabla J(w_k)$", fontsize=13)
ax.text(g_est[0] - 0.02, g_est[1] - 0.22, r"$g_k$", color="#1f77b4", fontsize=13)
ax.text(err[0] * 0.5 + 0.02, err[1] * 0.5 + 0.02, r"$e_k$", color="#d62728",
        fontsize=13)
ax.text(0.9 * r, 0.42, r"$||e_k|| \leq \theta \, ||g_k||$", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
ax.set_xlim(-0.4, 1.5)
ax.set_ylim(-0.7, 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Condizione di discesa: $g_k$ nel cono di tolleranza")
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("figure/cono_discesa.pdf")
fig.savefig("figure/cono_discesa.png", dpi=150)
plt.close(fig)

print("Figure salvate in figure/")

# --- Stampa righe LaTeX delle tabelle -----------------------------------
with open("tabella6_1.txt", "w") as f:
    for label, rows in res["tab6_1"]:
        f.write(f"\\textbf{{{label}}} & & & & & \\\\\n")
        for name, th, it, t, e, mx in rows:
            f.write(f"  {name} & {th} & {it} & {t} & {e} & {mx} \\\\\n")
        f.write("\\midrule\n")
with open("tabella6_2.txt", "w") as f:
    for label, rows in res["tab6_2"]:
        f.write(f"\\textbf{{{label}}} & & & & & \\\\\n")
        for name, par, it, cg, t, e in rows:
            f.write(f"  {name} & {par} & {it} & {cg} & {t} & {e} \\\\\n")
        f.write("\\midrule\n")

print("Tabelle scritte in tabella6_1.txt e tabella6_2.txt")

