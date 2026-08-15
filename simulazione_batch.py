#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulazione autonoma: "Selezione Dinamica della Dimensione del Campione"
===========================================================================
Riproduce la FIGURA 5.3 della tesi: andamento della dimensione del batch
n_k (strategia dinamica basata sulla Condizione di Controllo della Varianza,
CCV) in funzione dell'iterazione k, confrontato con un batch fisso.

Parametri (IDENTICI a sim_exp.py usato per la tesi):
  - problema: regressione lineare (loss quadratica), N = 1000 esempi,
    m = 10 variabili, X ~ N(0, I)  ->  kappa ~ 1.4
  - punto iniziale: w0 = (2, ..., 2)
  - batch iniziale: n0 = 16
  - tolleranza CCV: theta = 0.5
  - max iterazioni: 250
  - passo: 1/L con line search di Armijo (backtracking)
  - arresto: ||grad J(w)|| < 1e-6
  - seed fisso per riproducibilita'

In output genera (nella cartella figure_sim/, per non sovrascrivere le
figure della tesi):
  - figure_sim/batch_size.png / .pdf   : n_k vs k (dinamico vs fisso)
  - figure_sim/convergenza.png / .pdf  : J(w_k)-J(w*) vs k (scala log),
    per theta in {0.1, 0.5, 0.9}   [compagno della Fig. 5.3, = Fig. 6.1]

Dipendenze: numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PARAMETRI
# ----------------------------------------------------------------------
SEED     = 42
N        = 1000   # numero di esempi
m        = 10     # dimensionalita' dello spazio dei parametri
THETA    = 0.5    # tolleranza nella CCV
BATCH0   = 16     # dimensione iniziale del batch
MAX_ITER = 250    # budget di iterazioni
TOL      = 1e-6   # tolleranza per il criterio di arresto

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# PROBLEMA: regressione lineare  J(w) = 1/(2N) sum_i (x_i^T w - y_i)^2
# ----------------------------------------------------------------------
X = rng.standard_normal((N, m))
w_true = rng.standard_normal(m)
y = X @ w_true + 0.1 * rng.standard_normal(N)

H = X.T @ X / N
L  = np.linalg.eigvalsh(H).max()          # costante di Lipschitz
lam = np.linalg.eigvalsh(H).min()         # convessita' forte
w_star = np.linalg.solve(X.T @ X, X.T @ y)
kappa = L / lam

def loss_i(w, i):
    return 0.5 * (X[i] @ w - y[i]) ** 2

def grad_i(w, i):
    return X[i] * (X[i] @ w - y[i])

def grad_full(w):
    return X.T @ (X @ w - y) / N

def J(w):
    return 0.5 * np.mean((X @ w - y) ** 2)

print(f"kappa = {kappa:.2f}   L = {L:.3f}   lambda = {lam:.3f}")


# ----------------------------------------------------------------------
# METODO DEL GRADIENTE A CAMPIONE DINAMICO (CCV + line search Armijo)
# ----------------------------------------------------------------------
def dynamic_gd(theta, batch0=BATCH0, max_iter=MAX_ITER, alpha=None,
               tol=TOL, seed=SEED):
    """Ritorna (w, batch_sizes) eseguendo il gradiente dinamico con CCV.

    Regola CCV (formula pratica della tesi, N >> n):
        se  ||V_hat||_1 / n  >  theta^2 * ||g||^2
        allora  n <- min( ceil(||V_hat||_1 / (theta^2 ||g||^2)) + 1, N )
    """
    r = np.random.default_rng(seed)
    w = np.full(m, 2.0)
    n = max(batch0, 2)
    if alpha is None:
        alpha = 1.0 / L
    batch_sizes = [n]

    for _ in range(max_iter):
        inds = r.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in inds])
        g = grads.mean(axis=0)

        # --- CCV ---
        if n > 1:
            V_norm1 = np.var(grads, axis=0, ddof=1).sum()
            gg = g @ g
            if gg > 1e-16 and V_norm1 / n > theta ** 2 * gg:
                n = min(int(np.ceil(V_norm1 / (theta ** 2 * gg))) + 1, N)

        # --- line search di Armijo (backtracking) sul batch corrente ---
        step = alpha
        Jc = np.mean([loss_i(w, i) for i in inds])
        gd = -(g @ g)
        if g @ g > 1e-16:
            for _ in range(30):
                w_new = w + step * (-g)
                if np.mean([loss_i(w_new, i) for i in inds]) <= Jc + 1e-4 * step * gd:
                    break
                step *= 0.5
            w = w + step * (-g)
        else:
            w = w + step * (-g)

        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < tol:
            break
    return w, batch_sizes


# ----------------------------------------------------------------------
# FIGURA 5.3: n_k vs k  (dinamico vs batch fisso)
# ----------------------------------------------------------------------
import os
os.makedirs("figure_sim", exist_ok=True)

w, sizes = dynamic_gd(THETA)

ks = np.arange(len(sizes))
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.step(ks, sizes, where="mid", color="#1f77b4", lw=1.8,
        label="Dinamico (CCV)")
ax.axhline(BATCH0, color="#d62728", ls="--", lw=1.4,
           label=f"Batch fisso $n={BATCH0}$")
ax.set_xlabel(r"Iterazione $k$")
ax.set_ylabel(r"Dimensione del batch $n_k$")
ax.set_title("Evoluzione dinamica della dimensione del batch")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figure_sim/batch_size.pdf")
fig.savefig("figure_sim/batch_size.png", dpi=150)
plt.close(fig)
print(f"Figura 5.3 salvata in figure_sim/batch_size.pdf  "
      f"({len(sizes)} iterazioni, batch finale = {sizes[-1]})")


# ----------------------------------------------------------------------
# FIGURA COMPAGNA (Fig. 6.1): convergenza per theta in {0.1, 0.5, 0.9}
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for th, col in zip((0.1, 0.5, 0.9), ("#1f77b4", "#d62728", "#2ca02c")):
    w = np.full(m, 2.0)
    n = BATCH0
    hist = [J(w) - J(w_star)]
    for _ in range(MAX_ITER):
        inds = rng.choice(N, size=n, replace=False)
        g = np.mean([grad_i(w, i) for i in inds], axis=0)
        if n > 1:
            V = np.var([grad_i(w, i) for i in inds], axis=0, ddof=1).sum()
            gg = g @ g
            if gg > 1e-16 and V / n > th ** 2 * gg:
                n = min(int(np.ceil(V / (th ** 2 * gg))) + 1, N)
        step = 1.0 / L
        Jc = np.mean([loss_i(w, i) for i in inds])
        gd = -(g @ g)
        for _ in range(30):
            wn = w - step * g
            if np.mean([loss_i(wn, i) for i in inds]) <= Jc + 1e-4 * step * gd:
                break
            step *= 0.5
        w = wn
        hist.append(J(w) - J(w_star))
        if np.linalg.norm(grad_full(w)) < TOL:
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
fig.savefig("figure_sim/convergenza.pdf")
fig.savefig("figure_sim/convergenza.png", dpi=150)
plt.close(fig)
print("Figura compagnona (convergenza) salvata in figure_sim/convergenza.pdf")
print("FATTO")

