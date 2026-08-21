#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulazione "Selezione Dinamica della Dimensione del Campione"
===================================================================
Riproduce FEDELMENTE l'implementazione dell'applicazione web interattiva
(visualizzazione.html): preset "Quadratica ben condizionata (kappa~1.1)",
dataset sintetico "centrato" (la media campionaria dei coefficienti coincide
esattamente con i coefficienti di J). Esegue due algoritmi:

  - Dynamic GD con CCV e line search di Wolfe (default dell'app);
  - BB-CCV: Barzilai-Borwein con passo adattivo (safeguard [alpha/20, 5*alpha]),
    CCV sul batch e line search di Armijo (listato Appendice B.4 di tesi.tex).

I plot di figure_sim/ mostrano il confronto tra i due metodi.

Impostazioni (default dell'app):
  preset   = quad_well   J(w) = (w1-1)^2 + (w2+2)^2 + 0.1*w1*w2
  N        = 200         numero di esempi
  w0       = [2.0, -3.0]
  alpha    = 0.1
  theta    = 0.5
  batch0   = 5
  max_iter = 30
  seed     = 42
  arresto  = ||grad_full(w)|| < 1e-6

Output (in figure_sim/):
  - batch_size.pdf/png    : n_k vs k (dinamico CCV, fit a^k, batch fisso)
  - convergenza.pdf/png   : J(w_k)-J(w_*) vs k (scala log)

Dipendenze: numpy, matplotlib
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PARAMETRI (default dell'app)
# ----------------------------------------------------------------------
SEED     = 42
N        = 200        # dimensione del dataset
W0       = [2.0, -3.0]
ALPHA    = 0.1        # passo iniziale della line search
THETA    = 0.5        # tolleranza CCV
BATCH0   = 5          # batch iniziale
MAX_ITER = 30         # budget di iterazioni
TOL      = 1e-6       # tolleranza arresto (hardcoded nell'app)

# ----------------------------------------------------------------------
# PRESET quad_well + dataset sintetico CENTRATO
# (stesso codice dell'app: raw -> sottrazione della media -> a_i, b_i, c_i)
# ----------------------------------------------------------------------
def J(w):
    x, y = w[0], w[1]
    return (x - 1.0)**2 + (y + 2.0)**2 + 0.1 * x * y

def gradJ(w):
    x, y = w[0], w[1]
    return np.array([2.0*(x - 1.0) + 0.1*y,
                     2.0*(y + 2.0) + 0.1*x])

np.random.seed(SEED)
raw_a = 1.0 + 0.2 * np.random.randn(N)
raw_b = -2.0 + 0.2 * np.random.randn(N)
raw_c = 0.1 + 0.05 * np.random.randn(N)
a_i = raw_a - np.mean(raw_a) + 1.0     # media esatta == 1.0
b_i = raw_b - np.mean(raw_b) - 2.0     # media esatta == -2.0
c_i = raw_c - np.mean(raw_c) + 0.1     # media esatta == 0.1

def loss_i(w, i):
    x, y = w[0], w[1]
    return (x - a_i[i])**2 + (y - b_i[i])**2 + c_i[i] * x * y

def grad_i(w, i):
    x, y = w[0], w[1]
    return np.array([2.0*(x - a_i[i]) + c_i[i]*y,
                     2.0*(y - b_i[i]) + c_i[i]*x])

def grad_full(w):
    return np.mean([grad_i(w, i) for i in range(N)], axis=0)

W_STAR = np.array([1.0, -2.0])   # minimo del preset quad_well

# ----------------------------------------------------------------------
# ALGORITMO: Dynamic GD - implementazione IDENTICA all'app (Wolfe + CCV)
# ----------------------------------------------------------------------
def dynamic_gd(w0, theta, max_iter, alpha, batch0):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]
    for k in range(max_iter):
        indices = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)

        # --- Condizione di Controllo della Varianza (CCV) ---
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g, g)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, N)

        # --- line search di WOLFE sul batch corrente ---
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])

        c1, c2 = 1e-4, 0.9
        step = alpha
        J_curr = J_batch(w)
        g_norm2 = np.dot(g, g)
        d = -g
        gd = -g_norm2
        if g_norm2 > 1e-16:
            for _ in range(30):
                w_new = w + step * d
                if J_batch(w_new) <= J_curr + c1 * step * gd:
                    g_new = np.mean([grad_i(w_new, i) for i in indices],
                                    axis=0)
                    if np.dot(g_new, d) >= c2 * gd:
                        break
                step *= 0.5
            else:
                step = 0.0
        w = w + step * d

        if np.linalg.norm(grad_full(w)) < TOL:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes

# ----------------------------------------------------------------------
# ALGORITMO: BB-CCV (Barzilai-Borwein con CCV) - listato Appendice B.4
# passo adattivo BB con safeguard + line search di Armijo + CCV identica
# ----------------------------------------------------------------------
def bb_dynamic_gd(w0, theta, max_iter, alpha, batch0):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]

    w_prev = w.copy()
    g_prev = None

    for k in range(max_iter):
        indices = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)

        # --- passo di Barzilai-Borwein ---
        if k > 0 and g_prev is not None:
            s = w - w_prev
            y = g - g_prev
            sy = np.dot(s, y)
            if abs(sy) > 1e-14:
                step_bb = np.dot(s, s) / sy
                step = np.clip(step_bb, alpha / 20.0, alpha * 5.0)
            else:
                step = alpha
        else:
            step = alpha

        w_prev = w.copy()
        g_prev = g.copy()

        # --- line search di ARMIJO sul batch corrente ---
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])

        c1 = 1e-4
        J_curr = J_batch(w)
        g_norm2 = np.dot(g, g)

        if g_norm2 > 1e-16:
            for _ in range(30):
                w_new = w - step * g
                if J_batch(w_new) <= J_curr - c1 * step * g_norm2:
                    break
                step *= 0.5
            else:
                step = 0.0

        w = w - step * g

        # --- Condizione di Controllo della Varianza (CCV), identica al GD ---
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g, g)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, N)

        history.append(w.copy().tolist())
        batch_sizes.append(n)

        if np.linalg.norm(grad_full(w)) < TOL:
            break

    return history, batch_sizes

# ----------------------------------------------------------------------
# ESECUZIONE: entrambi gli algoritmi con i default dell'app
# ----------------------------------------------------------------------
hist_gd,  bs_gd  = dynamic_gd(W0, THETA, MAX_ITER, ALPHA, BATCH0)
hist_bb,  bs_bb  = bb_dynamic_gd(W0, THETA, MAX_ITER, ALPHA, BATCH0)

def metriche(history, batch_sizes):
    pts_J = [float(J(np.array(w))) for w in history]
    errs  = [float(np.linalg.norm(np.array(w) - W_STAR)) for w in history]
    J_star = float(J(W_STAR))
    return (len(history), batch_sizes[-1], pts_J[-1] - J_star, errs[-1])

nit_gd, nf_gd, dJ_gd, err_gd = metriche(hist_gd, bs_gd)
nit_bb, nf_bb, dJ_bb, err_bb = metriche(hist_bb, bs_bb)

print("Dynamic GD:", f"iter={nit_gd}  batch_fin={nf_gd}"
      f"  J(w_k)-J(w*)={dJ_gd:.2e}  ||w-w*||={err_gd:.2e}")
print("BB-CCV    :", f"iter={nit_bb}  batch_fin={nf_bb}"
      f"  J(w_k)-J(w*)={dJ_bb:.2e}  ||w-w*||={err_bb:.2e}")

# ----------------------------------------------------------------------
# FIGURA 1: n_k vs k (come il pannello "n_k vs a^k" dell'app)
# ----------------------------------------------------------------------
os.makedirs("figure_sim", exist_ok=True)

def compute_best_fit_a(batch_sizes):
    """Fit a > 1 tale che n_k ~ a^k (minimi quadrati in scala log)."""
    ks, logs = [], []
    for k, nk in enumerate(batch_sizes):
        if nk > 0:
            ks.append(k)
            logs.append(np.log(nk))
    if len(ks) < 2:
        return 1.0
    num = sum(k * l for k, l in zip(ks, logs))
    den = sum(k * k for k in ks)
    return float(np.exp(num / den)) if den else 1.0

a_fit_gd = compute_best_fit_a(bs_gd)
a_fit_bb = compute_best_fit_a(bs_bb)
ks_gd = np.arange(len(bs_gd))
ks_bb = np.arange(len(bs_bb))

fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.step(ks_gd, bs_gd, where="mid", color="#1f77b4", lw=1.8,
        label="Dynamic GD (CCV)")
ax.plot(ks_gd, a_fit_gd ** ks_gd, color="#1f77b4", ls="--", lw=1.2,
        label=rf"fit GD $a^k$ ($a={a_fit_gd:.4f}$)")
ax.step(ks_bb, bs_bb, where="mid", color="#2ca02c", lw=1.8,
        label="BB-CCV")
ax.plot(ks_bb, a_fit_bb ** ks_bb, color="#2ca02c", ls="--", lw=1.2,
        label=rf"fit BB $a^k$ ($a={a_fit_bb:.4f}$)")
ax.axhline(BATCH0, color="#d62728", ls=":", lw=1.4,
           label=f"Batch fisso $n={BATCH0}$")
ax.set_xlabel(r"Iterazione $k$")
ax.set_ylabel(r"Dimensione del batch $n_k$")
ax.set_title("Evoluzione dinamica della dimensione del batch (CCV)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figure_sim/batch_size.pdf")
fig.savefig("figure_sim/batch_size.png", dpi=150)
plt.close(fig)
print(f"Figura batch_size salvata in figure_sim/  "
      f"(GD: {len(bs_gd)} it, batch fin. {bs_gd[-1]}, a={a_fit_gd:.4f} | "
      f"BB: {len(bs_bb)} it, batch fin. {bs_bb[-1]}, a={a_fit_bb:.4f})")

# ----------------------------------------------------------------------
# FIGURA 2: convergenza ||w_k - w*|| (scala log) - come la metrica "errs"
# dell'app (l'app monitora la convergenza tramite la norma, dato che
# W_STAR=[1,-2] è il minimo nominale e J(.)-J(W_STAR) può essere < 0)
# ----------------------------------------------------------------------
def err_serie(history):
    return np.maximum(
        np.asarray([np.linalg.norm(np.array(w) - W_STAR) for w in history]),
        1e-14)

err_gd = err_serie(hist_gd)
err_bb = err_serie(hist_bb)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.semilogy(range(len(err_gd)), err_gd, color="#1f77b4", lw=1.6,
            marker="o", ms=3.5, label="Dynamic GD (Wolfe + CCV)")
ax.semilogy(range(len(err_bb)), err_bb, color="#2ca02c", lw=1.6,
            marker="s", ms=3.5, label="BB-CCV (Armijo + CCV)")
ax.set_xlabel(r"Iterazioni $k$")
ax.set_ylabel(r"$\|w_k - w_*\|_2$")
ax.set_title("Convergenza del gradiente a campione dinamico")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figure_sim/convergenza.pdf")
fig.savefig("figure_sim/convergenza.png", dpi=150)
plt.close(fig)
print(f"Figura convergenza salvata in figure_sim/  "
      f"(errore finale ||w-w*|| GD = {err_gd[-1]:.2e}, "
      f"BB = {err_bb[-1]:.2e})")
print("FATTO")

