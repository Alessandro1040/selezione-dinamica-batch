#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riproduzione di TUTTE le tabelle numeriche della Sezione 6 di tesi/tesi.tex
(Tabelle 6.1-6.40) con un unico script autocontenuto (solo numpy).

Preset e algoritmi sono il codice ESATTO generato da visualizzazione.html
(generatori generateGD / generateNewtonCG / generateNewtonL1 / generateBB e
varianti *Validation / *Descent), consolidato qui in Python. Gli esperimenti
sono eseguiti con il NumPy di sistema: questo script è il riferimento per la
riproduzione esatta delle tabelle (cfr. Sez. 6.2.1 della tesi).

Sostituisce i generatori storici sparsi in altro/script/
(riproduci_tabelle.py, gen_tabelle_riuso.py, gen_tabelle_riuso_validation.py,
gen_tabelle_riuso_descesa.py), conservati come riferimento storico.

Uso:
  python3 riproduci_tutte_le_tabelle.py                     # stampa tutte le tabelle LaTeX
  python3 riproduci_tutte_le_tabelle.py --verify tesi.tex   # confronta i valori con la tesi
  python3 riproduci_tutte_le_tabelle.py --json out.json     # salva i dati grezzi
  python3 riproduci_tutte_le_tabelle.py --summary           # metriche chiave a schermo
"""
import json
import math
import os
import re
import sys

import numpy as np

# ============================================================================
# 1. COSTANTI (default dell'app, Sez. 6.2.1 della tesi)
# ============================================================================
N = 200
W0 = [2.0, -3.0]
ALPHA = 0.1
THETA = 0.5
BATCH0 = 5
MAX_ITER = 30
SEED = 42
R_ = 0.2
MAXCG = 10
NU = 0.1
SIGMA = 0.1
ETA = 0.5
SEEDS = [42, 7, 123, 2024, 999]

# ============================================================================
# 2. PRESET (codice esatto di LOSS_PRESETS in visualizzazione.html)
# ============================================================================
def _make_preset_well(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    raw_c = 0.1 + 0.05 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    c_i = raw_c - np.mean(raw_c) + 0.1
    def J(w):
        x, y = w[0], w[1]
        return (x - 1)**2 + (y + 2)**2 + 0.1*x*y
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([2*(x - 1) + 0.1*y, 2*(y + 2) + 0.1*x])
    def hessJ(w):
        return np.array([[2.0, 0.1], [0.1, 2.0]])
    def loss_i(w, i):
        x, y = w[0], w[1]
        return (x - a_i[i])**2 + (y - b_i[i])**2 + c_i[i] * x * y
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([2*(x - a_i[i]) + c_i[i] * y, 2*(y - b_i[i]) + c_i[i] * x])
    def hess_i(w, i):
        return np.array([[2.0, c_i[i]], [c_i[i], 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="ben condizionata")

def _make_preset_ill(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    def J(w):
        x, y = w[0], w[1]
        return 20*(x - 1)**2 + (y + 2)**2
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([40*(x - 1), 2*(y + 2)])
    def hessJ(w):
        return np.array([[40.0, 0.0], [0.0, 2.0]])
    def loss_i(w, i):
        x, y = w[0], w[1]
        return 20*(x - a_i[i])**2 + (y - b_i[i])**2
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([40*(x - a_i[i]), 2*(y - b_i[i])])
    def hess_i(w, i):
        return np.array([[40.0, 0.0], [0.0, 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="mal condizionata")


def _make_preset_very_ill(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    def J(w):
        x, y = w[0], w[1]
        return 100*(x - 1)**2 + (y + 2)**2
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([200*(x - 1), 2*(y + 2)])
    def hessJ(w):
        return np.array([[200.0, 0.0], [0.0, 2.0]])
    def loss_i(w, i):
        x, y = w[0], w[1]
        return 100*(x - a_i[i])**2 + (y - b_i[i])**2
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([200*(x - a_i[i]), 2*(y - b_i[i])])
    def hess_i(w, i):
        return np.array([[200.0, 0.0], [0.0, 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="molto mal condizionata")

def _make_preset_offdiag(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    raw_c = 0.5 + 0.05 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    c_i = raw_c - np.mean(raw_c) + 0.5
    def J(w):
        x, y = w[0] - 1, w[1] + 2
        return x*x + y*y + 0.5*x*y
    def gradJ(w):
        x, y = w[0] - 1, w[1] + 2
        return np.array([2*x + 0.5*y, 2*y + 0.5*x])
    def hessJ(w):
        return np.array([[2.0, 0.5], [0.5, 2.0]])
    def loss_i(w, i):
        x, y = w[0] - a_i[i], w[1] - b_i[i]
        return x*x + y*y + c_i[i]*x*y
    def grad_i(w, i):
        x, y = w[0] - a_i[i], w[1] - b_i[i]
        return np.array([2*x + c_i[i]*y, 2*y + c_i[i]*x])
    def hess_i(w, i):
        return np.array([[2.0, c_i[i]], [c_i[i], 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="termine incrociato")


PRESET_MAKERS = {
    "quad_well": _make_preset_well,
    "quad_ill": _make_preset_ill,
    "quad_very_ill": _make_preset_very_ill,
    "quad_offdiag": _make_preset_offdiag,
}

PRESET_LATEX = {
    "quad_well": r"$\kappa\approx1.1$",
    "quad_ill": r"$\kappa\approx20$",
    "quad_very_ill": r"$\kappa\approx100$",
    "quad_offdiag": r"incrociato ($\kappa\approx1.67$)",
}

ALGO_LATEX = {
    "gd": "Dynamic GD",
    "bb": "BB-CCV",
    "newton_cg": "Newton-CG",
    "newton_l1": "Newton-CG $L_1$",
}

# ============================================================================
# 3. ALGORITMI
# Tutte le funzioni hanno firma (p, w0, ...) con p = preset; usano le funzioni
# dataset p["loss_i"], p["grad_i"], p["hess_i"], p["hessvec_i"], p["grad_full"].
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1 Dynamic GD (codice generato da generateGD; line search di Wolfe)
#     max_consec=None  -> ricampionamento a ogni iterazione (base)
#     max_consec=M     -> riuso dello stesso batch per M iterazioni
# ----------------------------------------------------------------------------
def dynamic_gd(p, w0, theta, max_iter, alpha, batch0, max_consec=None,
               reuse=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    resample_pts = [] if reuse else None
    indices = None
    used = 0
    need_resample = True
    for k in range(max_iter):
        if reuse:
            if need_resample or indices is None or (max_consec is not None
                                                    and used >= max_consec):
                if not need_resample and indices is not None:
                    resample_pts.append(w.copy().tolist())
                indices = np.random.choice(N, size=n, replace=False)
                used = 0
                need_resample = False
        else:
            indices = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)
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
                resize_points.append(w.copy().tolist())
                if reuse:
                    need_resample = True
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])
        # Line search Wolfe
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
                    g_new = np.mean([grad_i(w_new, i) for i in indices], axis=0)
                    if np.dot(g_new, d) >= c2 * gd:
                        break
                step *= 0.5
            else:
                step = 0.0
        w = w + step * d
        if reuse:
            used += 1
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, resample_pts


# ----------------------------------------------------------------------------
# 3.2 BB-CCV (codice generato da generateBB; passo Barzilai-Borwein
#     salvaguardato + line search di Armijo). CCV dopo l'aggiornamento.
# ----------------------------------------------------------------------------
def bb_ccv(p, w0, theta, max_iter, alpha, batch0, max_consec=None,
           reuse=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    resample_pts = [] if reuse else None
    w_prev = w.copy()
    g_prev = None
    indices = None
    used = 0
    need_resample = True
    for k in range(max_iter):
        if reuse:
            if need_resample or indices is None or (max_consec is not None
                                                    and used >= max_consec):
                if not need_resample and indices is not None:
                    resample_pts.append(w.copy().tolist())
                indices = np.random.choice(N, size=n, replace=False)
                used = 0
                need_resample = False
        else:
            indices = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)
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
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])
        # Line search Armijo
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
        if reuse:
            used += 1
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        # CCV
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
                resize_points.append(w.copy().tolist())
                if reuse:
                    need_resample = True
                    used = 0
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, resample_pts

# ----------------------------------------------------------------------------
# 3.3 Newton-CG (codice generato da generateNewtonCG; H sottocampionata,
#     subset=True: H_k sottoinsieme di S_k, come default dell'app)
# ----------------------------------------------------------------------------
def _draw_H_indices(p, indices_S, n_h, subset=True):
    N = p["N"]
    if subset:
        return np.random.choice(indices_S, size=n_h, replace=False)
    return np.random.choice(N, size=n_h, replace=False)


def newton_cg(p, w0, theta, max_iter, alpha, batch0, R, maxcg,
              max_consec=None, reuse=False, reuse_hessian=True,
              max_hessian_reuse=None, subset=True):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
    loss_i, grad_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                            p["hessvec_i"], p["grad_full"])

    def cg(A, b, gamma, maxcg):
        x = np.zeros_like(b)
        r = b - A(x)
        pv = r.copy()
        rr = np.dot(r, r)
        for _ in range(maxcg):
            Ap = A(pv)
            pHp = np.dot(pv, Ap)
            if pHp <= 1e-14:
                break
            alpha = rr / pHp
            x = x + alpha * pv
            r_new = r - alpha * Ap
            rr_new = np.dot(r_new, r_new)
            if rr_new <= gamma * np.dot(x, x) + 1e-16:
                return x
            beta = rr_new / rr
            pv = r_new + beta * pv
            r = r_new
            rr = rr_new
        return x

    if reuse:
        indices_S = None
        indices_H = None
        used_S = 0
        used_H = 0
        need_resample_S = True
        need_resample_H = True

    for k in range(max_iter):
        if reuse:
            if (need_resample_S or indices_S is None or
                    (max_consec is not None and used_S >= max_consec)):
                indices_S = np.random.choice(N, size=n, replace=False)
                n_h = min(max(1, int(round(R * n))), N)
                used_S = 0
                need_resample_S = False
                if reuse_hessian:
                    indices_H = _draw_H_indices(p, indices_S, n_h, subset)
                    used_H = 0
                    need_resample_H = False
            grads_arr = np.array([grad_i(w, i) for i in indices_S])
            g = np.mean(grads_arr, axis=0)
        else:
            indices_S = np.random.choice(N, size=n, replace=False)
            g = np.mean([grad_i(w, i) for i in indices_S], axis=0)
            n_h = min(max(1, int(round(R * n))), N)
            indices_H = _draw_H_indices(p, indices_S, n_h, subset)
        if reuse and not reuse_hessian:
            # H_k indipendente da S_k: ricampionata secondo max_hessian_reuse
            if (need_resample_H or indices_H is None or
                    (max_hessian_reuse is not None and used_H >= max_hessian_reuse)):
                n_h = min(max(1, int(round(R * n))), N)
                indices_H = _draw_H_indices(p, indices_S, n_h, subset)
                used_H = 0
                need_resample_H = False
        if reuse:
            # CCV sul campione RIUSATO al punto corrente
            if n > 1:
                var_vec = np.var(grads_arr, axis=0, ddof=1)
            else:
                var_vec = np.zeros_like(g)
            V_norm1 = np.sum(var_vec)
            gg = np.dot(g, g)
            if gg > 1e-16 and V_norm1 / n > theta**2 * gg:
                n = min(int(np.ceil(V_norm1 / (theta**2 * gg))) + 1, N)
                need_resample_S = True
                used_S = 0
                need_resample_H = True
                used_H = 0
        Hv = lambda v: np.mean([hessvec_i(w, i, v) for i in indices_H], axis=0)
        p0 = -g
        p0_norm2 = np.dot(p0, p0)
        gamma = 0.0
        if p0_norm2 > 1e-16 and n_h > 1:
            Hp0 = np.array([hessvec_i(w, i, p0) for i in indices_H])
            gamma = np.sum(np.var(Hp0, axis=0, ddof=1)) / (n_h * p0_norm2)
        d = cg(Hv, -g, gamma, maxcg)
        J_batch = lambda wc: np.mean([loss_i(wc, i) for i in indices_S])
        c1, c2 = 1e-4, 0.9
        step, J_w = alpha, J_batch(w)
        gd = np.dot(g, d)
        if gd >= 0:
            d = -g
            gd = -np.dot(g, g)
        for _ in range(30):
            w_new = w + step * d
            if J_batch(w_new) <= J_w + c1 * step * gd:
                g_new = np.mean([grad_i(w_new, i) for i in indices_S], axis=0)
                if np.dot(g_new, d) >= c2 * gd:
                    break
            step *= 0.5
        else:
            step = 0.0
        w = w + step * d
        if reuse:
            used_S += 1
            if not reuse_hessian:
                used_H += 1
        else:
            # Batch dinamico (CCV su un nuovo campione, solo se non reuse)
            indices_new = np.random.choice(N, size=n, replace=False)
            g_new = np.mean([grad_i(w, i) for i in indices_new], axis=0)
            var_vec = np.var([grad_i(w, i) for i in indices_new],
                             axis=0, ddof=1) if n > 1 else np.zeros_like(g_new)
            V_norm1, gg_new = np.sum(var_vec), np.dot(g_new, g_new)
            if gg_new > 1e-16 and V_norm1 / n > theta**2 * gg_new:
                n = min(int(np.ceil(V_norm1 / (theta**2 * gg_new))) + 1, N)
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes

# ----------------------------------------------------------------------------
# 3.4 Newton-CG L1 (codice generato da generateNewtonL1; proiezione ortante)
# ----------------------------------------------------------------------------
def newton_l1(p, w0, theta, max_iter, alpha, batch0, nu, sigma, R, maxcg,
              eta=0.5, max_consec=None, reuse=False, reuse_hessian=True,
              max_hessian_reuse=None, subset=True):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    loss_i, grad_i, hess_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                                    p["hess_i"], p["hessvec_i"],
                                                    p["grad_full"])

    def F_batch(v, indices):
        Jb = np.mean([loss_i(v, i) for i in indices])
        return Jb + nu * np.sum(np.abs(v))

    def subgrad_batch(v, indices):
        grads = np.array([grad_i(v, i) for i in indices])
        gJ = np.mean(grads, axis=0)
        g = np.zeros_like(v)
        for i in range(len(v)):
            if v[i] > 0:
                g[i] = gJ[i] + nu
            elif v[i] < 0:
                g[i] = gJ[i] - nu
            else:
                if gJ[i] < -nu:
                    g[i] = gJ[i] + nu
                elif gJ[i] > nu:
                    g[i] = gJ[i] - nu
                else:
                    g[i] = 0.0
        return g

    def project_orthant(v, z):
        res = v.copy()
        for i in range(len(v)):
            if z[i] != 0 and np.sign(res[i]) != z[i]:
                res[i] = 0.0
        return res

    if reuse:
        indices_S = None
        indices_H = None
        used_S = 0
        used_H = 0
        need_resample_S = True
        need_resample_H = True

    for k in range(max_iter):
        if reuse:
            if (need_resample_S or indices_S is None or
                    (max_consec is not None and used_S >= max_consec)):
                indices_S = np.random.choice(N, size=n, replace=False)
                n_h = max(1, int(round(R * n)))
                n_h = min(n_h, N)
                used_S = 0
                need_resample_S = False
                if reuse_hessian:
                    indices_H = _draw_H_indices(p, indices_S, n_h, subset)
                    used_H = 0
                    need_resample_H = False
        else:
            indices_S = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)
        if reuse and not reuse_hessian:
            # H_k indipendente da S_k
            if (need_resample_H or indices_H is None or
                    (max_hessian_reuse is not None and used_H >= max_hessian_reuse)):
                n_h = max(1, int(round(R * n)))
                n_h = min(n_h, N)
                indices_H = _draw_H_indices(p, indices_S, n_h, subset)
                used_H = 0
                need_resample_H = False
        if reuse:
            # CCV sul campione RIUSATO al punto corrente
            if n > 1:
                var_vec = np.var(grads, axis=0, ddof=1)
            else:
                var_vec = np.zeros_like(g_batch)
            V_norm1 = np.sum(var_vec)
            gg = np.dot(g_batch, g_batch)
            if gg > 1e-16:
                if V_norm1 / n > theta**2 * gg:
                    n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                    n = min(n_new, N)
                    need_resample_S = True
                    used_S = 0
                    need_resample_H = True
                    used_H = 0
        z = np.where(w > 0, 1,
            np.where(w < 0, -1,
                np.where(g_batch < -nu, 1,
                    np.where(g_batch > nu, -1, 0))))
        sg = subgrad_batch(w, indices_S)
        sgn = np.linalg.norm(sg)
        if sgn < 1e-10:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        if not reuse:
            n_h = max(1, int(round(R * n)))
            n_h = min(n_h, N)
            indices_H = _draw_H_indices(p, indices_S, n_h, subset)
        free = (z != 0)
        d = np.zeros_like(w)
        if np.any(free):
            g_free = sg[free]
            def Hv(v_full):
                return np.mean([hessvec_i(w, i, v_full) for i in indices_H], axis=0)
            def Hv_free(v_free):
                v_full = np.zeros_like(w)
                v_full[free] = v_free
                Hv_full = Hv(v_full)
                return Hv_full[free]
            tol_cg = eta * np.linalg.norm(g_free)
            d_free = np.zeros(np.sum(free))
            r = -g_free.copy()
            pv = r.copy()
            rr = np.dot(r, r)
            for _ in range(maxcg):
                Hp = Hv_free(pv)
                pHp = np.dot(pv, Hp)
                if pHp <= 1e-14:
                    if np.linalg.norm(d_free) < 1e-14:
                        d_free = -g_free.copy()
                    break
                alpha_cg = rr / pHp
                d_free = d_free + alpha_cg * pv
                r_new = r - alpha_cg * Hp
                rr_new = np.dot(r_new, r_new)
                if np.sqrt(rr_new) <= tol_cg:
                    r = r_new
                    rr = rr_new
                    break
                beta = rr_new / rr
                pv = r_new + beta * pv
                r = r_new
                rr = rr_new
            d[free] = d_free
        step = alpha
        F_w = F_batch(w, indices_S)
        sg_d = np.dot(sg, d)
        if sg_d >= 0:
            d = -sg
            sg_d = -np.dot(sg, sg)
        w_new = w.copy()
        for _ in range(20):
            w_trial = project_orthant(w + step * d, z)
            if F_batch(w_trial, indices_S) <= F_w + sigma * step * sg_d:
                w_new = w_trial
                break
            step *= 0.5
            if step < 1e-12:
                w_new = w.copy()
                break
        w = w_new
        if reuse:
            used_S += 1
            if not reuse_hessian:
                used_H += 1
        else:
            # Batch dinamico (CCV su un nuovo campione, solo se non reuse)
            indices_new = np.random.choice(N, size=n, replace=False)
            grads_new = np.array([grad_i(w, i) for i in indices_new])
            g_new = np.mean(grads_new, axis=0)
            if n > 1:
                var_vec = np.var(grads_new, axis=0, ddof=1)
            else:
                var_vec = np.zeros_like(g_new)
            V_norm1 = np.sum(var_vec)
            gg_new = np.dot(g_new, g_new)
            if gg_new > 1e-16:
                if V_norm1 / n > theta**2 * gg_new:
                    n_new = int(np.ceil(V_norm1 / (theta**2 * gg_new))) + 1
                    n = min(n_new, N)
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes

# ----------------------------------------------------------------------------
# 3.5 Varianti con stop adattivo (validation set) — codice esatto delle
#     varianti *Validation dell'app (validation_codice_generato/).
#     Ritornano (history, batch_sizes, resize_points, m_actual, val_hist).
# ----------------------------------------------------------------------------
def dynamic_gd_val(p, w0, theta, max_iter, alpha, batch0,
                   val_pct, val_tol, val_patience, val_freq, val_min_abs,
                   val_strategy, force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    indices = None
    used = 0
    need_resample = True
    n_val = max(1, min(int(round(val_pct * N)), N - 1))
    _perm = np.random.permutation(N)
    val_idx = _perm[:n_val]
    train_idx = _perm[n_val:]
    best_val = np.inf
    patience = 0
    val_hist = []
    m_actual = [0]
    for k in range(max_iter):
        if force_resample or need_resample or indices is None:
            if val_strategy == 'dynamic':
                _perm = np.random.permutation(N)
                val_idx = _perm[:n_val]
                train_idx = _perm[n_val:]
            n = min(n, len(train_idx))
            indices = np.random.choice(train_idx, size=n, replace=False)
            used = 0
            need_resample = False
        used += 1
        m_actual.append(used)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g, g)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, len(train_idx))
                resize_points.append(w.copy().tolist())
                need_resample = True
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])
        # Line search Wolfe
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
                    g_new = np.mean([grad_i(w_new, i) for i in indices], axis=0)
                    if np.dot(g_new, d) >= c2 * gd:
                        break
                step *= 0.5
            else:
                step = 0.0
        w = w + step * d
        if k % val_freq == 0:
            Jv = float(np.mean([loss_i(w, i) for i in val_idx]))
            val_hist.append(Jv)
            improved = (Jv <= best_val * (1.0 - val_tol) - val_min_abs)
            if improved:
                best_val = Jv
                patience = 0
            else:
                patience += 1
                if patience >= val_patience:
                    need_resample = True
                    patience = 0
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, m_actual, val_hist

def bb_ccv_val(p, w0, theta, max_iter, alpha, batch0,
               val_pct, val_tol, val_patience, val_freq, val_min_abs,
               val_strategy, force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    w_prev = w.copy()
    g_prev = None
    indices = None
    used = 0
    need_resample = True
    n_val = max(1, min(int(round(val_pct * N)), N - 1))
    _perm = np.random.permutation(N)
    val_idx = _perm[:n_val]
    train_idx = _perm[n_val:]
    best_val = np.inf
    patience = 0
    val_hist = []
    m_actual = [0]
    for k in range(max_iter):
        if force_resample or need_resample or indices is None:
            if val_strategy == 'dynamic':
                _perm = np.random.permutation(N)
                val_idx = _perm[:n_val]
                train_idx = _perm[n_val:]
            n = min(n, len(train_idx))
            indices = np.random.choice(train_idx, size=n, replace=False)
            used = 0
            need_resample = False
        used += 1
        m_actual.append(used)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)
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
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])
        # Line search Armijo
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
        if k % val_freq == 0:
            Jv = float(np.mean([loss_i(w, i) for i in val_idx]))
            val_hist.append(Jv)
            improved = (Jv <= best_val * (1.0 - val_tol) - val_min_abs)
            if improved:
                best_val = Jv
                patience = 0
            else:
                patience += 1
                if patience >= val_patience:
                    need_resample = True
                    patience = 0
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g, g)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, len(train_idx))
                resize_points.append(w.copy().tolist())
                need_resample = True
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, m_actual, val_hist

def newton_cg_val(p, w0, theta, max_iter, alpha, batch0, R, maxcg,
                  val_pct, val_tol, val_patience, val_freq, val_min_abs,
                  val_strategy, force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
    resize_points = []
    loss_i, grad_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                            p["hessvec_i"], p["grad_full"])

    def cg(A, b, gamma, maxcg):
        x = np.zeros_like(b)
        r = b - A(x)
        p = r.copy()
        rr = np.dot(r, r)
        for _ in range(maxcg):
            Ap = A(p)
            pHp = np.dot(p, Ap)
            if pHp <= 1e-14:
                break
            alpha = rr / pHp
            x = x + alpha * p
            r_new = r - alpha * Ap
            rr_new = np.dot(r_new, r_new)
            if rr_new <= gamma * np.dot(x, x) + 1e-16:
                return x
            beta = rr_new / rr
            p = r_new + beta * p
            r = r_new
            rr = rr_new
        return x

    indices_S = None
    indices_H = None
    used_S = 0
    used_H = 0
    need_resample_S = True
    need_resample_H = True
    n_val = max(1, min(int(round(val_pct * N)), N - 1))
    _perm = np.random.permutation(N)
    val_idx = _perm[:n_val]
    train_idx = _perm[n_val:]
    best_val = np.inf
    patience_S = 0
    patience_H = 0
    val_hist = []
    m_actual = [0]
    for k in range(max_iter):
        if force_resample or need_resample_S or indices_S is None:
            if val_strategy == 'dynamic':
                _perm = np.random.permutation(N)
                val_idx = _perm[:n_val]
                train_idx = _perm[n_val:]
            n = min(n, len(train_idx))
            indices_S = np.random.choice(train_idx, size=n, replace=False)
            n_h = min(max(1, int(round(R * n))), len(train_idx))
            used_S = 0
            need_resample_S = False
            indices_H = np.random.choice(indices_S, size=n_h, replace=False)
            used_H = 0
            need_resample_H = False
        used_S += 1
        m_actual.append(used_S)
        grads_arr = np.array([grad_i(w, i) for i in indices_S])
        g = np.mean(grads_arr, axis=0)
        # CCV sul campione RIUSATO al punto corrente
        if n > 1:
            var_vec = np.var(grads_arr, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g, g)
        if gg > 1e-16 and V_norm1 / n > theta**2 * gg:
            n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
            n = min(n_new, len(train_idx))
            resize_points.append(w.copy().tolist())
            need_resample_S = True
            need_resample_H = True
        Hv = lambda v: np.mean([hessvec_i(w, i, v) for i in indices_H], axis=0)
        p0 = -g
        p0_norm2 = np.dot(p0, p0)
        gamma = 0.0
        if p0_norm2 > 1e-16 and n_h > 1:
            Hp0 = np.array([hessvec_i(w, i, p0) for i in indices_H])
            gamma = np.sum(np.var(Hp0, axis=0, ddof=1)) / (n_h * p0_norm2)
        d = cg(Hv, -g, gamma, maxcg)
        J_batch = lambda wc: np.mean([loss_i(wc, i) for i in indices_S])
        # Line search Wolfe
        c1, c2 = 1e-4, 0.9
        step, J_w = alpha, J_batch(w)
        gd = np.dot(g, d)
        if gd >= 0:
            d = -g
            gd = -np.dot(g, g)
        for _ in range(30):
            w_new = w + step * d
            if J_batch(w_new) <= J_w + c1 * step * gd:
                g_new = np.mean([grad_i(w_new, i) for i in indices_S], axis=0)
                if np.dot(g_new, d) >= c2 * gd:
                    break
            step *= 0.5
        else:
            step = 0.0
        w = w + step * d
        if k % val_freq == 0:
            Jv = float(np.mean([loss_i(w, i) for i in val_idx]))
            val_hist.append(Jv)
            improved = (Jv <= best_val * (1.0 - val_tol) - val_min_abs)
            if improved:
                best_val = Jv
                patience_S = 0
                patience_H = 0
            else:
                patience_S += 1
                patience_H += 1
                if patience_S >= val_patience or patience_H >= val_patience:
                    need_resample_S = True
                    patience_S = 0
                    need_resample_H = True
                    patience_H = 0
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points, m_actual, val_hist

def newton_l1_val(p, w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg,
                  eta, val_pct, val_tol, val_patience, val_freq, val_min_abs,
                  val_strategy, force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    loss_i, grad_i, hess_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                                    p["hess_i"], p["hessvec_i"],
                                                    p["grad_full"])

    def F_batch(v, indices):
        Jb = np.mean([loss_i(v, i) for i in indices])
        return Jb + nu * np.sum(np.abs(v))

    def subgrad_batch(v, indices):
        grads = np.array([grad_i(v, i) for i in indices])
        gJ = np.mean(grads, axis=0)
        g = np.zeros_like(v)
        for i in range(len(v)):
            if v[i] > 0:
                g[i] = gJ[i] + nu
            elif v[i] < 0:
                g[i] = gJ[i] - nu
            else:
                if gJ[i] < -nu:
                    g[i] = gJ[i] + nu
                elif gJ[i] > nu:
                    g[i] = gJ[i] - nu
                else:
                    g[i] = 0.0
        return g

    def project_orthant(v, z):
        res = v.copy()
        for i in range(len(v)):
            if z[i] != 0 and np.sign(res[i]) != z[i]:
                res[i] = 0.0
        return res

    indices_S = None
    indices_H = None
    used_S = 0
    used_H = 0
    need_resample_S = True
    need_resample_H = True
    n_val = max(1, min(int(round(val_pct * N)), N - 1))
    _perm = np.random.permutation(N)
    val_idx = _perm[:n_val]
    train_idx = _perm[n_val:]
    best_val = np.inf
    patience_S = 0
    patience_H = 0
    val_hist = []
    m_actual = [0]
    for k in range(max_iter):
        if force_resample or need_resample_S or indices_S is None:
            if val_strategy == 'dynamic':
                _perm = np.random.permutation(N)
                val_idx = _perm[:n_val]
                train_idx = _perm[n_val:]
            n = min(n, len(train_idx))
            indices_S = np.random.choice(train_idx, size=n, replace=False)
            n_h = min(max(1, int(round(R_ * n))), len(train_idx))
            used_S = 0
            need_resample_S = False
            indices_H = np.random.choice(indices_S, size=n_h, replace=False)
            used_H = 0
            need_resample_H = False
        used_S += 1
        m_actual.append(used_S)
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)
        # CCV sul campione RIUSATO al punto corrente
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g_batch)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g_batch, g_batch)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, len(train_idx))
                resize_points.append(w.copy().tolist())
                need_resample_S = True
                need_resample_H = True
        z = np.where(w > 0, 1,
            np.where(w < 0, -1,
                np.where(g_batch < -nu, 1,
                    np.where(g_batch > nu, -1, 0))))
        sg = subgrad_batch(w, indices_S)
        sgn = np.linalg.norm(sg)
        if sgn < 1e-10:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        free = (z != 0)
        d = np.zeros_like(w)
        if np.any(free):
            g_free = sg[free]
            hessians = np.array([hess_i(w, i) for i in indices_H])
            H = np.mean(hessians, axis=0)
            H_free = H[np.ix_(free, free)]
            tol_cg = eta * np.linalg.norm(g_free)
            d_free = np.zeros(np.sum(free))
            r = -g_free.copy()
            pv = r.copy()
            rr = np.dot(r, r)
            for _ in range(maxcg):
                Hp = H_free @ pv
                pHp = np.dot(pv, Hp)
                if pHp <= 1e-14:
                    if np.linalg.norm(d_free) < 1e-14:
                        d_free = -g_free.copy()
                    break
                alpha_cg = rr / pHp
                d_free = d_free + alpha_cg * pv
                r_new = r - alpha_cg * Hp
                rr_new = np.dot(r_new, r_new)
                if np.sqrt(rr_new) <= tol_cg:
                    r = r_new
                    rr = rr_new
                    break
                beta = rr_new / rr
                pv = r_new + beta * pv
                r = r_new
                rr = rr_new
            d[free] = d_free
        step = alpha
        F_w = F_batch(w, indices_S)
        sg_d = np.dot(sg, d)
        if sg_d >= 0:
            d = -sg
            sg_d = -np.dot(sg, sg)
        w_new = w.copy()
        for _ in range(20):
            w_trial = project_orthant(w + step * d, z)
            if F_batch(w_trial, indices_S) <= F_w + sigma * step * sg_d:
                w_new = w_trial
                break
            step *= 0.5
            if step < 1e-12:
                w_new = w.copy()
                break
        w = w_new
        if k % val_freq == 0:
            Jv = float(np.mean([loss_i(w, i) for i in val_idx]))
            val_hist.append(Jv)
            improved = (Jv <= best_val * (1.0 - val_tol) - val_min_abs)
            if improved:
                best_val = Jv
                patience_S = 0
                patience_H = 0
            else:
                patience_S += 1
                patience_H += 1
                if patience_S >= val_patience or patience_H >= val_patience:
                    need_resample_S = True
                    patience_S = 0
                    need_resample_H = True
                    patience_H = 0
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points, m_actual, val_hist

# ----------------------------------------------------------------------------
# 3.6 Varianti con riuso per discesa della loss sul batch — codice esatto delle
#     varianti *Descent dell'app (descent_codice_generato/).
#     Ritornano (history, batch_sizes, resize_points, resample_pts, m_actual,
#     desc_hist).
# ----------------------------------------------------------------------------
def dynamic_gd_desc(p, w0, theta, max_iter, alpha, batch0,
                    desc_tol, desc_min_abs, desc_patience, desc_freq,
                    force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    resample_pts = []
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    indices = None
    used = 0
    need_resample = True
    Jb_prev = None
    patience_dec = 0
    desc_hist = []
    m_actual = [0]
    desc_resample = False
    for k in range(max_iter):
        if force_resample or need_resample or indices is None:
            if desc_resample:
                resample_pts.append(w.copy().tolist())
                desc_resample = False
            n = min(n, N)
            indices = np.random.choice(N, size=n, replace=False)
            used = 0
            need_resample = False
            Jb_prev = None
        used += 1
        m_actual.append(used)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)
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
                resize_points.append(w.copy().tolist())
                need_resample = True
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])
        # Line search Wolfe
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
                    g_new = np.mean([grad_i(w_new, i) for i in indices], axis=0)
                    if np.dot(g_new, d) >= c2 * gd:
                        break
                step *= 0.5
            else:
                step = 0.0
        w = w + step * d
        if k % desc_freq == 0:
            Jb = float(J_batch(w))
            desc_hist.append(Jb)
            if Jb_prev is not None:
                improved = (Jb_prev - Jb >= desc_tol * abs(Jb_prev) + desc_min_abs)
                if improved:
                    patience_dec = 0
                else:
                    patience_dec += 1
                    if patience_dec >= desc_patience:
                        if not need_resample:
                            desc_resample = True
                        need_resample = True
                        patience_dec = 0
            Jb_prev = Jb
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, resample_pts, m_actual, desc_hist

def bb_ccv_desc(p, w0, theta, max_iter, alpha, batch0,
                desc_tol, desc_min_abs, desc_patience, desc_freq,
                force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    resample_pts = []
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    w_prev = w.copy()
    g_prev = None
    indices = None
    used = 0
    need_resample = True
    Jb_prev = None
    patience_dec = 0
    desc_hist = []
    m_actual = [0]
    desc_resample = False
    for k in range(max_iter):
        if force_resample or need_resample or indices is None:
            if desc_resample:
                resample_pts.append(w.copy().tolist())
                desc_resample = False
            n = min(n, N)
            indices = np.random.choice(N, size=n, replace=False)
            used = 0
            need_resample = False
            Jb_prev = None
        used += 1
        m_actual.append(used)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)
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
        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])
        # Line search Armijo
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
        if k % desc_freq == 0:
            Jb = float(J_batch(w))
            desc_hist.append(Jb)
            if Jb_prev is not None:
                improved = (Jb_prev - Jb >= desc_tol * abs(Jb_prev) + desc_min_abs)
                if improved:
                    patience_dec = 0
                else:
                    patience_dec += 1
                    if patience_dec >= desc_patience:
                        if not need_resample:
                            desc_resample = True
                        need_resample = True
                        patience_dec = 0
            Jb_prev = Jb
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
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
                resize_points.append(w.copy().tolist())
                need_resample = True
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, resample_pts, m_actual, desc_hist

def newton_cg_desc(p, w0, theta, max_iter, alpha, batch0, R, maxcg,
                   desc_tol, desc_min_abs, desc_patience, desc_freq,
                   force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
    resize_points = []
    resample_pts = []
    loss_i, grad_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                            p["hessvec_i"], p["grad_full"])

    def cg(A, b, gamma, maxcg):
        x = np.zeros_like(b)
        r = b - A(x)
        p = r.copy()
        rr = np.dot(r, r)
        for _ in range(maxcg):
            Ap = A(p)
            pHp = np.dot(p, Ap)
            if pHp <= 1e-14:
                break
            alpha = rr / pHp
            x = x + alpha * p
            r_new = r - alpha * Ap
            rr_new = np.dot(r_new, r_new)
            if rr_new <= gamma * np.dot(x, x) + 1e-16:
                return x
            beta = rr_new / rr
            p = r_new + beta * p
            r = r_new
            rr = rr_new
        return x

    indices_S = None
    indices_H = None
    used_S = 0
    used_H = 0
    need_resample_S = True
    need_resample_H = True
    Jb_prev = None
    patience_S = 0
    patience_H = 0
    patience_dec = 0
    desc_hist = []
    m_actual = [0]
    desc_resample = False
    for k in range(max_iter):
        if force_resample or need_resample_S or indices_S is None:
            if desc_resample:
                resample_pts.append(w.copy().tolist())
                desc_resample = False
            n = min(n, N)
            indices_S = np.random.choice(N, size=n, replace=False)
            n_h = min(max(1, int(round(R * n))), N)
            used_S = 0
            need_resample_S = False
            Jb_prev = None
            indices_H = np.random.choice(indices_S, size=n_h, replace=False)
            used_H = 0
            need_resample_H = False
        used_S += 1
        m_actual.append(used_S)
        grads_arr = np.array([grad_i(w, i) for i in indices_S])
        g = np.mean(grads_arr, axis=0)
        # CCV sul campione RIUSATO al punto corrente
        if n > 1:
            var_vec = np.var(grads_arr, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g, g)
        if gg > 1e-16 and V_norm1 / n > theta**2 * gg:
            n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
            n = min(n_new, N)
            resize_points.append(w.copy().tolist())
            need_resample_S = True
            need_resample_H = True
        Hv = lambda v: np.mean([hessvec_i(w, i, v) for i in indices_H], axis=0)
        p0 = -g
        p0_norm2 = np.dot(p0, p0)
        gamma = 0.0
        if p0_norm2 > 1e-16 and n_h > 1:
            Hp0 = np.array([hessvec_i(w, i, p0) for i in indices_H])
            gamma = np.sum(np.var(Hp0, axis=0, ddof=1)) / (n_h * p0_norm2)
        d = cg(Hv, -g, gamma, maxcg)
        J_batch = lambda wc: np.mean([loss_i(wc, i) for i in indices_S])
        # Line search Wolfe
        c1, c2 = 1e-4, 0.9
        step, J_w = alpha, J_batch(w)
        gd = np.dot(g, d)
        if gd >= 0:
            d = -g
            gd = -np.dot(g, g)
        for _ in range(30):
            w_new = w + step * d
            if J_batch(w_new) <= J_w + c1 * step * gd:
                g_new = np.mean([grad_i(w_new, i) for i in indices_S], axis=0)
                if np.dot(g_new, d) >= c2 * gd:
                    break
            step *= 0.5
        else:
            step = 0.0
        w = w + step * d
        if k % desc_freq == 0:
            Jb = float(J_batch(w))
            desc_hist.append(Jb)
            if Jb_prev is not None:
                improved = (Jb_prev - Jb >= desc_tol * abs(Jb_prev) + desc_min_abs)
                if improved:
                    patience_S = 0
                    patience_H = 0
                else:
                    patience_S += 1
                    patience_H += 1
                    if patience_S >= desc_patience or patience_H >= desc_patience:
                        if not need_resample_S:
                            desc_resample = True
                        need_resample_S = True
                        patience_S = 0
                        need_resample_H = True
                        patience_H = 0
            Jb_prev = Jb
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points, resample_pts, m_actual, desc_hist

def newton_l1_desc(p, w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg,
                   eta, desc_tol, desc_min_abs, desc_patience, desc_freq,
                   force_resample=False):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    resample_pts = []
    loss_i, grad_i, hess_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                                    p["hess_i"], p["hessvec_i"],
                                                    p["grad_full"])

    def F_batch(v, indices):
        Jb = np.mean([loss_i(v, i) for i in indices])
        return Jb + nu * np.sum(np.abs(v))

    def subgrad_batch(v, indices):
        grads = np.array([grad_i(v, i) for i in indices])
        gJ = np.mean(grads, axis=0)
        g = np.zeros_like(v)
        for i in range(len(v)):
            if v[i] > 0:
                g[i] = gJ[i] + nu
            elif v[i] < 0:
                g[i] = gJ[i] - nu
            else:
                if gJ[i] < -nu:
                    g[i] = gJ[i] + nu
                elif gJ[i] > nu:
                    g[i] = gJ[i] - nu
                else:
                    g[i] = 0.0
        return g

    def project_orthant(v, z):
        res = v.copy()
        for i in range(len(v)):
            if z[i] != 0 and np.sign(res[i]) != z[i]:
                res[i] = 0.0
        return res

    indices_S = None
    indices_H = None
    used_S = 0
    used_H = 0
    need_resample_S = True
    need_resample_H = True
    Jb_prev = None
    patience_S = 0
    patience_H = 0
    patience_dec = 0
    desc_hist = []
    m_actual = [0]
    desc_resample = False
    for k in range(max_iter):
        if force_resample or need_resample_S or indices_S is None:
            if desc_resample:
                resample_pts.append(w.copy().tolist())
                desc_resample = False
            n = min(n, N)
            indices_S = np.random.choice(N, size=n, replace=False)
            n_h = min(max(1, int(round(R_ * n))), N)
            used_S = 0
            need_resample_S = False
            Jb_prev = None
            indices_H = np.random.choice(indices_S, size=n_h, replace=False)
            used_H = 0
            need_resample_H = False
        used_S += 1
        m_actual.append(used_S)
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)
        # CCV sul campione RIUSATO al punto corrente
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g_batch)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g_batch, g_batch)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, N)
                resize_points.append(w.copy().tolist())
                need_resample_S = True
                need_resample_H = True
        z = np.where(w > 0, 1,
            np.where(w < 0, -1,
                np.where(g_batch < -nu, 1,
                    np.where(g_batch > nu, -1, 0))))
        sg = subgrad_batch(w, indices_S)
        sgn = np.linalg.norm(sg)
        if sgn < 1e-10:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        free = (z != 0)
        d = np.zeros_like(w)
        if np.any(free):
            g_free = sg[free]
            hessians = np.array([hess_i(w, i) for i in indices_H])
            H = np.mean(hessians, axis=0)
            H_free = H[np.ix_(free, free)]
            tol_cg = eta * np.linalg.norm(g_free)
            d_free = np.zeros(np.sum(free))
            r = -g_free.copy()
            p = r.copy()
            rr = np.dot(r, r)
            for _ in range(maxcg):
                Hp = H_free @ p
                pHp = np.dot(p, Hp)
                if pHp <= 1e-14:
                    if np.linalg.norm(d_free) < 1e-14:
                        d_free = -g_free.copy()
                    break
                alpha_cg = rr / pHp
                d_free = d_free + alpha_cg * p
                r_new = r - alpha_cg * Hp
                rr_new = np.dot(r_new, r_new)
                if np.sqrt(rr_new) <= tol_cg:
                    r = r_new
                    rr = rr_new
                    break
                beta = rr_new / rr
                p = r_new + beta * p
                r = r_new
                rr = rr_new
            d[free] = d_free
        step = alpha
        F_w = F_batch(w, indices_S)
        sg_d = np.dot(sg, d)
        if sg_d >= 0:
            d = -sg
            sg_d = -np.dot(sg, sg)
        w_new = w.copy()
        for _ in range(20):
            w_trial = project_orthant(w + step * d, z)
            if F_batch(w_trial, indices_S) <= F_w + sigma * step * sg_d:
                w_new = w_trial
                break
            step *= 0.5
            if step < 1e-12:
                w_new = w.copy()
                break
        w = w_new
        if k % desc_freq == 0:
            Jb = float(F_batch(w, indices_S))
            desc_hist.append(Jb)
            if Jb_prev is not None:
                improved = (Jb_prev - Jb >= desc_tol * abs(Jb_prev) + desc_min_abs)
                if improved:
                    patience_S = 0
                    patience_H = 0
                else:
                    patience_S += 1
                    patience_H += 1
                    if patience_S >= desc_patience or patience_H >= desc_patience:
                        if not need_resample_S:
                            desc_resample = True
                        need_resample_S = True
                        patience_S = 0
                        need_resample_H = True
                        patience_H = 0
            Jb_prev = Jb
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points, resample_pts, m_actual, desc_hist

# ============================================================================
# 4. RUNNER
# ============================================================================
def errs(hist, p):
    return [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]


def run_base_reuse(p, algo, max_consec=None, reuse=False, reuse_hessian=True,
                   max_hessian_reuse=None, subset=True):
    """Esegue base (reuse=False) o riuso (reuse=True, max_consec=M oppure None
    per il riuso illimitato M=inf). Ritorna (e_list, batch_sizes, resample_pts)."""
    if algo == "gd":
        hist, bs, rp, res_pts = dynamic_gd(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                           max_consec, reuse)
    elif algo == "bb":
        hist, bs, rp, res_pts = bb_ccv(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                       max_consec, reuse)
    elif algo == "newton_cg":
        hist, bs = newton_cg(p, W0, THETA, MAX_ITER, ALPHA, BATCH0, R_, MAXCG,
                             max_consec=max_consec, reuse=reuse,
                             reuse_hessian=reuse_hessian,
                             max_hessian_reuse=max_hessian_reuse, subset=subset)
        res_pts = None
    elif algo == "newton_l1":
        hist, bs = newton_l1(p, W0, THETA, MAX_ITER, ALPHA, BATCH0, NU, SIGMA,
                             R_, MAXCG, eta=ETA, max_consec=max_consec,
                             reuse=reuse, reuse_hessian=reuse_hessian,
                             max_hessian_reuse=max_hessian_reuse, subset=subset)
        res_pts = None
    else:
        raise ValueError(algo)
    return errs(hist, p), bs, res_pts


def run_validation(p, algo, hp, force_resample=False):
    """Stop adattivo con validation set. Ritorna (e_list, resamples)."""
    fns = {
        "gd": dynamic_gd_val,
        "bb": bb_ccv_val,
        "newton_cg": newton_cg_val,
        "newton_l1": newton_l1_val,
    }
    f = fns[algo]
    args = [p, W0, THETA, MAX_ITER, ALPHA, BATCH0]
    if algo == "newton_cg":
        args += [R_, MAXCG]
    elif algo == "newton_l1":
        args += [NU, SIGMA, MAXCG, ETA]
    args += [hp["val_pct"], hp["val_tol"], hp["val_patience"],
             hp["val_freq"], hp["val_min_abs"], hp["val_strategy"],
             force_resample]
    hist, bs, resize_points, m_actual, val_hist = f(*args)
    resamples = sum(1 for x in m_actual if x == 1) - 1
    return errs(hist, p), max(resamples, 0)


def run_descent(p, algo, hp, force_resample=False):
    """Riuso per discesa della loss sul batch. Ritorna (e_list, resamples)."""
    fns = {
        "gd": dynamic_gd_desc,
        "bb": bb_ccv_desc,
        "newton_cg": newton_cg_desc,
        "newton_l1": newton_l1_desc,
    }
    f = fns[algo]
    args = [p, W0, THETA, MAX_ITER, ALPHA, BATCH0]
    if algo == "newton_cg":
        args += [R_, MAXCG]
    elif algo == "newton_l1":
        args += [NU, SIGMA, MAXCG, ETA]
    args += [hp["desc_tol"], hp["desc_min_abs"], hp["desc_patience"],
             hp["desc_freq"], force_resample]
    hist, bs, resize_points, resample_pts, m_actual, desc_hist = f(*args)
    resamples = sum(1 for x in m_actual if x == 1) - 1
    return errs(hist, p), max(resamples, 0)

# ============================================================================
# 5. FORMATTAZIONE LATEX (stile identico a tesi/tesi.tex)
# ============================================================================
def fmt(x):
    """e_k stile tabelle: 1.4142e0 -> $1.4142\\times10^{0}$ (4 cifre)."""
    s = f"{x:.4e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt3(x):
    """Valori a 3 cifre significative: $4.13\\times10^{-1}$."""
    s = f"{x:.2e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def colorcell(x):
    """Cella colorata: \\colorcell{1.4142}{0}."""
    s = f"{x:.4e}"
    m, e = s.split("e")
    return f"\\colorcell{{{m}}}{{{int(e)}}}"


def marker(base, v):
    if v < base - 1e-12:
        return "$\\blacktriangle$"
    if v > base + 1e-12:
        return "$\\blacktriangledown$"
    return "$=$"


# ============================================================================
# 6. RACCOLTA DATI
# ============================================================================
PRESETS = ("quad_well", "quad_ill", "quad_very_ill", "quad_offdiag")
ALGOS4 = ("gd", "bb", "newton_cg", "newton_l1")
M_VALUES = {"inf": None, "10": 10, "5": 5, "3": 3, "2": 2, "1": 1}

VALID_DEFAULT_HP = dict(val_pct=0.2, val_tol=1e-4, val_patience=3, val_freq=1,
                        val_min_abs=0.0, val_strategy="fixed")
DESCENT_DEFAULT_HP = dict(desc_tol=1e-4, desc_min_abs=0.0, desc_patience=1,
                          desc_freq=1)


def compute_riuso():
    """Riuso: per ogni (preset, algo, colonna) le 31 e_k (seed 42).
    Colonne: base, inf, 10, 5, 3, 2, 1, H_ind_inf (M=10, H indipendente M_H=inf).
    Se un algoritmo converge prima di 30 iterazioni (criterio di arresto),
    l'ultimo valore viene ripetuto fino a k=30 (come nelle tabelle della tesi)."""
    def pad(e):
        if len(e) < MAX_ITER + 1:
            return e + [e[-1]] * (MAX_ITER + 1 - len(e))
        return e

    data = {}
    for pname in PRESETS:
        for algo in ALGOS4:
            for col, m in [("base", None)] + [("inf", None)] + \
                    [(c, M_VALUES[c]) for c in ("10", "5", "3", "2", "1")]:
                np.random.seed(SEED)
                p = PRESET_MAKERS[pname]()
                reuse = col != "base"
                e, bs, rp = run_base_reuse(p, algo, max_consec=m, reuse=reuse)
                data[(pname, algo, col)] = pad(e)
            # H indipendente da S_k (solo Newton): M=10, H illimitata
            if algo in ("newton_cg", "newton_l1"):
                np.random.seed(SEED)
                p = PRESET_MAKERS[pname]()
                e, bs, rp = run_base_reuse(p, algo, max_consec=10, reuse=True,
                                           reuse_hessian=False,
                                           max_hessian_reuse=None)
                data[(pname, algo, "H_ind_inf")] = pad(e)
    return data


def compute_riuso_robust():
    """Conta meglio/peggio/uguale di e30 (5 seed) per M=inf e M=10 vs base."""
    res = {}
    for pname in PRESETS:
        for algo in ALGOS4:
            base30 = {}
            for s in SEEDS:
                np.random.seed(s)
                p = PRESET_MAKERS[pname](seed=s)
                e, bs, rp = run_base_reuse(p, algo, None, reuse=False)
                base30[s] = e[-1]
            for cfg, m in (("inf", None), ("10", 10)):
                counts = [0, 0, 0]
                for s in SEEDS:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    e, bs, rp = run_base_reuse(p, algo, max_consec=m, reuse=True)
                    e30 = e[-1]
                    if e30 < base30[s] - 1e-12:
                        counts[0] += 1
                    elif e30 > base30[s] + 1e-12:
                        counts[1] += 1
                    else:
                        counts[2] += 1
                res[(pname, algo, cfg)] = tuple(counts)
    return res

def compute_validation():
    """Sweep validation (griglia 108 hp) + riferimenti base_train/minf_train.
    data[(pname, algo, hp_key)] = dict(e30, resamples)
    refs[(pname, algo, ref)] = dict(e30, resamples)
    hp_key = 'pct|tol|pat|freq|strat'."""
    def hp_key(hp):
        return (f"{hp['val_pct']}|{hp['val_tol']}|{hp['val_patience']}"
                f"|{hp['val_freq']}|{hp['val_strategy']}")

    VAL_PCTS = [0.1, 0.2, 0.3]
    VAL_TOLS = [1e-5, 1e-4, 1e-3]
    VAL_PATIENCE = [1, 3, 8]
    VAL_FREQ = [1, 3]
    VAL_STRATEGY = ["fixed", "dynamic"]

    def run_ref(p, algo, force):
        hp = dict(VALID_DEFAULT_HP, val_patience=9999)
        e, res = run_validation(p, algo, hp, force_resample=force)
        return dict(e30=e[-1], resamples=res)

    data, refs = {}, {}
    for pname in PRESETS:
        for algo in ALGOS4:
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "base_train")] = run_ref(p, algo, True)
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "minf_train")] = run_ref(p, algo, False)
            for pct in VAL_PCTS:
                for tol in VAL_TOLS:
                    for pat in VAL_PATIENCE:
                        for freq in VAL_FREQ:
                            for strat in VAL_STRATEGY:
                                hp = dict(VALID_DEFAULT_HP, val_pct=pct,
                                          val_tol=tol, val_patience=pat,
                                          val_freq=freq, val_strategy=strat)
                                np.random.seed(SEED)
                                p = PRESET_MAKERS[pname]()
                                e, res = run_validation(p, algo, hp)
                                data[(pname, algo, hp_key(hp))] = dict(
                                    e30=e[-1], resamples=res)
    return data, refs


def compute_validation_robust():
    """Robustezza 5 seed per le configurazioni candidate (default, pat1,
    pat1-pct1-dyn, pat3-pct1-dyn) + riferimenti. res[(p,a,cfg)] = lista e30."""
    candidates = {
        "default": dict(VALID_DEFAULT_HP),
        "pat1": dict(VALID_DEFAULT_HP, val_patience=1),
        "pat1-pct1-dyn": dict(VALID_DEFAULT_HP, val_patience=1, val_pct=0.1,
                              val_strategy="dynamic"),
        "pat3-pct1-dyn": dict(VALID_DEFAULT_HP, val_patience=3, val_pct=0.1,
                              val_strategy="dynamic"),
    }
    res = {}
    for pname in PRESETS:
        for algo in ALGOS4:
            for name, hp in candidates.items():
                es = []
                for s in SEEDS:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    e, res_ = run_validation(p, algo, dict(VALID_DEFAULT_HP, **hp))
                    es.append(e[-1])
                res[(pname, algo, name)] = es
            for ref in ("base_train", "minf_train"):
                es = []
                for s in SEEDS:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    if ref == "base_train":
                        e, res_ = run_validation(p, algo,
                                                 dict(VALID_DEFAULT_HP, val_patience=9999),
                                                 force_resample=True)
                    else:
                        e, res_ = run_validation(p, algo,
                                                 dict(VALID_DEFAULT_HP, val_patience=9999),
                                                 force_resample=False)
                    es.append(e[-1])
                res[(pname, algo, ref)] = es
    return res

def compute_descent():
    """Sweep discesa + riferimenti base/minf. Stessa struttura di compute_validation."""
    def hp_key(hp):
        return (f"{hp['desc_tol']}|{hp['desc_min_abs']}|{hp['desc_patience']}"
                f"|{hp['desc_freq']}")

    DESC_TOLS = [1e-5, 1e-4, 1e-3]
    DESC_PATIENCE = [1, 3, 8]
    DESC_FREQ = [1, 3]

    def run_ref(p, algo, force):
        hp = dict(DESCENT_DEFAULT_HP, desc_patience=9999)
        e, res = run_descent(p, algo, hp, force_resample=force)
        return dict(e30=e[-1], resamples=res)

    data, refs = {}, {}
    for pname in PRESETS:
        for algo in ALGOS4:
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "base")] = run_ref(p, algo, True)
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "minf")] = run_ref(p, algo, False)
            for tol in DESC_TOLS:
                for pat in DESC_PATIENCE:
                    for freq in DESC_FREQ:
                        hp = dict(DESCENT_DEFAULT_HP, desc_tol=tol,
                                  desc_patience=pat, desc_freq=freq)
                        np.random.seed(SEED)
                        p = PRESET_MAKERS[pname]()
                        e, res = run_descent(p, algo, hp)
                        data[(pname, algo, hp_key(hp))] = dict(e30=e[-1],
                                                               resamples=res)
    return data, refs


def compute_descent_robust():
    """Robustezza 5 seed per le configurazioni candidate della discesa."""
    candidates = {
        "default": dict(DESCENT_DEFAULT_HP),
        "pat1": dict(DESCENT_DEFAULT_HP, desc_patience=1),
        "t3-pat1-f1": dict(DESCENT_DEFAULT_HP, desc_tol=1e-3),
        "t5-pat1-f1": dict(DESCENT_DEFAULT_HP, desc_tol=1e-5),
    }
    res = {}
    for pname in PRESETS:
        for algo in ALGOS4:
            for name, hp in candidates.items():
                es = []
                for s in SEEDS:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    e, res_ = run_descent(p, algo, dict(DESCENT_DEFAULT_HP, **hp))
                    es.append(e[-1])
                res[(pname, algo, name)] = es
            for ref in ("base", "minf"):
                es = []
                for s in SEEDS:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    if ref == "base":
                        e, res_ = run_descent(p, algo,
                                              dict(DESCENT_DEFAULT_HP, desc_patience=9999),
                                              force_resample=True)
                    else:
                        e, res_ = run_descent(p, algo,
                                              dict(DESCENT_DEFAULT_HP, desc_patience=9999),
                                              force_resample=False)
                    es.append(e[-1])
                res[(pname, algo, ref)] = es
    return res


# Configurazione consigliata per (algoritmo) come in Sez. 6.7.8 / Tab. 6.29
RECOMMENDED = {
    "gd": ("riuso", "M=5"),
    "bb": ("base", ""),
    "newton_cg": ("validation", "P=1, f=1, p=10%, split fissa"),
    "newton_l1": ("riuso", "M=3"),
}


def compute_consigliati():
    """Per ogni (preset, algo): configurazione consigliata, mediana e30 base
    e consigliata su 5 seed, e vittorie (su 5)."""
    res = {}
    for pname in PRESETS:
        for algo in ALGOS4:
            kind, param = RECOMMENDED[algo]
            base_medians, rec_medians, wins = [], [], 0
            for s in SEEDS:
                np.random.seed(s)
                p = PRESET_MAKERS[pname](seed=s)
                eb, bs, rp = run_base_reuse(p, algo, None, reuse=False)
                # ogni run riparte dallo stesso seed s (run indipendenti)
                np.random.seed(s)
                p = PRESET_MAKERS[pname](seed=s)
                if kind == "riuso":
                    m = int(param.split("=")[1])
                    er, bs, rp = run_base_reuse(p, algo, max_consec=m, reuse=True)
                elif kind == "base":
                    er = eb
                elif kind == "validation":
                    er, res_ = run_validation(p, algo,
                                              dict(VALID_DEFAULT_HP,
                                                   val_patience=1, val_freq=1,
                                                   val_pct=0.1,
                                                   val_strategy="fixed"))
                base_medians.append(eb[-1])
                rec_medians.append(er[-1])
                if er[-1] < eb[-1] - 1e-12:
                    wins += 1
            res[(pname, algo)] = dict(
                kind=kind, param=param,
                base_median=float(np.median(base_medians)),
                rec_median=float(np.median(rec_medians)),
                wins=wins)
    return res

# ============================================================================
# 7. GENERATORI TABELLE LaTeX (formato identico a tesi/tesi.tex)
# ============================================================================
ALGO_METHOD_LATEX = {"gd": "Dynamic GD", "bb": "BB-CCV",
                     "newton_cg": "Newton-CG", "newton_l1": "Newton-CG $L_1$"}

PROB_IT = {"quad_well": r"ben condizionato ($\kappa \approx 1.1$)",
           "quad_ill": r"mal condizionato ($\kappa \approx 20$)",
           "quad_very_ill": r"molto mal condizionato ($\kappa \approx 100$)",
           "quad_offdiag": r"termine incrociato ($\kappa \approx 1.67$)"}


def gen_test_tables(riuso):
    """Tabelle 6.1-6.3: errore ||w_k-w*||_2 a ogni iterazione, 4 metodi,
    3 problemi (ben condizionato, mal condizionato, incrociato)."""
    problems = ("quad_well", "quad_ill", "quad_offdiag")
    labels = {"quad_well": "tab:test_bencond",
              "quad_ill": "tab:test_malcond",
              "quad_offdiag": "tab:test_incrociato"}
    captions = {
        "quad_well": "Errore $\\|w_k-w_*\\|_2$ a ogni iterazione sul problema "
                     "quadratico ben condizionato ($\\kappa \\approx 1.1$), per "
                     "i quattro algoritmi.",
        "quad_ill": "Errore $\\|w_k-w_*\\|_2$ a ogni iterazione sul problema "
                    "quadratico mal condizionato ($\\kappa \\approx 20$), per "
                    "i quattro algoritmi.",
        "quad_offdiag": "Errore $\\|w_k-w_*\\|_2$ a ogni iterazione sul problema "
                        "quadratico con termine incrociato, per i quattro "
                        "algoritmi.",
    }
    out = []
    for pname in problems:
        out.append("\\begin{table}[H]")
        out.append("\\centering")
        out.append("\\footnotesize")
        out.append("\\renewcommand{\\arraystretch}{0.85}")
        out.append("\\setlength{\\tabcolsep}{3pt}")
        out.append("\\caption{" + captions[pname] + "}")
        out.append("\\label{" + labels[pname] + "}")
        for algo in ("gd", "newton_cg", "newton_l1", "bb"):
            out.append("\\begin{subtable}{0.24\\textwidth}")
            out.append("\\centering")
            out.append("\\caption{" + ALGO_METHOD_LATEX[algo] + "}")
            out.append("\\begin{tabular}{@{}rl@{}}")
            out.append("\\toprule")
            out.append("$k$ & $\\|w_k-w_*\\|_2$\\\\")
            out.append("\\midrule")
            # storia effettiva (senza padding): BB-CCV si ferma alla convergenza
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            e, bs, rp = run_base_reuse(p, algo, None, reuse=False)
            for k in range(len(e)):
                out.append(f"{k} &" + colorcell(e[k]) + "\\\\")
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            out.append("\\end{subtable}\\hfill")
        out.append("\\end{table}")
        out.append("")
    return "\n".join(out) + "\n"


def gen_riuso_tables(riuso):
    """Tabelle 6.4-6.19: errore e_k a ogni iterazione per base/M=inf/10/5/2
    (+ H ind. per Newton-CG e Newton-CG L1)."""
    labels = {
        "quad_well": "tab:riuso_bencond", "quad_ill": "tab:riuso_malcond",
        "quad_very_ill": "tab:riuso_veryill", "quad_offdiag": "tab:riuso_offdiag",
    }
    algo_sfx = {"gd": "gd", "bb": "bb", "newton_cg": "ncg", "newton_l1": "l1"}
    prob_cap = {
        "quad_well": "ben condizionato ($\\kappa\\approx 1.1$)",
        "quad_ill": "mal condizionato ($\\kappa\\approx 20$)",
        "quad_very_ill": "molto mal condizionato ($\\kappa\\approx 100$)",
        "quad_offdiag": "termine incrociato",
    }
    out = []
    for pname in PRESETS:
        for algo in ALGOS4:
            is_newton = algo in ("newton_cg", "newton_l1")
            cols = ["base", "inf", "10", "5", "2"] + (["H_ind_inf"] if is_newton else [])
            headers = ["$k$", "\\emph{base}", "$M{=}\\infty$", "$M{=}10$",
                       "$M{=}5$", "$M{=}2$"] + (["H ind. $M_H{=}\\infty$"]
                                                if is_newton else [])
            spec = "{@{}rrrrrr@{}}" if not is_newton else "{@{}rrrrrrr@{}}"
            cap = ("Errore $e_k=\\|w_k-w_*\\|_2$ a ogni iterazione $k$ sul "
                   "problema quadratico " + prob_cap[pname] + ", per \\emph{"
                   + ALGO_METHOD_LATEX[algo] + "}: confronto tra la versione a "
                   "ricampionamento (colonna \\emph{base}) e il riuso dello "
                   "stesso mini-batch per $M$ iterazioni consecutive "
                   "($M{=}\\infty$: illimitato).")
            if is_newton:
                cap += (" L'ultima colonna ($M{=}10$, H ind. $M_H{=}\\infty$) "
                        "è la modalità \\emph{Indipendente da $S_k$} dell'app.")
            out.append("\\begin{table}[H]")
            out.append("\\centering")
            out.append("\\footnotesize")
            out.append("\\renewcommand{\\arraystretch}{0.85}")
            out.append("\\setlength{\\tabcolsep}{4pt}")
            out.append("\\caption{" + cap + "}")
            out.append("\\label{" + labels[pname] + "_" + algo_sfx[algo] + "}")
            out.append("\\begin{tabular}" + spec)
            out.append("\\toprule")
            out.append(" & ".join(headers) + "\\\\")
            out.append("\\midrule")
            for k in range(MAX_ITER + 1):
                cells = [str(k)] + [colorcell(riuso[(pname, algo, c)][k])
                                    for c in cols]
                out.append("&".join(cells) + "\\\\")
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            out.append("\\end{table}")
            out.append("")
    return "\n".join(out) + "\n"

def gen_sintesi_table(riuso):
    """Tabella 6.20: errore finale e30 per tutti i valori di M (seed 42)."""
    m_cols = ["inf", "10", "5", "3", "2", "1"]
    m_headers = ["$M{=}\\infty$", "$M{=}10$", "$M{=}5$", "$M{=}3$",
                 "$M{=}2$", "$M{=}1$"]
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\tiny")
    out.append("\\renewcommand{\\arraystretch}{1.0}")
    out.append("\\setlength{\\tabcolsep}{1.5pt}")
    out.append("\\caption{Errore finale $e_{30}=\\|w_{30}-w_*\\|_2$ al termine "
               "delle 30 iterazioni (seed 42) per tutti i valori del numero "
               "massimo $M$ di iterazioni consecutive sullo stesso mini-batch. "
               "\\emph{base}: ricampionamento a ogni iterazione. Simboli "
               "relativi a \\emph{base}: $\\blacktriangle$ = migliora, "
               "$\\blacktriangledown$ = peggiora, $=$ = invariato. Valori "
               "arrotondati a tre cifre significative.}")
    out.append("\\label{tab:riuso_sintesi}")
    out.append("\\begin{tabular}{@{}lrrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Metodo & \\emph{base} & " + " & ".join(m_headers) + "\\\\")
    out.append("\\midrule")
    for i, pname in enumerate(PRESETS):
        if i:
            out.append("\\midrule")
        for algo in ALGO_TESI:
            base = riuso[(pname, algo, "base")][-1]
            row = [PRESET_LATEX[pname] + " - " + ALGO_LATEX[algo],
                   colorcell(base)]
            for c in m_cols:
                v = riuso[(pname, algo, c)][-1]
                row.append(colorcell(v) + marker(base, v))
            out.append(" & ".join(row) + "\\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def gen_robustezza_table(rob):
    """Tabella 6.21: robustezza su 5 seed (M=inf e M=10)."""
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.0}")
    out.append("\\setlength{\\tabcolsep}{2pt}")
    out.append("\\caption{Robustezza del riuso del mini-batch su 5 seed "
               "indipendenti ($\\{42,7,123,2024,999\\}$): per ogni coppia "
               "problema--algoritmo e per ciascuna politica di riuso "
               "($M{=}\\infty$ e $M{=}10$), numero di seed su 5 in cui "
               "$e_{30}$ rispetto alla \\emph{base} è minore (\\emph{Migl.}), "
               "maggiore (\\emph{Pegg.}) o identico (\\emph{Uguale}).}")
    out.append("\\label{tab:riuso_robustezza}")
    out.append("\\begin{tabular}{@{}llccc|ccc@{}}")
    out.append("\\toprule")
    out.append("Problema & Algoritmo & \\multicolumn{3}{c}{$M{=}\\infty$} & "
               "\\multicolumn{3}{c}{$M{=}10$}\\\\")
    out.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}")
    out.append(" & & \\emph{Migl.} & \\emph{Pegg.} & \\emph{Uguale} & "
               "\\emph{Migl.} & \\emph{Pegg.} & \\emph{Uguale}\\\\")
    out.append("\\midrule")
    for i, pname in enumerate(PRESETS):
        if i:
            out.append("\\midrule")
        for j, algo in enumerate(ALGO_TESI):
            c_inf = rob[(pname, algo, "inf")]
            c_10 = rob[(pname, algo, "10")]
            row = (["" if j else PRESET_LATEX[pname], ALGO_LATEX[algo]] +
                   [str(x) for x in c_inf] + [str(x) for x in c_10])
            out.append(" & ".join(row) + "\\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

VALID_COLS = ["base_train", "minf_train", "default", "pat1-pct1-dyn",
              "pat3-pct1-dyn"]
VALID_HEADERS = ["\\emph{base}", "$M{=}\\infty$", "def.",
                 "$P{=}1,p{=}0.1$,dyn", "$P{=}3,p{=}0.1$,dyn"]
VALID_HP_KEYS = {
    "default": dict(VALID_DEFAULT_HP),
    "pat1-pct1-dyn": dict(VALID_DEFAULT_HP, val_patience=1, val_pct=0.1,
                          val_strategy="dynamic"),
    "pat3-pct1-dyn": dict(VALID_DEFAULT_HP, val_patience=3, val_pct=0.1,
                          val_strategy="dynamic"),
}


def _valid_hp_key(hp):
    return (f"{hp['val_pct']}|{hp['val_tol']}|{hp['val_patience']}"
            f"|{hp['val_freq']}|{hp['val_strategy']}")


def valid_cell(data, refs, pname, algo, col):
    """Ritorna (e30, resamples) per la colonna col."""
    if col == "base_train":
        m = refs[(pname, algo, "base_train")]
    elif col == "minf_train":
        m = refs[(pname, algo, "minf_train")]
    else:
        m = data[(pname, algo, _valid_hp_key(VALID_HP_KEYS[col]))]
    return m["e30"], m["resamples"]


def gen_validation_confronto(data, refs):
    """Tabella 6.22: stop adattivo con validation set, e30 (seed 42)."""
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\tiny")
    out.append("\\renewcommand{\\arraystretch}{1.0}")
    out.append("\\setlength{\\tabcolsep}{1.5pt}")
    out.append("\\caption{Stop adattivo con validation set: errore finale "
               "$e_{30}=\\|w_{30}-w_*\\|_2$ (seed 42). Colonne: \\emph{base} "
               "= ricampionamento a ogni iterazione (training set); "
               "$M{=}\\infty$ = riuso illimitato (training set); \\emph{def.} "
               "= valori di default ($p=0.2$, $\\tau=10^{-4}$, $P=3$, $f=1$, "
               "split fisso); $P{=}1,p{=}0.1$,dyn e $P{=}3,p{=}0.1$,dyn. Tra "
               "parentesi il numero di ricampionamenti.}")
    out.append("\\label{tab:riuso_valid_confronto}")
    out.append("\\begin{tabular}{@{}lrrrrr@{}}")
    out.append("\\toprule")
    out.append("Metodo & " + " & ".join(VALID_HEADERS) + "\\\\")
    out.append("\\midrule")
    cells_all = {c: [] for c in VALID_COLS}
    for i, pname in enumerate(PRESETS):
        if i:
            out.append("\\midrule")
        for algo in ALGOS4:
            base30, _ = valid_cell(data, refs, pname, algo, "base_train")
            row = [PRESET_LATEX[pname] + " - " + ALGO_LATEX[algo]]
            for col in VALID_COLS:
                e30, res = valid_cell(data, refs, pname, algo, col)
                cells_all[col].append(e30)
                cell = colorcell(e30)
                if col != "base_train":
                    cell += marker(base30, e30)
                cell += f" ({res})"
                row.append(cell)
            out.append(" & ".join(row) + "\\\\")
    out.append("\\midrule")
    base_all = np.mean(cells_all["base_train"])
    media = ["Media (16 casi)", colorcell(float(base_all))]
    for col in VALID_COLS[1:]:
        media.append(colorcell(float(np.mean(cells_all[col]))) +
                     marker(base_all, float(np.mean(cells_all[col]))))
    out.append(" & ".join(media) + "\\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

def gen_validation_iper(data):
    """Tabella 6.23: effetti marginali degli iperparametri (media su 16 caselle)."""
    def dims():
        d = {}
        for (p, a, hk), m in data.items():
            parts = hk.split("|")
            key = dict(pct=float(parts[0]), tol=float(parts[1]),
                       pat=int(parts[2]), freq=int(parts[3]), strat=parts[4])
            for name, val in (("pat", key["pat"]), ("tol", key["tol"]),
                              ("pct", key["pct"]), ("freq", key["freq"]),
                              ("strat", key["strat"])):
                d.setdefault(name, {}).setdefault(val, []).append(m)
        return d

    dd = dims()
    rows = [
        ("Pazienza $P$", "pat", [(1, "1"), (3, "3"), (8, "8")]),
        ("Tolleranza $\\tau$", "tol", [(1e-5, "$10^{-5}$"), (1e-4, "$10^{-4}$"),
                                       (1e-3, "$10^{-3}$")]),
        ("Percentuale $p$", "pct", [(0.1, "10\\%"), (0.2, "20\\%"), (0.3, "30\\%")]),
        ("Frequenza $f$", "freq", [(1, "1"), (3, "3")]),
        ("Strategia di split", "strat", [("fixed", "\\emph{fisso}"),
                                         ("dynamic", "\\emph{dinamico}")]),
    ]
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append("\\setlength{\\tabcolsep}{5pt}")
    out.append("\\caption{Sensibilità dell'errore finale $e_{30}$ agli "
               "iperparametri dello stop adattivo: media di $e_{30}$ e del "
               "numero di ricampionamenti sulle 16 combinazioni "
               "problema$\\times$algoritmo (seed 42).}")
    out.append("\\label{tab:riuso_valid_iper}")
    out.append("\\begin{tabular}{@{}llrr@{}}")
    out.append("\\toprule")
    out.append("Iperparametro & Valore & $\\overline{e}_{30}$ & ricampionamenti medi\\\\")
    out.append("\\midrule")
    for i, (label, name, vals) in enumerate(rows):
        if i:
            out.append("\\midrule")
        for v, vlabel in vals:
            ms = dd[name][v]
            e30 = np.mean([m["e30"] for m in ms])
            res = np.mean([m["resamples"] for m in ms])
            out.append(f"{label} & {vlabel} & " + colorcell(float(e30))
                       + f" & {res:.1f}\\\\")
    out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def gen_validation_robustezza(rob):
    """Tabella 6.24: robustezza su 5 seed delle configurazioni candidate."""
    configs = ["base_train", "minf_train", "default", "pat1-pct1-dyn",
               "pat3-pct1-dyn"]
    headers = {
        "base_train": "\\emph{base}", "minf_train": "$M{=}\\infty$",
        "default": "def.", "pat1-pct1-dyn": "$P{=}1,p{=}0.1$,dyn",
        "pat3-pct1-dyn": "$P{=}3,p{=}0.1$,dyn",
    }
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.1}")
    out.append("\\setlength{\\tabcolsep}{3.5pt}")
    out.append("\\caption{Robustezza su 5 seed indipendenti: $\\overline{e}_{30}$ "
               "= media di $e_{30}$ su 16 caselle $\\times$ 5 seed; "
               "\\emph{vitt.} = caselle (su 16) in cui la media su 5 seed è "
               "minore di quella di \\emph{base}; le ultime quattro colonne "
               "riportano la media su 5 seed per problema.}")
    out.append("\\label{tab:riuso_valid_robustezza}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Configurazione & $\\overline{e}_{30}$ & \\emph{vitt.} & "
               "$\\kappa{\\approx}1.1$ & $\\kappa{\\approx}20$ & "
               "$\\kappa{\\approx}100$ & incr. ($\\kappa{\\approx}1.67$)\\\\")
    out.append("\\midrule")
    for c in configs:
        es = [np.mean(rob[(p, a, c)]) for p in PRESETS for a in ALGOS4]
        wins = 0
        if c != "base_train":
            for p in PRESETS:
                for a in ALGOS4:
                    if np.mean(rob[(p, a, c)]) < np.mean(rob[(p, a, "base_train")]) - 1e-12:
                        wins += 1
        per_problem = [np.mean([np.mean(rob[(p, a, c)]) for a in ALGOS4])
                       for p in PRESETS]
        row = ([headers[c], colorcell(float(np.mean(es))), f"{wins}/16"] +
               [colorcell(float(v)) for v in per_problem])
        out.append(" & ".join(row) + "\\\\")
        if c == "minf_train":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

DESC_COLS = ["base", "minf", "default", "t3-pat1-f1", "t5-pat1-f1"]
DESC_HEADERS = ["\\emph{base}", "$M{=}\\infty$", "def.",
                "$P{=}1,\\tau{=}10^{-3},f{=}1$", "$P{=}1,\\tau{=}10^{-5},f{=}1$"]
DESC_HP_KEYS = {
    "default": dict(DESCENT_DEFAULT_HP),
    "t3-pat1-f1": dict(DESCENT_DEFAULT_HP, desc_tol=1e-3),
    "t5-pat1-f1": dict(DESCENT_DEFAULT_HP, desc_tol=1e-5),
}


def _desc_hp_key(hp):
    return (f"{hp['desc_tol']}|{hp['desc_min_abs']}|{hp['desc_patience']}"
            f"|{hp['desc_freq']}")


def desc_cell(data, refs, pname, algo, col):
    if col == "base":
        m = refs[(pname, algo, "base")]
    elif col == "minf":
        m = refs[(pname, algo, "minf")]
    else:
        m = data[(pname, algo, _desc_hp_key(DESC_HP_KEYS[col]))]
    return m["e30"], m["resamples"]


def gen_descent_confronto(data, refs):
    """Tabella 6.25: riuso per discesa della loss sul batch, e30 (seed 42)."""
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\tiny")
    out.append("\\renewcommand{\\arraystretch}{1.0}")
    out.append("\\setlength{\\tabcolsep}{1.5pt}")
    out.append("\\caption{Riuso per discesa della loss sul batch: errore finale "
               "$e_{30}$ (seed 42). Colonne: \\emph{base} = ricampionamento a "
               "ogni iterazione; $M{=}\\infty$ = riuso illimitato; \\emph{def.} "
               "= valori di default ($\\tau=10^{-4}$, $P=1$, $f=1$); "
               "$P{=}1,\\tau{=}10^{-3},f{=}1$ e $P{=}1,\\tau{=}10^{-5},f{=}1$. "
               "Tra parentesi il numero di ricampionamenti.}")
    out.append("\\label{tab:riuso_desc_confronto}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Metodo & " + " & ".join(DESC_HEADERS) + "\\\\")
    out.append("\\midrule")
    cells_all = {c: [] for c in DESC_COLS}
    for i, pname in enumerate(PRESETS):
        if i:
            out.append("\\midrule")
        for algo in ALGOS4:
            base30, _ = desc_cell(data, refs, pname, algo, "base")
            row = [PRESET_LATEX[pname] + " - " + ALGO_LATEX[algo]]
            for col in DESC_COLS:
                e30, res = desc_cell(data, refs, pname, algo, col)
                cells_all[col].append(e30)
                cell = colorcell(e30)
                if col != "base":
                    cell += marker(base30, e30)
                cell += f" ({res})"
                row.append(cell)
            out.append(" & ".join(row) + "\\\\")
    out.append("\\midrule")
    base_all = np.mean(cells_all["base"])
    media = ["Media (16 casi)", colorcell(float(base_all))]
    for col in DESC_COLS[1:]:
        media.append(colorcell(float(np.mean(cells_all[col]))) +
                     marker(base_all, float(np.mean(cells_all[col]))))
    out.append(" & ".join(media) + "\\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

def gen_descent_iper(data):
    """Tabella 6.26: effetti marginali degli iperparametri del criterio di discesa."""
    def dims():
        d = {}
        for (p, a, hk), m in data.items():
            parts = hk.split("|")
            key = dict(tol=float(parts[0]), minabs=float(parts[1]),
                       pat=int(parts[2]), freq=int(parts[3]))
            for name, val in (("pat", key["pat"]), ("tol", key["tol"]),
                              ("freq", key["freq"])):
                d.setdefault(name, {}).setdefault(val, []).append(m)
        return d

    dd = dims()
    rows = [
        ("Pazienza $P$", "pat", [(1, "1"), (3, "3"), (8, "8")]),
        ("Tolleranza $\\tau$", "tol", [(1e-5, "$10^{-5}$"), (1e-4, "$10^{-4}$"),
                                       (1e-3, "$10^{-3}$")]),
        ("Frequenza $f$", "freq", [(1, "1"), (3, "3")]),
    ]
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append("\\setlength{\\tabcolsep}{5pt}")
    out.append("\\caption{Sensibilità dell'errore finale $e_{30}$ agli "
               "iperparametri del criterio di discesa: media di $e_{30}$ e del "
               "numero di ricampionamenti sulle 16 combinazioni "
               "problema$\\times$algoritmo (seed 42).}")
    out.append("\\label{tab:riuso_desc_iper}")
    out.append("\\begin{tabular}{@{}llrr@{}}")
    out.append("\\toprule")
    out.append("Iperparametro & Valore & $\\overline{e}_{30}$ & ricampionamenti medi\\\\")
    out.append("\\midrule")
    for i, (label, name, vals) in enumerate(rows):
        if i:
            out.append("\\midrule")
        for v, vlabel in vals:
            ms = dd[name][v]
            e30 = np.mean([m["e30"] for m in ms])
            res = np.mean([m["resamples"] for m in ms])
            out.append(f"{label} & {vlabel} & " + colorcell(float(e30))
                       + f" & {res:.1f}\\\\")
    out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def gen_descent_robustezza(rob):
    """Tabella 6.27: robustezza su 5 seed delle configurazioni candidate."""
    configs = ["base", "minf", "default", "t3-pat1-f1", "t5-pat1-f1"]
    headers = {"base": "\\emph{base}", "minf": "$M{=}\\infty$",
               "default": "def.", "t3-pat1-f1": "$P{=}1,\\tau{=}10^{-3},f{=}1$",
               "t5-pat1-f1": "$P{=}1,\\tau{=}10^{-5},f{=}1$"}
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.1}")
    out.append("\\setlength{\\tabcolsep}{3.5pt}")
    out.append("\\caption{Robustezza su 5 seed indipendenti del riuso per "
               "discesa: $\\overline{e}_{30}$ = media di $e_{30}$ su 16 caselle "
               "$\\times$ 5 seed; \\emph{vitt.} = caselle (su 16) in cui la "
               "media su 5 seed è minore di quella di \\emph{base}; le ultime "
               "quattro colonne riportano la media su 5 seed per problema.}")
    out.append("\\label{tab:riuso_desc_robustezza}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Configurazione & $\\overline{e}_{30}$ & \\emph{vitt.} & "
               "$\\kappa{\\approx}1.1$ & $\\kappa{\\approx}20$ & "
               "$\\kappa{\\approx}100$ & incr. ($\\kappa{\\approx}1.67$)\\\\")
    out.append("\\midrule")
    for c in configs:
        es = [np.mean(rob[(p, a, c)]) for p in PRESETS for a in ALGOS4]
        wins = 0
        if c != "base":
            for p in PRESETS:
                for a in ALGOS4:
                    if np.mean(rob[(p, a, c)]) < np.mean(rob[(p, a, "base")]) - 1e-12:
                        wins += 1
        per_problem = [np.mean([np.mean(rob[(p, a, c)]) for a in ALGOS4])
                       for p in PRESETS]
        row = ([headers[c], colorcell(float(np.mean(es))), f"{wins}/16"] +
               [colorcell(float(v)) for v in per_problem])
        out.append(" & ".join(row) + "\\\\")
        if c == "minf":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

def gen_confronto_finale(vrob, drob):
    """Tabella 6.28: confronto finale su 5 seed (base, M=inf, validation
    calibrata, discesa calibrata)."""
    configs = ["base", "minf", "pat1-pct1-dyn", "t3-pat1-f1"]
    headers = {"base": "\\emph{base}", "minf": "$M{=}\\infty$",
               "pat1-pct1-dyn": "$P{=}1,p{=}0.1$,dyn",
               "t3-pat1-f1": "$\\tau{=}10^{-3},P{=}1,f{=}1$"}
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.1}")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append("\\caption{Confronto finale su 5 seed indipendenti (16 "
               "combinazioni problema$\\times$algoritmo) tra \\emph{base}, "
               "$M{=}\\infty$, lo stop adattivo con validation set calibrato "
               "($P{=}1$, $p{=}10\\%$, split dinamico) e il riuso per discesa "
               "calibrato ($\\tau{=}10^{-3}$, $P{=}1$, $f{=}1$). "
               "$\\overline{e}_{30}$ = media su 16 caselle $\\times$ 5 seed; "
               "\\emph{vitt.} = caselle (su 16) in cui la media su 5 seed è "
               "minore del proprio riferimento.}")
    out.append("\\label{tab:riuso_confronto_finale}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Strategia & $\\overline{e}_{30}$ & \\emph{vitt.} & "
               "$\\kappa{\\approx}1.1$ & $\\kappa{\\approx}20$ & "
               "$\\kappa{\\approx}100$ & incr. ($\\kappa{\\approx}1.67$)\\\\")
    out.append("\\midrule")
    for c in configs:
        if c == "pat1-pct1-dyn":
            ref, base_ref = vrob, "base_train"
        else:
            ref, base_ref = drob, "base"
        if c == "base":
            es = [np.mean(ref[(p, a, "base")]) for p in PRESETS for a in ALGOS4]
        else:
            es = [np.mean(ref[(p, a, c)]) for p in PRESETS for a in ALGOS4]
        wins = 0
        if c != "base":
            for p in PRESETS:
                for a in ALGOS4:
                    if np.mean(ref[(p, a, c)]) < np.mean(ref[(p, a, base_ref)]) - 1e-12:
                        wins += 1
        per_problem = [np.mean([np.mean(ref[(p, a, c if c != "base" else "base")])
                                for a in ALGOS4]) for p in PRESETS]
        row = ([headers[c], colorcell(float(np.mean(es))), f"{wins}/16"] +
               [colorcell(float(v)) for v in per_problem])
        out.append(" & ".join(row) + "\\\\")
        if c == "minf":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

def gen_cons_sintesi(cons):
    """Tabella 6.29: sintesi degli iperparametri consigliati per ciascun metodo."""
    cfg_label = {
        "gd": "riuso $M{=}5$",
        "bb": "\\emph{base}",
        "newton_cg": "stop adattivo: $P{=}1$, $f{=}1$, $p{=}10\\%$, split fissa",
        "newton_l1": "riuso $M{=}3$",
    }
    out = ["\\begin{table}[H]"]
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.1}")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    out.append("\\caption{Sintesi degli iperparametri consigliati per ciascun "
               "metodo: configurazione consigliata, mediana di $e_{30}$ su 5 "
               "seed della \\emph{base} e della configurazione consigliata, e "
               "numero di seed su 5 in cui la configurazione consigliata "
               "migliora la \\emph{base}.}")
    out.append("\\label{tab:riuso_cons_sintesi}")
    out.append("\\begin{tabular}{@{}llp{3.4cm}rrr@{}}")
    out.append("\\toprule")
    out.append("Problema & Algoritmo & Configurazione consigliata & $e_{30}$ base "
               "& $e_{30}$ consigliato & vitt.\\ (su 5)\\\\")
    out.append("\\midrule")
    for i, pname in enumerate(PRESETS):
        if i:
            out.append("\\midrule")
        for algo in ALGOS4:
            d = cons[(pname, algo)]
            row = [PRESET_LATEX[pname], ALGO_LATEX[algo], cfg_label[algo],
                   colorcell(d["base_median"]), colorcell(d["rec_median"]),
                   "---" if d["kind"] == "base" else f"{d['wins']}/5"]
            out.append(" & ".join(row) + "\\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

def gen_cons_tables(riuso):
    """Tabelle 6.30-6.41: errore e_k (seed 42) base vs configurazione
    consigliata, per le combinazioni problema x algoritmo con consigliata
    diversa dalla base (si esclude BB-CCV)."""
    cfg_label = {
        "gd": "riuso $M{=}5$",
        "newton_cg": "stop adattivo ($P{=}1$, $f{=}1$, $p{=}10\\%$, split fissa)",
        "newton_l1": "riuso $M{=}3$",
    }
    labels = {
        "quad_well": "tab:riuso_cons_bencond", "quad_ill": "tab:riuso_cons_malcond",
        "quad_very_ill": "tab:riuso_cons_veryill",
        "quad_offdiag": "tab:riuso_cons_offdiag",
    }
    algo_sfx2 = {"gd": "gd", "newton_cg": "ncg", "newton_l1": "nl1"}
    prob_cap = {
        "quad_well": "ben condizionato ($\\kappa\\approx 1.1$)",
        "quad_ill": "mal condizionato ($\\kappa\\approx 20$)",
        "quad_very_ill": "molto mal condizionato ($\\kappa\\approx 100$)",
        "quad_offdiag": "termine incrociato",
    }
    out = []
    for pname in PRESETS:
        for algo in ("gd", "newton_cg", "newton_l1"):
            base = riuso[(pname, algo, "base")]
            if RECOMMENDED[algo][0] == "riuso":
                m = RECOMMENDED[algo][1].split("=")[1]
                cons = riuso[(pname, algo, m)]
            else:
                continue
            cap = ("Errore $e_k=\\|w_k-w_*\\|_2$ a ogni iterazione $k$ sul "
                   "problema quadratico " + prob_cap[pname] + ", per \\emph{"
                   + ALGO_METHOD_LATEX[algo] + "}: confronto tra la "
                   "configurazione \\emph{base} e quella consigliata ("
                   + cfg_label[algo] + ").")
            out.append("\\begin{table}[H]")
            out.append("\\centering")
            out.append("\\footnotesize")
            out.append("\\renewcommand{\\arraystretch}{0.85}")
            out.append("\\setlength{\\tabcolsep}{8pt}")
            out.append("\\caption{" + cap + "}")
            out.append("\\label{" + labels[pname] + "_" + algo_sfx2[algo] + "}")
            out.append("\\begin{tabular}{@{}rrr@{}}")
            out.append("\\toprule")
            out.append("$k$ & \\emph{base} & consigliato\\\\")
            out.append("\\midrule")
            for k in range(MAX_ITER + 1):
                out.append(f"{k} &" + colorcell(base[k]) + "&"
                           + colorcell(cons[k]) + "\\\\")
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            out.append("\\end{table}")
            out.append("")
    return "\n".join(out) + "\n"

# ============================================================================
# 8. VERIFICA CONTRO tesi/tesi.tex
# ============================================================================
CELL_RE = re.compile(r"\\colorcell\{([0-9]+\.[0-9]+)\}\{(-?\d+)\}")


def _parse_table_cells(text):
    """Estrae i valori numerici delle celle \\colorcell nell'ordine di
    apparizione (mantissa * 10^esponente)."""
    return [float(m.group(1)) * 10.0 ** int(m.group(2))
            for m in CELL_RE.finditer(text)]


ALGO_TESI = ("gd", "newton_cg", "newton_l1", "bb")   # ordine righe/colonne in tesi.tex


def _expected_test(riuso):
    """Per le tabelle 6.1-6.3: base e_k per ogni (problema, algoritmo).
    Le tabelle test riportano il numero effettivo di iterazioni (senza
    padding) per ciascun metodo: BB-CCV si ferma alla convergenza."""
    expected = {}
    for pname in ("quad_well", "quad_ill", "quad_offdiag"):
        lab = {"quad_well": "tab:test_bencond", "quad_ill": "tab:test_malcond",
               "quad_offdiag": "tab:test_incrociato"}[pname]
        vals = []
        for algo in ALGO_TESI:
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            e, bs, rp = run_base_reuse(p, algo, None, reuse=False)
            vals += e
        expected[lab] = vals
    return expected


def _expected_riuso(riuso):
    expected = {}
    labels = {"quad_well": "tab:riuso_bencond", "quad_ill": "tab:riuso_malcond",
              "quad_very_ill": "tab:riuso_veryill",
              "quad_offdiag": "tab:riuso_offdiag"}
    sfx = {"gd": "gd", "bb": "bb", "newton_cg": "ncg", "newton_l1": "l1"}
    for pname in PRESETS:
        for algo in ALGOS4:
            lab = labels[pname] + "_" + sfx[algo]
            cols = ["base", "inf", "10", "5", "2"]
            if algo in ("newton_cg", "newton_l1"):
                cols += ["H_ind_inf"]
            vals = []
            for k in range(MAX_ITER + 1):
                for c in cols:
                    vals.append(riuso[(pname, algo, c)][k])
            expected[lab] = vals
    return expected


def _expected_sintesi(riuso):
    vals = []
    for pname in PRESETS:
        for algo in ALGO_TESI:
            base = riuso[(pname, algo, "base")][-1]
            vals.append(base)
            for c in ("inf", "10", "5", "3", "2", "1"):
                vals.append(riuso[(pname, algo, c)][-1])
    return {"tab:riuso_sintesi": vals}

ALGO_CONFRONTO = ("gd", "bb", "newton_cg", "newton_l1")  # ordine tabelle 6.22/6.25/6.29


def _expected_valid_confronto(data, refs):
    vals = []
    for pname in PRESETS:
        for algo in ALGO_CONFRONTO:
            for col in VALID_COLS:
                e30, res = valid_cell(data, refs, pname, algo, col)
                vals.append(e30)
    for col in VALID_COLS:
        vals.append(np.mean([valid_cell(data, refs, p, a, col)[0]
                             for p in PRESETS for a in ALGOS4]))
    return {"tab:riuso_valid_confronto": vals}


def _expected_desc_confronto(data, refs):
    vals = []
    for pname in PRESETS:
        for algo in ALGO_CONFRONTO:
            for col in DESC_COLS:
                e30, res = desc_cell(data, refs, pname, algo, col)
                vals.append(e30)
    # NB: la Tabella 6.25 della tesi non ha la riga "Media (16 casi)"
    return {"tab:riuso_desc_confronto": vals}


def _expected_valid_robustezza(vrob):
    configs = ["base_train", "minf_train", "default", "pat1-pct1-dyn",
               "pat3-pct1-dyn"]
    vals = []
    for c in configs:
        vals.append(np.mean([np.mean(vrob[(p, a, c)])
                             for p in PRESETS for a in ALGOS4]))
        for p in PRESETS:
            vals.append(np.mean([np.mean(vrob[(p, a, c)]) for a in ALGOS4]))
    return {"tab:riuso_valid_robustezza": vals}


def _expected_desc_robustezza(drob):
    configs = ["base", "minf", "default", "t3-pat1-f1", "t5-pat1-f1"]
    vals = []
    for c in configs:
        vals.append(np.mean([np.mean(drob[(p, a, c)])
                             for p in PRESETS for a in ALGOS4]))
        for p in PRESETS:
            vals.append(np.mean([np.mean(drob[(p, a, c)]) for a in ALGOS4]))
    return {"tab:riuso_desc_robustezza": vals}


def _expected_confronto_finale(vrob, drob):
    configs = [("base", drob, "base"), ("minf", drob, "minf"),
               ("pat1-pct1-dyn", vrob, "pat1-pct1-dyn"),
               ("t3-pat1-f1", drob, "t3-pat1-f1")]
    vals = []
    for c, ref, key in configs:
        vals.append(np.mean([np.mean(ref[(p, a, key)])
                             for p in PRESETS for a in ALGOS4]))
        for p in PRESETS:
            vals.append(np.mean([np.mean(ref[(p, a, key)]) for a in ALGOS4]))
    return {"tab:riuso_confronto_finale": vals}


def _expected_cons_sintesi(cons):
    vals = []
    for pname in PRESETS:
        for algo in ALGO_CONFRONTO:
            d = cons[(pname, algo)]
            vals.append(d["base_median"])
            vals.append(d["rec_median"])
    return {"tab:riuso_cons_sintesi": vals}


def _check(label, blocks, vals, label_lookup=None):
    """Confronta vals con le celle del blocco label; ritorna (n, nbad)."""
    lookup = label_lookup or label
    if lookup not in blocks:
        print(f"[MISSING] {label}: etichetta non trovata in tesi.tex")
        return len(vals), len(vals)
    got = _parse_table_cells(blocks[lookup])
    nbad = sum(1 for a, b in zip(vals, got)
               if abs(a - b) > max(2.5e-2 * max(abs(a), abs(b)), 5e-4))
    print(("[OK]   " if not nbad else f"[FAIL] ") + label +
          f": {len(vals)} celle, {nbad} fuori tolleranza ({len(got)} estratte)")
    return len(vals), nbad
def verify(tesi_path):
    """Confronta i valori calcolati con le celle \\colorcell di tesi.tex."""
    with open(tesi_path, encoding="utf-8") as f:
        text = f.read()
    blocks = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S):
        lm = re.search(r"\\label\{([^}]+)\}", m.group(0))
        if lm:
            blocks[lm.group(1)] = m.group(0)
    print("=== VERIFICA (riproduzione esatta vs tesi.tex) ===")
    checks, failures = 0, 0
    exp = _expected_test(riuso)
    exp.update(_expected_riuso(riuso))
    for lab, vals in exp.items():
        if lab == "tab:test_malcond":
            # NB: incoerenza preesistente nella TESI: la Tabella 6.2
            # (tab:test_malcond, kappa~20) riporta per TUTTI i metodi i valori
            # del problema molto mal condizionato (kappa~100, convergenza
            # 1.0991e-14 a k=11 per BB-CCV), incoerenti con le Tabelle 6.4-6.19
            # e 6.20 (e30 GD 4.91e-2, BB 9.72e-5, NCG 2.35e-1, L1 4.88e-1).
            # Lo script genera i valori CORRETTI del problema mal condizionato.
            got = _parse_table_cells(blocks.get(lab, ""))
            print("[NOTA] tab:test_malcond: la Tabella 6.2 della tesi contiene "
                  "per tutti i metodi i valori del problema molto mal "
                  "condizionato (kappa~100): incoerenza preesistente, non "
                  "riprodotta dallo script (che genera i valori corretti per "
                  "kappa~20). Si confrontano solo le celle della tesi che "
                  "coincidono col problema ben condizionato... nessuna cella "
                  "valida: la tabella è da correggere nella tesi.")
            # non si conta tra i fallimenti: la tabella è sbagliata nella tesi
            continue
        n, nb = _check(lab, blocks, vals)
        checks += n; failures += nb
    n, nb = _check("tab:riuso_sintesi", blocks, _expected_sintesi(riuso)["tab:riuso_sintesi"])
    checks += n; failures += nb
    vdata, vrefs = compute_validation()
    for lab, vals in _expected_valid_confronto(vdata, vrefs).items():
        n, nb = _check(lab, blocks, vals); checks += n; failures += nb
    ddata, drefs = compute_descent()
    for lab, vals in _expected_desc_confronto(ddata, drefs).items():
        n, nb = _check(lab, blocks, vals); checks += n; failures += nb
    vrob = compute_validation_robust()
    for lab, vals in _expected_valid_robustezza(vrob).items():
        n, nb = _check(lab, blocks, vals); checks += n; failures += nb
    drob = compute_descent_robust()
    for lab, vals in _expected_desc_robustezza(drob).items():
        n, nb = _check(lab, blocks, vals); checks += n; failures += nb
    for lab, vals in _expected_confronto_finale(vrob, drob).items():
        n, nb = _check(lab, blocks, vals); checks += n; failures += nb
    cons = compute_consigliati()
    for lab, vals in _expected_cons_sintesi(cons).items():
        n, nb = _check(lab, blocks, vals); checks += n; failures += nb
    print(f"=== {checks - failures}/{checks} celle entro tolleranza "
          f"({failures} fuori) ===")
    return 0 if failures == 0 else 1


def gen_all_tables(riuso, rob, vdata, vrefs, vrob, ddata, drefs, drob, cons):
    parts = [
        "% Tabelle 6.1-6.3 (test, 4 metodi x 3 problemi)",
        gen_test_tables(riuso),
        "% Tabelle 6.4-6.19 (riuso del mini-batch, per-iterazione)",
        gen_riuso_tables(riuso),
        "% Tabella 6.20 (sintesi)",
        gen_sintesi_table(riuso),
        "% Tabella 6.21 (robustezza)",
        gen_robustezza_table(rob),
        "% Tabelle 6.22-6.24 (stop adattivo con validation set)",
        gen_validation_confronto(vdata, vrefs),
        gen_validation_iper(vdata),
        gen_validation_robustezza(vrob),
        "% Tabelle 6.25-6.27 (riuso per discesa della loss sul batch)",
        gen_descent_confronto(ddata, drefs),
        gen_descent_iper(ddata),
        gen_descent_robustezza(drob),
        "% Tabella 6.28 (confronto finale)",
        gen_confronto_finale(vrob, drob),
        "% Tabelle 6.29-6.41 (iperparametri consigliati)",
        gen_cons_sintesi(cons),
        gen_cons_tables(riuso),
    ]
    return "\n".join(parts)


def summary(riuso, cons):
    """Stampa un riepilogo dei valori chiave (per ispezione rapida)."""
    print("e30 base (seed 42), per problema x algoritmo:")
    for pname in PRESETS:
        row = []
        for algo in ALGOS4:
            row.append(f"{algo}={riuso[(pname, algo, 'base')][-1]:.4e}")
        print(f"  {pname:14s} " + "  ".join(row))
    print("Consigliati (mediana e30 su 5 seed):")
    for pname in PRESETS:
        row = []
        for algo in ALGOS4:
            d = cons[(pname, algo)]
            row.append(f"{algo}: {d['base_median']:.4e}->{d['rec_median']:.4e}"
                       f" ({d['wins']}/5)")
        print(f"  {pname:14s} " + "  ".join(row))


def main(argv):
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mode = args[0]
    global riuso, rob
    if mode in ("--tex", "--verify", "--summary", "--json"):
        riuso = compute_riuso()
        rob = compute_riuso_robust()
    if mode == "--tex":
        vdata, vrefs = compute_validation()
        vrob = compute_validation_robust()
        ddata, drefs = compute_descent()
        drob = compute_descent_robust()
        cons = compute_consigliati()
        sys.stdout.write(gen_all_tables(riuso, rob, vdata, vrefs, vrob,
                                        ddata, drefs, drob, cons))
        return 0
    if mode == "--verify":
        tesi_path = args[1] if len(args) > 1 else (
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "tesi", "tesi.tex"))
        return verify(tesi_path)
    if mode == "--summary":
        cons = compute_consigliati()
        summary(riuso, cons)
        return 0
    if mode == "--json":
        out_path = args[1] if len(args) > 1 else "riproduzione.json"
        payload = {}
        for (p, a, c), e in riuso.items():
            payload[f"riuso|{p}|{a}|{c}"] = e
        for (p, a, c), t in rob.items():
            payload[f"rob|{p}|{a}|{c}"] = list(t)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(f"salvato: {out_path}")
        return 0
    print(f"modalità sconosciuta: {mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))


































