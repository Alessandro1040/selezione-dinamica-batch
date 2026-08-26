#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigenerazione delle 8 tabelle Newton del riuso del mini-batch (ex Appendice E,
ora Sezione 6.7 di tesi/tesi.tex; il sorgente standalone e' conservato in
altro/appendice_riuso.tex), inclusa la colonna per il riuso dell'Hessiana
INDIPENDENTE da S_k (iperparametro M_H) introdotta nell'app web con il commit
e682db0 (22/08/2026). L'intestazione usa i nomi descrittivi
$M{=}\\infty$, $M{=}10$, $M{=}5$, $M{=}2$, "H ind. $M_H{=}\\infty$" e le
didascalie spiegano l'ultima colonna (modalita' legata/indipendente da S_k).

La logica degli algoritmi e' quella del codice Python generato da
visualizzazione.html (generateNewtonCG / generateNewtonL1):
  - modalita' "Legato a S_k" (reuse_hessian=True, default): H_k e' ricampionata
    insieme a S_k, nello stesso blocco;
  - modalita' "Indipendente da S_k" (reuse_hessian=False): H_k ha un proprio
    contatore used_H e un proprio max max_hessian_reuse (M_H); se M_H e' None
    (illimitato) H_k resta legata solo alla CCV su S_k.
La CCV violata ricampiona sempre sia S_k sia H_k (need_resample_S/H = True).

Uso:
  python3 gen_tabelle_riuso.py --data [APPENDICE_TEX]
      Esegue gli esperimenti e confronta le colonne esistenti (base, M=inf,
      M=10, M=5, M=2) delle 8 tabelle Newton con i valori gia' in
      APPENDICE_TEX. Stampa i valori della nuova colonna (M=10, H indipendente
      M_H=inf) e del caso complementare (M=inf, M_H=10).
  python3 gen_tabelle_riuso.py --tex [APPENDICE_TEX] [OUT]
      Riscrive le 8 tabelle Newton in APPENDICE_TEX con l'intestazione
      descrittiva (base, M=inf, M=10, M=5, M=2, H ind. M_H=inf) e scrive il
      risultato in OUT (default stdout).
      Default APPENDICE_TEX: altro/appendice_riuso.tex (riferimento storico).
"""
import re
import sys

import numpy as np

N        = 200
W0       = [2.0, -3.0]
ALPHA    = 0.1
THETA    = 0.5
BATCH0   = 5
MAX_ITER = 30
SEED     = 42
R_       = 0.2
MAXCG    = 10
NU       = 0.1
SIGMA    = 0.1
ETA      = 0.5

NEW_HEADER = r"H ind. $M_H{=}\infty$"

# Tabelle Newton: (label, numero tabella, preset, algoritmo)
NEWTON_TABLES = [
    ("tab:riuso_bencond_ncg", "E.2",  "quad_well",     "newton_cg"),
    ("tab:riuso_bencond_l1",  "E.3",  "quad_well",     "newton_l1"),
    ("tab:riuso_malcond_ncg", "E.6",  "quad_ill",      "newton_cg"),
    ("tab:riuso_malcond_l1",  "E.7",  "quad_ill",      "newton_l1"),
    ("tab:riuso_veryill_ncg", "E.10", "quad_very_ill", "newton_cg"),
    ("tab:riuso_veryill_l1",  "E.11", "quad_very_ill", "newton_l1"),
    ("tab:riuso_offdiag_ncg", "E.14", "quad_offdiag",  "newton_cg"),
    ("tab:riuso_offdiag_l1",  "E.15", "quad_offdiag",  "newton_l1"),
]


# ----------------------------------------------------------------------
# PRESET (codice esatto dell'app, LOSS_PRESETS in visualizzazione.html)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# NEWTON-CG (codice generato da generateNewtonCG)
# ----------------------------------------------------------------------
def newton_cg(p, w0, theta, max_iter, alpha, batch0, R, maxcg,
              reuse=False, max_consec=None, reuse_hessian=True,
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
            if need_resample_S or indices_S is None or (max_consec is not None and used_S >= max_consec):
                indices_S = np.random.choice(N, size=n, replace=False)
                n_h = min(max(1, int(round(R * n))), N)
                used_S = 0
                need_resample_S = False
                if reuse_hessian:
                    indices_H = _draw_H_indices(subset, indices_S, n_h, N)
                    used_H = 0
                    need_resample_H = False
            grads_arr = np.array([grad_i(w, i) for i in indices_S])
            g = np.mean(grads_arr, axis=0)
        else:
            indices_S = np.random.choice(N, size=n, replace=False)
            g = np.mean([grad_i(w, i) for i in indices_S], axis=0)
            n_h = min(max(1, int(round(R * n))), N)
            indices_H = _draw_H_indices(subset, indices_S, n_h, N)
        if reuse and not reuse_hessian:
            # H_k indipendente da S_k: si ricampiona secondo max_hessian_reuse
            if need_resample_H or indices_H is None or (max_hessian_reuse is not None and used_H >= max_hessian_reuse):
                n_h = min(max(1, int(round(R * n))), N)
                indices_H = _draw_H_indices(subset, indices_S, n_h, N)
                used_H = 0
                need_resample_H = False
        if reuse:
            # CCV sul campione RIUSATO al punto corrente (gratis: usa i gradienti del passo)
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
            var_vec = np.var([grad_i(w, i) for i in indices_new], axis=0, ddof=1) \
                if n > 1 else np.zeros_like(g_new)
            V_norm1, gg_new = np.sum(var_vec), np.dot(g_new, g_new)
            if gg_new > 1e-16 and V_norm1 / n > theta**2 * gg_new:
                n = min(int(np.ceil(V_norm1 / (theta**2 * gg_new))) + 1, N)
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes


# ----------------------------------------------------------------------
# NEWTON-L1 (codice generato da generateNewtonL1, variante Hessian-free)
# ----------------------------------------------------------------------
def newton_l1(p, w0, theta, max_iter, alpha, batch0, nu, sigma, R, maxcg, eta=0.5,
              reuse=False, max_consec=None, reuse_hessian=True,
              max_hessian_reuse=None, subset=True):
    N = p["N"]
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    loss_i, grad_i, hessvec_i, grad_full = (p["loss_i"], p["grad_i"],
                                            p["hessvec_i"], p["grad_full"])

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
            if need_resample_S or indices_S is None or (max_consec is not None and used_S >= max_consec):
                indices_S = np.random.choice(N, size=n, replace=False)
                n_h = max(1, int(round(R * n)))
                n_h = min(n_h, N)
                used_S = 0
                need_resample_S = False
                if reuse_hessian:
                    indices_H = _draw_H_indices(subset, indices_S, n_h, N)
                    used_H = 0
                    need_resample_H = False
        else:
            indices_S = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)
        if reuse and not reuse_hessian:
            # H_k indipendente da S_k: si ricampiona secondo max_hessian_reuse
            if need_resample_H or indices_H is None or (max_hessian_reuse is not None and used_H >= max_hessian_reuse):
                n_h = max(1, int(round(R * n)))
                n_h = min(n_h, N)
                indices_H = _draw_H_indices(subset, indices_S, n_h, N)
                used_H = 0
                need_resample_H = False
        if reuse:
            # CCV sul campione RIUSATO al punto corrente (gratis: usa i gradienti del passo)
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
            indices_H = _draw_H_indices(subset, indices_S, n_h, N)
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


# ----------------------------------------------------------------------
# ESECUZIONE ESPERIMENTI
# ----------------------------------------------------------------------
# Configurazioni: (nome colonna, kwargs per l'algoritmo)
CONFIGS = [
    ("base",            dict(reuse=False)),
    ("M=inf",           dict(reuse=True, max_consec=None, reuse_hessian=True,  max_hessian_reuse=None)),
    ("M=10",            dict(reuse=True, max_consec=10,   reuse_hessian=True,  max_hessian_reuse=None)),
    ("M=5",             dict(reuse=True, max_consec=5,    reuse_hessian=True,  max_hessian_reuse=None)),
    ("M=3",             dict(reuse=True, max_consec=3,    reuse_hessian=True,  max_hessian_reuse=None)),
    ("M=2",             dict(reuse=True, max_consec=2,    reuse_hessian=True,  max_hessian_reuse=None)),
    ("M=1",             dict(reuse=True, max_consec=1,    reuse_hessian=True,  max_hessian_reuse=None)),
    # Nuova colonna: batch M=10 con H_k INDIPENDENTE da S_k, M_H illimitato (CCV).
    ("M=10,H_ind_inf",  dict(reuse=True, max_consec=10,   reuse_hessian=False, max_hessian_reuse=None)),
    # Caso complementare: batch illimitato, H_k indipendente con M_H=10.
    ("M=inf,H_ind_10",  dict(reuse=True, max_consec=None, reuse_hessian=False, max_hessian_reuse=10)),
]
# Colonne delle 8 tabelle Newton (E.2...E.15), in ordine:
# base, M=inf, M=10, M=5, M=2, poi la nuova colonna.
NEWTON_COLS = ["base", "M=inf", "M=10", "M=5", "M=2", "M=10,H_ind_inf"]
# Colonne della sintesi E.17 (per tutti i valori di M).
SINTESI_COLS = ["M=inf", "M=10", "M=5", "M=3", "M=2", "M=1"]
SEEDS = [42, 7, 123, 2024, 999]


def fmt(x):
    """Formatta e_k nello stile delle tabelle: $1.4142\times10^{0}$."""
    s = f"{x:.4e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _draw_H_indices(subset, indices_S, n_h, N):
    """Sottocampionamento Hessiana: True = H_k e' sottoinsieme di S_k (default app),
    False = H_k e' estratta dall'intero dataset N."""
    if subset:
        return np.random.choice(indices_S, size=n_h, replace=False)
    return np.random.choice(N, size=n_h, replace=False)


def errs(hist, p):
    return [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]


def run_experiments(subset=True):
    """Ritorna data[(preset, algo, config)] = lista e_k (31 valori, k=0..30)."""
    data = {}
    for label, tabnum, pname, algo in NEWTON_TABLES:
        for cname, ckwargs in CONFIGS:
            np.random.seed(SEED)          # dataset rigenerato da seed 42
            p = PRESET_MAKERS[pname]()
            if algo == "newton_cg":
                hist, _ = newton_cg(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                    R_, MAXCG, subset=subset, **ckwargs)
            else:
                hist, _ = newton_l1(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                    NU, SIGMA, R_, MAXCG, subset=subset, **ckwargs)
            data[(pname, algo, cname)] = errs(hist, p)
    return data


def run_robustness(subset=True):
    """Robustezza su 5 seed: per ogni (preset, algo, cfg) ritorna (meglio, peggio,
    uguale) contando gli e30 di cfg vs base sui seed SEEDS."""
    res = {}
    for label, tabnum, pname, algo in NEWTON_TABLES:
        base30 = {}
        for s in SEEDS:
            p = PRESET_MAKERS[pname](seed=s)
            if algo == "newton_cg":
                hist, _ = newton_cg(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                    R_, MAXCG, subset=subset)
            else:
                hist, _ = newton_l1(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                    NU, SIGMA, R_, MAXCG, subset=subset)
            base30[s] = errs(hist, p)[-1]
        for cfg_name in ("M=inf", "M=10"):
            ck = dict([c for c in CONFIGS if c[0] == cfg_name][0][1])
            counts = [0, 0, 0]            # meglio, peggio, uguale
            for s in SEEDS:
                p = PRESET_MAKERS[pname](seed=s)
                if algo == "newton_cg":
                    hist, _ = newton_cg(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                        R_, MAXCG, subset=subset, **ck)
                else:
                    hist, _ = newton_l1(p, W0, THETA, MAX_ITER, ALPHA, BATCH0,
                                        NU, SIGMA, R_, MAXCG, subset=subset, **ck)
                e = errs(hist, p)[-1]
                if e < base30[s] - 1e-12:
                    counts[0] += 1
                elif e > base30[s] + 1e-12:
                    counts[1] += 1
                else:
                    counts[2] += 1
            res[(pname, algo, cfg_name)] = tuple(counts)
    return res


def fmt3(x):
    """Formatta a 3 cifre significative nello stile di E.17: $4.13\times10^{-1}$."""
    s = f"{x:.2e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def marker(base, v):
    if v < base - 1e-12:
        return "$\\blacktriangle$"
    if v > base + 1e-12:
        return "$\\blacktriangledown$"
    return "$=$"


# ----------------------------------------------------------------------
# PARSING del .tex (per validazione e chirurgia delle tabelle)
# ----------------------------------------------------------------------
CELL_RE = re.compile(r"\$([0-9]+\.[0-9]+)\\times10\^\{(-?\d+)\}\$")


def parse_cell(s):
    m = CELL_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"cella non riconosciuta: {s!r}")
    return float(m.group(1)) * 10.0 ** int(m.group(2))


def find_table_block(lines, label):
    r"""Indici [start, end) del blocco \begin{table}...\end{table} che contiene label."""
    li = next(i for i, l in enumerate(lines) if f"\\label{{{label}}}" in l)
    start = li
    while start > 0 and "\\begin{table}" not in lines[start]:
        start -= 1
    end = li
    while end < len(lines) and "\\end{table}" not in lines[end]:
        end += 1
    return start, end + 1


def parse_tabular(lines, start, end):
    """Estrae intestazione e righe dati dal blocco tabular (entro [start,end))."""
    ts = next(i for i in range(start, end) if lines[i].startswith("\\begin{tabular}"))
    te = next(i for i in range(start, end) if lines[i].startswith("\\end{tabular}"))
    top = next(i for i in range(ts, te) if lines[i].startswith("\\toprule"))
    mid = next(i for i in range(ts, te) if lines[i].startswith("\\midrule"))
    bot = next(i for i in range(ts, te) if lines[i].startswith("\\bottomrule"))
    header = lines[top + 1]
    rows = lines[mid + 1:bot]
    return ts, te, header, rows


def split_row(line):
    """Splitta una riga 'a & b & c\\' in celle (senza '\\\\' finale)."""
    body = line.rstrip().rstrip("\\")
    return [c.strip() for c in body.split("&")]








# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mode = args[0]
    tex_path = args[1] if len(args) > 1 else "altro/appendice_riuso.tex"

    if mode == "--diagnosi":
        # Conferma che le tabelle attuali sono state generate con subset=False.
        data_false = run_experiments(subset=False)
        with open(tex_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        ok = True
        for label, tabnum, pname, algo in NEWTON_TABLES:
            start, end = find_table_block(lines, label)
            ts, te, header, rows = parse_tabular(lines, start, end)
            parsed = [[parse_cell(c) for c in split_row(r)[1:]] for r in rows]
            worst = 0.0
            for ci, cname in enumerate(NEWTON_COLS[:5]):
                comp = data_false[(pname, algo, cname)]
                for ri in range(min(len(parsed), len(comp))):
                    worst = max(worst, abs(parsed[ri][ci] - comp[ri]))
            status = "OK (riproduce le tabelle attuali)" if worst < 1.1e-4 else "FAIL"
            if status.startswith("FAIL"):
                ok = False
            print(f"  {tabnum} {label}: worst|diff|(subset=False) = {worst:.2e}  [{status}]")
        return 0 if ok else 2

    data = run_experiments(subset=True)

    if mode == "--data":
        # Coerenza con riproduci_tabelle.py (Tabelle 6.3-6.5 della tesi)
        import importlib.util
        spec = importlib.util.spec_from_file_location("rt", "altro/script/riproduci_tabelle.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        print("== COERENZA base (subset=True) vs riproduci_tabelle.py (Tabb. 6.3-6.5) ==")
        for pname in ("quad_well", "quad_very_ill", "quad_offdiag"):
            pmaker = {"quad_well": rt.make_preset_well,
                      "quad_very_ill": rt.make_preset_very_ill,
                      "quad_offdiag": rt.make_preset_offdiag}[pname]
            for algo in ("newton_cg", "newton_l1"):
                np.random.seed(rt.SEED)
                p = pmaker()
                if algo == "newton_cg":
                    hist, _ = rt.newton_cg(p, rt.W0, rt.THETA, rt.MAX_ITER,
                                           rt.ALPHA, rt.BATCH0, rt.R_, rt.MAXCG)
                else:
                    hist, _ = rt.newton_l1(p, rt.W0, rt.THETA, rt.MAX_ITER,
                                           rt.ALPHA, rt.BATCH0, rt.NU, rt.SIGMA,
                                           rt.R_, rt.MAXCG)
                e_ref = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
                e_mine = data[(pname, algo, "base")]
                worst = max(abs(a - b) for a, b in zip(e_ref, e_mine))
                print(f"  {pname} {algo}: worst|diff| = {worst:.2e} "
                      f"[{'OK' if worst < 1.1e-4 else 'FAIL'}]")
        print()
        print("== e30 (subset=True) per tutte le colonne ==")
        for label, tabnum, pname, algo in NEWTON_TABLES:
            vals = [data[(pname, algo, c)][-1] for c in NEWTON_COLS]
            row = "  ".join(f"{c}:{v:.4e}" for c, v in zip(NEWTON_COLS, vals))
            print(f"  {tabnum} {label}: {row}")
        print()
        print("== NUOVA COLONNA (M=10, H indipendente M_H=inf), valori LaTeX ==")
        for label, tabnum, pname, algo in NEWTON_TABLES:
            vals = data[(pname, algo, "M=10,H_ind_inf")]
            print(f"  {tabnum} {label}:")
            for k, v in enumerate(vals):
                print(f"    {k:2d} {fmt(v)}")
        print()
        print("== CASO COMPLEMENTARE (M=inf, H indipendente M_H=10): e30 ==")
        for label, tabnum, pname, algo in NEWTON_TABLES:
            vals = data[(pname, algo, "M=inf,H_ind_10")]
            print(f"  {tabnum} {label}: e30 = {vals[-1]:.4e}")
        print()
        print("== E.17 SINTESI: righe Newton (subset=True) ==")
        for label, tabnum, pname, algo in NEWTON_TABLES:
            base30 = data[(pname, algo, "base")][-1]
            cells = [fmt3(base30)]
            for c in SINTESI_COLS:
                v = data[(pname, algo, c)][-1]
                cells.append(fmt3(v) + marker(base30, v))
            pname_it = {"quad_well": "$\\kappa\\approx1.1$", "quad_ill": "$\\kappa\\approx20$",
                        "quad_very_ill": "$\\kappa\\approx100$", "quad_offdiag": "incrociato ($\\kappa\\approx1.67$)"}[pname]
            algo_it = {"newton_cg": "Newton-CG", "newton_l1": "Newton-CG $L_1$"}[algo]
            print(f"  {pname_it} - {algo_it} & " + " & ".join(cells) + "\\\\")
        print()
        print("== E.18 ROBUSTEZZA: righe Newton (subset=True, 5 seed) ==")
        rob = run_robustness(subset=True)
        for label, tabnum, pname, algo in NEWTON_TABLES:
            c_inf = rob[(pname, algo, "M=inf")]
            c_10 = rob[(pname, algo, "M=10")]
            print(f"  {pname} {algo}: M=inf {c_inf[0]}/{c_inf[1]}/{c_inf[2]}   "
                  f"M=10 {c_10[0]}/{c_10[1]}/{c_10[2]}")
        return 0

    if mode == "--tex":
        out_path = args[2] if len(args) > 2 else None
        with open(tex_path, encoding="utf-8") as f:
            text = f.read()
        lines = text.splitlines()
        for label, tabnum, pname, algo in NEWTON_TABLES:
            start, end = find_table_block(lines, label)
            ts, te, header, rows = parse_tabular(lines, start, end)
            hcells = ["$k$", "\\emph{base}", "$M{=}\\infty$", "$M{=}10$",
                      "$M{=}5$", "$M{=}2$", NEW_HEADER]
            header_line = " & ".join(hcells) + "\\\\"
            data_rows = []
            for k in range(31):
                cells = [str(k)] + [fmt(data[(pname, algo, c)][k]) for c in NEWTON_COLS]
                data_rows.append(" & ".join(cells) + "\\\\")
            spec_old = lines[ts]
            if "{@{}rrrrrrr@{}}" not in spec_old:
                raise SystemExit(f"{label}: colonna spec inattesa: {spec_old!r}")
            middle = ([spec_old, "\\toprule", header_line, "\\midrule"]
                      + data_rows + ["\\bottomrule"])
            lines[start:end] = lines[start:ts] + middle + lines[te:end]
        out_text = "\n".join(lines) + "\n"
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out_text)
            print(f"scritto: {out_path}")
        else:
            sys.stdout.write(out_text)
        return 0

    print(f"modalita' sconosciuta: {mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
