#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riproduzione fedele delle Tabelle 6.1-6.3 della tesi
(errore ||w_k-w*||_2 a ogni iterazione, 4 metodi x 3 problemi).

Fonte del codice:
  - preset e algoritmi: visualizzazione.html (app interattiva, default: N=200,
    seed 42, w0=[2,-3], alpha=0.1, theta=0.5, batch0=5, max_iter=30, R=0.2,
    maxcg=10, nu=0.1, sigma=0.1, eta=0.5, hkSubset=subset, hessianFree=free,
    dynamicBatch=dynamic, line search=Wolfe per GD/Newton-CG)
  - BB-CCV: altro/script/bbccv.py (passo Barzilai-Borwein salvaguardato in
    [alpha/20, 5*alpha] + line search Armijo)
Ogni (problema, metodo) parte da np.random.seed(42) e rigenera il dataset.
"""
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


# ----------------------------------------------------------------------
# PRESET (codice esatto dell'app)
# ----------------------------------------------------------------------
def make_preset_well():
    """quad_well: J = (w1-1)^2 + (w2+2)^2 + 0.1 w1 w2  (kappa~1.1)"""
    def J(w):
        x, y = w[0], w[1]
        return (x - 1)**2 + (y + 2)**2 + 0.1*x*y
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([2*(x - 1) + 0.1*y, 2*(y + 2) + 0.1*x])
    def hessJ(w):
        return np.array([[2.0, 0.1], [0.1, 2.0]])
    np.random.seed(SEED)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    raw_c = 0.1 + 0.05 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    c_i = raw_c - np.mean(raw_c) + 0.1
    def loss_i(w, i):
        x, y = w[0], w[1]
        return (x - a_i[i])**2 + (y - b_i[i])**2 + c_i[i]*x*y
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([2*(x - a_i[i]) + c_i[i]*y, 2*(y - b_i[i]) + c_i[i]*x])
    def hess_i(w, i):
        return np.array([[2.0, c_i[i]], [c_i[i], 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(J=J, gradJ=gradJ, loss_i=loss_i, grad_i=grad_i,
                hess_i=hess_i, hessvec_i=hessvec_i, grad_full=grad_full,
                W_STAR=np.array([1.0, -2.0]), label="ben condizionata")


def make_preset_very_ill():
    """quad_very_ill: J = 100(w1-1)^2 + (w2+2)^2  (kappa~100)"""
    def J(w):
        x, y = w[0], w[1]
        return 100*(x - 1)**2 + (y + 2)**2
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([200*(x - 1), 2*(y + 2)])
    def hessJ(w):
        return np.array([[200.0, 0.0], [0.0, 2.0]])
    np.random.seed(SEED)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
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
    return dict(J=J, gradJ=gradJ, loss_i=loss_i, grad_i=grad_i,
                hess_i=hess_i, hessvec_i=hessvec_i, grad_full=grad_full,
                W_STAR=np.array([1.0, -2.0]), label="molto mal condizionata")


def make_preset_offdiag():
    """quad_offdiag: J = (x-1)^2 + (y+2)^2 + 0.5(x-1)(y+2), x=w0-1, y=w1+2"""
    def J(w):
        x, y = w[0] - 1, w[1] + 2
        return x*x + y*y + 0.5*x*y
    def gradJ(w):
        x, y = w[0] - 1, w[1] + 2
        return np.array([2*x + 0.5*y, 2*y + 0.5*x])
    def hessJ(w):
        return np.array([[2.0, 0.5], [0.5, 2.0]])
    np.random.seed(SEED)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    raw_c = 0.5 + 0.05 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    c_i = raw_c - np.mean(raw_c) + 0.5
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
    return dict(J=J, gradJ=gradJ, loss_i=loss_i, grad_i=grad_i,
                hess_i=hess_i, hessvec_i=hessvec_i, grad_full=grad_full,
                W_STAR=np.array([1.0, -2.0]), label="termine incrociato")


# ----------------------------------------------------------------------
# ALGORITMI (codice esatto dell'app + bbccv.py)
# ----------------------------------------------------------------------
def dynamic_gd(p, w0, theta, max_iter, alpha, batch0):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    for k in range(max_iter):
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
                    g_new = np.mean([grad_i(w_new, i) for i in indices], axis=0)
                    if np.dot(g_new, d) >= c2 * gd:
                        break
                step *= 0.5
            else:
                step = 0.0
        w = w + step * d
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes


def newton_cg(p, w0, theta, max_iter, alpha, batch0, R, maxcg):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
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
    for k in range(max_iter):
        indices_S = np.random.choice(N, size=n, replace=False)
        g = np.mean([grad_i(w, i) for i in indices_S], axis=0)
        n_h = min(max(1, int(round(R * n))), N)
        indices_H = np.random.choice(indices_S, size=n_h, replace=False)
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
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        indices_new = np.random.choice(N, size=n, replace=False)
        g_new = np.mean([grad_i(w, i) for i in indices_new], axis=0)
        var_vec = np.var([grad_i(w, i) for i in indices_new], axis=0, ddof=1) \
            if n > 1 else np.zeros_like(g_new)
        V_norm1, gg_new = np.sum(var_vec), np.dot(g_new, g_new)
        if gg_new > 1e-16 and V_norm1 / n > theta**2 * gg_new:
            n = min(int(np.ceil(V_norm1 / (theta**2 * gg_new))) + 1, N)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes


def newton_l1(p, w0, theta, max_iter, alpha, batch0, nu, sigma, R, maxcg, eta=0.5):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
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
    for k in range(max_iter):
        indices_S = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)
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
        n_h = max(1, int(round(R * n)))
        n_h = min(n_h, N)
        indices_H = np.random.choice(indices_S, size=n_h, replace=False)
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
            p = r.copy()
            rr = np.dot(r, r)
            for _ in range(maxcg):
                Hp = Hv_free(p)
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


def bb_ccv(p, w0, theta, max_iter, alpha, batch0):
    """BB-CCV, codice esatto di altro/script/bbccv.py"""
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    loss_i, grad_i, grad_full = p["loss_i"], p["grad_i"], p["grad_full"]
    w_prev = w.copy()
    g_prev = None
    for k in range(max_iter):
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
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes


# ----------------------------------------------------------------------
# ESECUZIONE + STAMPA
# ----------------------------------------------------------------------
PRESETS = [("ben  (kappa~1.1)", make_preset_well),
           ("mal  (kappa~100)", make_preset_very_ill),
           ("offd (termine incrociato)", make_preset_offdiag)]

METHODS = [("Dynamic GD", dynamic_gd, (THETA, MAX_ITER, ALPHA, BATCH0)),
           ("Newton-CG", newton_cg, (THETA, MAX_ITER, ALPHA, BATCH0, R_, MAXCG)),
           ("Newton-CG L1", newton_l1, (THETA, MAX_ITER, ALPHA, BATCH0, NU, SIGMA, R_, MAXCG)),
           ("BB-CCV", bb_ccv, (THETA, MAX_ITER, ALPHA, BATCH0))]

for pname, pmaker in PRESETS:
    print(f"\n===== PROBLEMA {pname} =====")
    for mname, mfun, margs in METHODS:
        np.random.seed(SEED)  # dataset rigenerato da seed 42
        p = pmaker()
        hist, bs = mfun(p, W0, *margs)
        errs = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
        print(f"\n-- {mname} (iter={len(hist)-1}, batch_finale={bs[-1]}) --")
        for k, e in enumerate(errs):
            print(f"{k:2d} {e:.4e}")



