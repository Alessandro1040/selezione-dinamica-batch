#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark NSynth: logistic regression multinomiale (11 famiglie strumentali)
sulle features audio mel, risolta con i quattro metodi della tesi:
  - Dynamic GD (CCV + line search di Wolfe)
  - Newton-CG (Hessiana sottocampionata + CG adattivo)
  - Newton-CG L1 (subgradiente + active set + proiezione sull'ortante)
  - BB-CCV (Barzilai-Borwein + CCV + Armijo)

Training = split "valid" di NSynth (12 678 clip), test = split "test"
(4096 clip). Gli strumenti dei due split sono disgiunti (per progettazione
del dataset): il test misura la generalizzazione a strumenti mai visti.

Uso:
  python3 run_benchmark.py <dir_feat> <out_dir>
"""
import os, sys, time, json
import numpy as np

SEED = 42
THETA = 0.5       # tolleranza CCV (default app)
ALPHA = 1.0       # passo iniziale line search
BATCH0 = 64       # batch iniziale
MAX_ITER = 300    # budget iterazioni
TOL = 1e-6        # arresto: ||grad_full||_2 < TOL
R = 0.1           # rapporto |H_k| / |S_k|
MAXCG = 50        # limite iterazioni CG
LAM = 1e-4        # regolarizzazione L2 (Dynamic GD, NCG, BB-CCV)
NU = 1e-3         # penalita' L1 (Newton-CG L1)


def load_data(dir_feat):
    X_tr = np.load(os.path.join(dir_feat, "valid_X.npy"))
    y_tr = np.load(os.path.join(dir_feat, "valid_y.npy"))
    X_te = np.load(os.path.join(dir_feat, "test_X.npy"))
    y_te = np.load(os.path.join(dir_feat, "test_y.npy"))
    with open(os.path.join(dir_feat, "valid_families.txt")) as f:
        families = [l for l in f.read().splitlines() if l]
    return X_tr, y_tr, X_te, y_te, families


def standardize(X_tr, X_te):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd

# ----------------------------------------------------------------------
# Problema: logistic regression multinomiale
# parametro w = W.ravel(), W di forma (C, D), D = d+1 (bias in ultima colonna)
# ----------------------------------------------------------------------
class LogReg:
    def __init__(self, X, y, C, lam):
        self.X = np.hstack([X, np.ones((X.shape[0], 1))])  # + bias
        self.y = y
        self.n, self.D = self.X.shape
        self.C = C
        self.p = C * self.D
        self.lam = lam

    def unpack(self, w):
        return w.reshape(self.C, self.D)

    def logits_batch(self, w, idx):
        return self.X[idx] @ self.unpack(w).T          # (b, C)

    def probs_batch(self, w, idx):
        z = self.logits_batch(w, idx)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def J_batch(self, w, idx):
        P = self.probs_batch(w, idx)
        l = -np.log(np.maximum(P[np.arange(len(idx)), self.y[idx]], 1e-15))
        reg = 0.5 * self.lam * np.sum(self.unpack(w)[:, :-1] ** 2)
        return float(l.mean()) + reg

    def grad_batch(self, w, idx):
        P = self.probs_batch(w, idx)
        Y = np.zeros_like(P); Y[np.arange(len(idx)), self.y[idx]] = 1.0
        G = (P - Y).T @ self.X[idx] / len(idx)          # (C, D)
        G[:, :-1] += self.lam * self.unpack(w)[:, :-1]
        return G.ravel()

    def per_example_grads(self, w, idx):
        P = self.probs_batch(w, idx)
        Y = np.zeros_like(P); Y[np.arange(len(idx)), self.y[idx]] = 1.0
        d = P - Y                                        # (b, C)
        return np.einsum("bi,bj->bij", d, self.X[idx])   # (b, C, D)

    def grad_full(self, w):
        return self.grad_batch(w, np.arange(self.n))

    def loss_full(self, w):
        return self.J_batch(w, np.arange(self.n))

    def hessvec_batch(self, w, v, idx):
        V = self.unpack(v)
        P = self.probs_batch(w, idx)
        Xb = self.X[idx]
        q = Xb @ V.T                                     # (b, C)
        r = P * q - P * (P * q).sum(axis=1, keepdims=True)   # (Dp)@q
        Hv = r.T @ Xb / len(idx)                         # (C, D)
        Hv[:, :-1] += self.lam * V[:, :-1]
        return Hv.ravel()

# ----------------------------------------------------------------------
# Blocchi comuni
# ----------------------------------------------------------------------
def ccv_update(per_grads, g_flat, n, N, theta):
    """Nuova dimensione del batch secondo la CCV (come nell'app)."""
    V_norm1 = float(np.var(per_grads, axis=0, ddof=1).sum()) if n > 1 else 0.0
    gg = float(np.dot(g_flat, g_flat))
    if gg > 1e-16 and V_norm1 / n > theta ** 2 * gg:
        return min(int(np.ceil(V_norm1 / (theta ** 2 * gg))) + 1, N)
    return n


def armijo_step(prob, w, d, g_flat, idx, alpha, c1=1e-4, max_bt=30):
    """Backtracking di Armijo sulla loss di batch."""
    J0 = prob.J_batch(w, idx)
    gd = float(np.dot(g_flat, d))
    step = alpha
    if gd >= 0:
        return 0.0
    for _ in range(max_bt):
        if prob.J_batch(w + step * d, idx) <= J0 + c1 * step * gd:
            return step
        step *= 0.5
    return 0.0


def wolfe_step(prob, w, d, g_flat, idx, alpha, c1=1e-4, c2=0.9, max_bt=30):
    """Condizioni di Wolfe sulla loss di batch (come nell'app)."""
    J0 = prob.J_batch(w, idx)
    gd = float(np.dot(g_flat, d))
    step = alpha
    if gd >= 0:
        return 0.0
    for _ in range(max_bt):
        wn = w + step * d
        if prob.J_batch(wn, idx) <= J0 + c1 * step * gd:
            gn = prob.grad_batch(wn, idx)
            if float(np.dot(gn, d)) >= c2 * gd:
                return step
        step *= 0.5
    return 0.0


# ----------------------------------------------------------------------
# 1) Dynamic GD (CCV + line search di Wolfe) — come simulazione_batch.py
# ----------------------------------------------------------------------
def dynamic_gd(prob, N, w0, max_iter, alpha, batch0, theta):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    hist_w, hist_n, hist_g = [w.copy()], [n], [prob.grad_full(w)]
    for k in range(max_iter):
        idx = np.random.choice(N, size=n, replace=False)
        g = prob.grad_batch(w, idx)
        pg = prob.per_example_grads(w, idx)
        n = ccv_update(pg, g, n, N, theta)
        d = -g
        step = wolfe_step(prob, w, d, g, idx, alpha)
        w = w + step * d
        hist_w.append(w.copy()); hist_n.append(n)
        gf = prob.grad_full(w); hist_g.append(gf)
        if np.linalg.norm(gf) < TOL:
            break
    return hist_w, hist_n, hist_g


# ----------------------------------------------------------------------
# 2) BB-CCV (Barzilai-Borwein + CCV + Armijo) — come tesi/bbccv.py
# ----------------------------------------------------------------------
def bb_ccv(prob, N, w0, max_iter, alpha, batch0, theta):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    hist_w, hist_n, hist_g = [w.copy()], [n], [prob.grad_full(w)]
    w_prev, g_prev = w.copy(), None
    for k in range(max_iter):
        idx = np.random.choice(N, size=n, replace=False)
        g = prob.grad_batch(w, idx)
        if k > 0 and g_prev is not None:
            s = w - w_prev
            y = g - g_prev
            sy = float(np.dot(s, y))
            if abs(sy) > 1e-14:
                step = float(np.clip(np.dot(s, s) / sy, alpha / 20., alpha * 5.))
            else:
                step = alpha
        else:
            step = alpha
        w_prev, g_prev = w.copy(), g.copy()
        d = -g
        step = armijo_step(prob, w, d, g, idx, step)
        w = w + step * d
        pg = prob.per_example_grads(w, idx)
        n = ccv_update(pg, g, n, N, theta)
        hist_w.append(w.copy()); hist_n.append(n)
        gf = prob.grad_full(w); hist_g.append(gf)
        if np.linalg.norm(gf) < TOL:
            break
    return hist_w, hist_n, hist_g

# ----------------------------------------------------------------------
# 3) Newton-CG (Hessiana sottocampionata + CG adattivo) — come l'app
# ----------------------------------------------------------------------
def cg(A, b, gamma, maxcg):
    x = np.zeros_like(b)
    r = b - A(x)
    p = r.copy()
    rr = float(np.dot(r, r))
    for _ in range(maxcg):
        Ap = A(p)
        pHp = float(np.dot(p, Ap))
        if pHp <= 1e-14:
            break
        al = rr / pHp
        x = x + al * p
        r_new = r - al * Ap
        rr_new = float(np.dot(r_new, r_new))
        if rr_new <= gamma * float(np.dot(x, x)) + 1e-16:
            return x
        beta = rr_new / rr
        p = r_new + beta * p
        r, rr = r_new, rr_new
    return x


def newton_cg(prob, N, w0, max_iter, alpha, batch0, theta, R, maxcg):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    hist_w, hist_n, hist_g = [w.copy()], [n], [prob.grad_full(w)]
    for k in range(max_iter):
        idx_S = np.random.choice(N, size=n, replace=False)
        g = prob.grad_batch(w, idx_S)
        n_h = min(max(1, int(round(R * n))), N)
        idx_H = np.random.choice(idx_S, size=n_h, replace=False)   # H_k ⊆ S_k
        Hv = lambda v: prob.hessvec_batch(w, v, idx_H)             # noqa: E731
        p0 = -g
        p0_norm2 = float(np.dot(p0, p0))
        gamma = 0.0
        if p0_norm2 > 1e-16 and n_h > 1:
            Hp0 = np.array([prob.hessvec_batch(w, p0, [i]) for i in idx_H])
            gamma = float(np.var(Hp0, axis=0, ddof=1).sum()) / (n_h * p0_norm2)
        d = cg(Hv, -g, gamma, maxcg)
        step = wolfe_step(prob, w, d, g, idx_S, alpha)
        w = w + step * d
        # CCV con un campione fresco (come nell'app)
        idx_new = np.random.choice(N, size=n, replace=False)
        g_new = prob.grad_batch(w, idx_new)
        pg = prob.per_example_grads(w, idx_new)
        n = ccv_update(pg, g_new, n, N, theta)
        hist_w.append(w.copy()); hist_n.append(n)
        gf = prob.grad_full(w); hist_g.append(gf)
        if np.linalg.norm(gf) < TOL:
            break
    return hist_w, hist_n, hist_g



# ----------------------------------------------------------------------
# 4) Newton-CG L1 (subgradiente + active set + proiezione ortante) — come l'app
# ----------------------------------------------------------------------
def project_orthant(v, z):
    res = v.copy()
    for i in range(len(v)):
        if z[i] != 0 and np.sign(res[i]) != z[i]:
            res[i] = 0.0
    return res


def subgrad_l1(prob0, v, nu):
    """Subgradiente della funzione F = J + nu*||w||_1 (min-norm)."""
    gJ = prob0.grad_full(v)
    g = np.zeros_like(v)
    for i in range(len(v)):
        if v[i] > 0:
            g[i] = gJ[i] + nu
        elif v[i] < 0:
            g[i] = gJ[i] - nu
        elif gJ[i] < -nu:
            g[i] = gJ[i] + nu
        elif gJ[i] > nu:
            g[i] = gJ[i] - nu
        else:
            g[i] = 0.0
    return g


def cg_free(Hv_free, g_free, tol, maxcg):
    """CG nel sottospazio libero: risolve H d = -g (come nell'app L1)."""
    d_free = np.zeros_like(g_free)
    r = -g_free.copy()
    p = r.copy()
    rr = float(np.dot(r, r))
    for _ in range(maxcg):
        Hp = Hv_free(p)
        pHp = float(np.dot(p, Hp))
        if pHp <= 1e-14:
            if np.linalg.norm(d_free) < 1e-14:
                d_free = -g_free.copy()
            break
        al = rr / pHp
        d_free = d_free + al * p
        r_new = r - al * Hp
        rr_new = float(np.dot(r_new, r_new))
        if np.sqrt(rr_new) <= tol:
            break
        beta = rr_new / rr
        p = r_new + beta * p
        r, rr = r_new, rr_new
    return d_free


def newton_l1(prob0, N, w0, max_iter, alpha, batch0, nu, sigma,
              maxcg, R, theta, eta=0.5):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    hist_w, hist_n, hist_g = [w.copy()], [n], [np.linalg.norm(subgrad_l1(prob0, w, nu))]

    def F_batch(v, idx):
        return prob0.J_batch(v, idx) + nu * np.sum(np.abs(v))

    def subgrad_batch(v, idx):
        grads = prob0.per_example_grads(v, idx).reshape(len(idx), -1)
        gJ = grads.mean(axis=0)
        g = np.zeros_like(v)
        for i in range(len(v)):
            if v[i] > 0:
                g[i] = gJ[i] + nu
            elif v[i] < 0:
                g[i] = gJ[i] - nu
            elif gJ[i] < -nu:
                g[i] = gJ[i] + nu
            elif gJ[i] > nu:
                g[i] = gJ[i] - nu
            else:
                g[i] = 0.0
        return g

    for k in range(max_iter):
        idx_S = np.random.choice(N, size=n, replace=False)
        g_batch = prob0.grad_batch(w, idx_S)
        z = np.where(w > 0, 1, np.where(w < 0, -1,
                        np.where(g_batch < -nu, 1, np.where(g_batch > nu, -1, 0))))
        sg = subgrad_batch(w, idx_S)
        if np.linalg.norm(sg) < 1e-10:
            break
        n_h = min(max(1, int(round(R * n))), N)
        idx_H = np.random.choice(idx_S, size=n_h, replace=False)
        free = (z != 0)
        d = np.zeros_like(w)
        if np.any(free):
            g_free = sg[free]

            def Hv_full(vf):
                return prob0.hessvec_batch(w, vf, idx_H)

            def Hv_free(vf):
                v_full = np.zeros_like(w)
                v_full[free] = vf
                return Hv_full(v_full)[free]

            tol_cg = eta * np.linalg.norm(g_free)
            d[free] = cg_free(Hv_free, g_free, tol_cg, maxcg)
        step = alpha
        F_w = F_batch(w, idx_S)
        sg_d = float(np.dot(sg, d))
        if sg_d >= 0:
            d = -sg
            sg_d = -float(np.dot(sg, sg))
        w_new = w.copy()
        for _ in range(20):
            w_trial = project_orthant(w + step * d, z)
            if F_batch(w_trial, idx_S) <= F_w + sigma * step * sg_d:
                w_new = w_trial
                break
            step *= 0.5
            if step < 1e-12:
                break
        w = w_new
        idx_new = np.random.choice(N, size=n, replace=False)
        g_new = prob0.grad_batch(w, idx_new)
        pg = prob0.per_example_grads(w, idx_new).reshape(len(idx_new), -1)
        n = ccv_update(pg, g_new, n, N, theta)
        hist_w.append(w.copy()); hist_n.append(n)
        hist_g.append(np.linalg.norm(subgrad_l1(prob0, w, nu)))
        if hist_g[-1] < TOL:
            break
    return hist_w, hist_n, hist_g


# ----------------------------------------------------------------------
# Runner + metriche + figure
# ----------------------------------------------------------------------
def predict(prob, w, X):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.argmax(Xa @ prob.unpack(w).T, axis=1)


def accuracy(prob_te, w, X_te, y_te):
    return float(np.mean(predict(prob_te, w, X_te) == y_te))


def main():
    if len(sys.argv) != 3:
        sys.exit("uso: run_benchmark.py <dir_feat> <out_dir>")
    dir_feat, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    X_tr, y_tr, X_te, y_te, families = load_data(dir_feat)
    X_tr, X_te = standardize(X_tr, X_te)
    C = len(families)
    N = X_tr.shape[0]
    prob = LogReg(X_tr, y_tr, C, LAM)       # obiettivo L2 (3 metodi)
    prob0 = LogReg(X_tr, y_tr, C, 0.0)      # obiettivo L1 (Newton-CG L1)
    prob_te = LogReg(X_te, y_te, C, 0.0)
    w0 = np.zeros(prob.p)
    print(f"train={N}, test={X_te.shape[0]}, classi={C}, "
          f"parametri={prob.p}, d={X_tr.shape[1]}")

    methods = [
        ("Dynamic GD", lambda: dynamic_gd(prob, N, w0, MAX_ITER, ALPHA, BATCH0, THETA)),
        ("Newton-CG", lambda: newton_cg(prob, N, w0, MAX_ITER, ALPHA, BATCH0, THETA, R, MAXCG)),
        ("Newton-CG L1", lambda: newton_l1(prob0, N, w0, MAX_ITER, ALPHA, BATCH0, NU, 1e-4, MAXCG, R, THETA)),
        ("BB-CCV", lambda: bb_ccv(prob, N, w0, MAX_ITER, ALPHA, BATCH0, THETA)),
    ]

    results = {}
    losses = {}
    accs = {}
    batches = {}
    for name, fn in methods:
        np.random.seed(SEED)
        t0 = time.time()
        hist_w, hist_n, hist_g = fn()
        dt = time.time() - t0
        wf = hist_w[-1]
        if name == "Newton-CG L1":
            loss_f = prob0.loss_full(wf) + NU * np.sum(np.abs(wf))
            nz = int(np.sum(wf != 0))
        else:
            loss_f = prob.loss_full(wf)
            nz = None
        acc = accuracy(prob_te, wf, X_te, y_te)
        results[name] = {
            "iterazioni": len(hist_w) - 1,
            "loss_finale": loss_f,
            "grad_norm_finale": float(np.linalg.norm(hist_g[-1])),
            "batch_finale": int(hist_n[-1]),
            "batch_max": int(max(hist_n)),
            "tempo_s": dt,
            "accuracy_test": acc,
            "non_zero": nz,
        }
        losses[name] = [float(prob.loss_full(w)) for w in hist_w]
        accs[name] = [accuracy(prob_te, w, X_te, y_te) for w in hist_w]
        batches[name] = [int(nn) for nn in hist_n]
        print(f"{name:15s} iter={results[name]['iterazioni']:3d} "
              f"acc={acc*100:5.2f}%  batch_fin={results[name]['batch_finale']:5d} "
              f"loss={loss_f:.4f}  tempo={dt:5.2f}s"
              + (f"  nnz={nz}" if nz is not None else ""))

    # salvataggio
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"risultati": results, "famiglie": families}, f, indent=2)
    np.savez(os.path.join(out_dir, "curves.npz"),
             losses=losses, accs=accs, batches=batches)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Dynamic GD": "#1f77b4", "Newton-CG": "#d62728",
              "Newton-CG L1": "#2ca02c", "BB-CCV": "#9467bd"}
    ks = np.arange(0, MAX_ITER + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, fn in methods:
        acc = accs[name]
        ax.plot(ks[:len(acc)], np.array(acc) * 100, lw=1.8, label=name, color=colors[name])
    ax.axhline(20.8, color="gray", ls="--", lw=1.2, label="Classe maggioritaria")
    ax.set_xlabel(r"Iterazione $k$")
    ax.set_ylabel(r"Accuratezza sul test (\%)")
    ax.set_title("NSynth: accuratezza di test vs iterazioni")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nsynth_accuracy.pdf"))
    fig.savefig(os.path.join(out_dir, "nsynth_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, fn in methods:
        b = batches[name]
        ax.step(np.arange(len(b)), b, where="mid", lw=1.6, label=name, color=colors[name])
    ax.set_xlabel(r"Iterazione $k$")
    ax.set_ylabel(r"Dimensione del batch $n_k$")
    ax.set_title("NSynth: dinamica del batch (CCV)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nsynth_batch.pdf"))
    fig.savefig(os.path.join(out_dir, "nsynth_batch.png"), dpi=150)
    plt.close(fig)

    print("\nOK: risultati e figure in", out_dir)


if __name__ == "__main__":
    main()

