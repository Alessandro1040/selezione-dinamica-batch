import numpy as np

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

def newton_cg(w0, theta, max_iter, alpha, batch0, R, maxcg, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
    resize_points = []
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
        if need_resample_S or indices_S is None:
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
        # CCV sul campione RIUSATO al punto corrente (gratis: usa i gradienti del passo)
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

val_pct = 0.2
val_tol = 0.0001
val_patience = 3
val_freq = 1
val_min_abs = 0
val_strategy = 'fixed'
history, batch_sizes, resize_points, m_actual, val_hist = newton_cg(w0, theta, max_iter, alpha, batch0, R, maxcg, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy)
