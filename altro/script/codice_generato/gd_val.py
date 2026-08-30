import numpy as np

def dynamic_gd(w0, theta, max_iter, alpha, batch0, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    resample_pts = []
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
    val_resample = False
    for k in range(max_iter):
        if need_resample or indices is None:
            if val_resample:
                resample_pts.append(w.copy().tolist())
                val_resample = False
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
                    if not need_resample:
                        val_resample = True
                    need_resample = True
                    patience = 0
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, resample_pts, m_actual, val_hist

val_pct = 0.2
val_tol = 0.0001
val_patience = 3
val_freq = 1
val_min_abs = 0
val_strategy = 'fixed'
history, batch_sizes, resize_points, resample_pts, m_actual, val_hist = dynamic_gd(w0, theta, max_iter, alpha, batch0, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy)
