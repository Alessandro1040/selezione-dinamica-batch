import numpy as np

def bb_dynamic_gd(w0, theta, max_iter, alpha, batch0, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
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
        if need_resample or indices is None:
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
        step = alpha
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

val_pct = 0.2
val_tol = 0.0001
val_patience = 3
val_freq = 1
val_min_abs = 0
val_strategy = 'fixed'
history, batch_sizes, resize_points, m_actual, val_hist = bb_dynamic_gd(w0, theta, max_iter, alpha, batch0, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy)
