import numpy as np

def bb_dynamic_gd(w0, theta, max_iter, alpha, batch0):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
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
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points

history, batch_sizes, resize_points = bb_dynamic_gd(w0, theta, max_iter, alpha, batch0)
