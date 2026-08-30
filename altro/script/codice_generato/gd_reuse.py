import numpy as np

def dynamic_gd(w0, theta, max_iter, alpha, batch0, max_consec):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    resample_pts = []
    indices = None
    used = 0
    need_resample = True
    for k in range(max_iter):
        if need_resample or indices is None or (max_consec is not None and used >= max_consec):
            if not need_resample and indices is not None:
                resample_pts.append(w.copy().tolist())
            indices = np.random.choice(N, size=n, replace=False)
            used = 0
            need_resample = False
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
        used += 1
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, resample_pts

max_consec = None  # k = max iterazioni consecutive sullo stesso mini-batch (None = illimitato)

history, batch_sizes, resize_points, resample_pts = dynamic_gd(w0, theta, max_iter, alpha, batch0, max_consec)
