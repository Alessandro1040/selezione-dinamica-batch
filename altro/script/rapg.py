import numpy as np

def rapg(w0, theta, max_iter, alpha, batch0):
    """
    RAPG — Robust Adaptive Preconditioned Gradient

    Ibrido che mantiene il core BB-CCV e aggiunge:
    - Precondizionamento diagonale limitato per mal condizionate
    - Regula Falsi per 1D (più robusto del secante)
    - Campionamento dinamico CCV
    """
    w = np.array(w0, dtype=float)
    N = max(len([grad_i(w, i) for i in range(max(batch0, 1))]), batch0)  # infer N
    n = min(max(batch0, 1), N)
    dim = len(w0)
    history = [w.copy().tolist()]
    batch_sizes = [n]

    # BB state
    w_prev = w.copy()
    g_prev = None

    # Preconditioner (bounded RMSprop)
    v = np.zeros_like(w)
    beta2 = 0.99
    eps = 1e-10

    # 1D: Regula Falsi state
    w_lo = w.copy()
    g_lo = None
    w_hi = None
    g_hi = None

    for k in range(max_iter):
        indices = np.random.choice(N, size=n, replace=False)
        grads = np.array([grad_i(w, i) for i in indices])
        g = np.mean(grads, axis=0)

        def J_batch(w_curr):
            return np.mean([loss_i(w_curr, i) for i in indices])

        # === BB step size ===
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

        # === Direction selection ===
        if dim == 1:
            # 1D: Regula Falsi for root finding g(w)=0
            if g_lo is None:
                g_lo = g.copy()
                w_lo = w.copy()

            # Try to establish bracket
            if g_hi is None and k > 0:
                test_w = w - alpha * np.sign(g[0])
                test_g = np.mean([grad_i(test_w, i) for i in indices], axis=0)
                if test_g[0] * g[0] < 0:
                    w_hi = test_w.copy()
                    g_hi = test_g.copy()

            if g_hi is not None and g_lo is not None and abs(g_hi[0] - g_lo[0]) > 1e-14:
                # Regula falsi: interpolate between bracket points
                w_new_1d = (w_lo[0] * g_hi[0] - w_hi[0] * g_lo[0]) / (g_hi[0] - g_lo[0])
                d = np.array([w_new_1d - w[0]])
                # Safeguard
                if abs(d[0]) > alpha * 20:
                    d = np.array([-np.sign(g[0]) * step])
            else:
                d = np.array([-np.sign(g[0]) * step])

            # Update bracket
            if g[0] * g_lo[0] > 0:
                w_lo = w.copy()
                g_lo = g.copy()
            elif g_hi is not None and g[0] * g_hi[0] > 0:
                w_hi = w.copy()
                g_hi = g.copy()
        else:
            # 2D+: Preconditioned BB gradient
            v = beta2 * v + (1 - beta2) * (g ** 2)
            v_hat = v / (1 - beta2 ** (k + 1))
            precond = np.clip(1.0 / (np.sqrt(v_hat) + eps), 0.1, 10.0)
            d = -precond * g * step

        # === Line search ===
        c1 = 1e-4
        J_curr = J_batch(w)

        if dim == 1:
            gTd = np.dot(g, d)
            step_size = 1.0
            if gTd < -1e-14:
                for _ in range(20):
                    w_new = w + step_size * d
                    if J_batch(w_new) <= J_curr + c1 * step_size * gTd:
                        break
                    step_size *= 0.5
                else:
                    step_size = 0.0
            else:
                step_size = 0.0
            w = w + step_size * d
        else:
            g_norm2 = np.dot(g, g)
            if g_norm2 > 1e-16:
                for _ in range(30):
                    w_new = w + d
                    if J_batch(w_new) <= J_curr - c1 * np.dot(d, g):
                        break
                    d *= 0.5
                else:
                    d = np.zeros_like(w)
            w = w + d

        # === CCV Dynamic Sampling ===
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

history, batch_sizes = rapg(w0, theta, max_iter, alpha, batch0)
