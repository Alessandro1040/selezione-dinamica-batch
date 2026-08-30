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

def newton_cg(w0, theta, max_iter, alpha, batch0, R, maxcg):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history, batch_sizes = [w.copy().tolist()], [n]
    resize_points = []
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
        var_vec = np.var([grad_i(w, i) for i in indices_new], axis=0, ddof=1) if n > 1 else np.zeros_like(g_new)
        V_norm1, gg_new = np.sum(var_vec), np.dot(g_new, g_new)
        if gg_new > 1e-16 and V_norm1 / n > theta**2 * gg_new:
            resize_points.append(w.copy().tolist())
            n = min(int(np.ceil(V_norm1 / (theta**2 * gg_new))) + 1, N)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points

history, batch_sizes, resize_points = newton_cg(w0, theta, max_iter, alpha, batch0, R, maxcg)
