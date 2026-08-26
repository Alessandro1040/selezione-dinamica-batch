import numpy as np

def newton_l1(w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg, eta, desc_tol, desc_min_abs, desc_patience, desc_freq):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
    resample_pts = []
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
    indices_S = None
    indices_H = None
    used_S = 0
    used_H = 0
    need_resample_S = True
    need_resample_H = True
    Jb_prev = None
    patience_S = 0
    patience_H = 0
    patience_dec = 0
    desc_hist = []
    m_actual = [0]
    desc_resample = False
    for k in range(max_iter):
        if need_resample_S or indices_S is None:
            if desc_resample:
                resample_pts.append(w.copy().tolist())
                desc_resample = False
            n = min(n, N)
            indices_S = np.random.choice(N, size=n, replace=False)
            n_h = min(max(1, int(round(R * n))), N)
            used_S = 0
            need_resample_S = False
            Jb_prev = None
            indices_H = np.random.choice(indices_S, size=n_h, replace=False)
            used_H = 0
            need_resample_H = False
        used_S += 1
        m_actual.append(used_S)
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)
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
                resize_points.append(w.copy().tolist())
                need_resample_S = True
                need_resample_H = True
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
        free = (z != 0)
        d = np.zeros_like(w)
        if np.any(free):
            g_free = sg[free]
            # Hessiana esplicita
            hessians = np.array([hess_i(w, i) for i in indices_H])
            H = np.mean(hessians, axis=0)
            H_free = H[np.ix_(free, free)]
            tol_cg = eta * np.linalg.norm(g_free)
            d_free = np.zeros(np.sum(free))
            r = -g_free.copy()
            p = r.copy()
            rr = np.dot(r, r)
            for _ in range(maxcg):
                Hp = H_free @ p
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
        if k % desc_freq == 0:
            Jb = float(F_batch(w, indices_S))
            desc_hist.append(Jb)
            if Jb_prev is not None:
                improved = (Jb_prev - Jb >= desc_tol * abs(Jb_prev) + desc_min_abs)
                if improved:
                    patience_S = 0
                    patience_H = 0
                else:
                    patience_S += 1
                    patience_H += 1
                    if patience_S >= desc_patience or patience_H >= desc_patience:
                        if not need_resample_S:
                            desc_resample = True
                        need_resample_S = True
                        patience_S = 0
                        need_resample_H = True
                        patience_H = 0
            Jb_prev = Jb
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points, resample_pts, m_actual, desc_hist

desc_tol = 0.0001
desc_min_abs = 0
desc_patience = 1
desc_freq = 1
history, batch_sizes, resize_points, resample_pts, m_actual, desc_hist = newton_l1(w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg, eta, desc_tol, desc_min_abs, desc_patience, desc_freq)
