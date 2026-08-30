import numpy as np

def newton_l1(w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg, eta=0.5, max_consec=None, reuse_hessian=True, max_hessian_reuse=None):
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
    for k in range(max_iter):
        if need_resample_S or indices_S is None or (max_consec is not None and used_S >= max_consec):
            if not need_resample_S and indices_S is not None:
                resample_pts.append(w.copy().tolist())
            indices_S = np.random.choice(N, size=n, replace=False)
            n_h = max(1, int(round(R * n)))
            n_h = min(n_h, N)
            used_S = 0
            need_resample_S = False
        grads = np.array([grad_i(w, i) for i in indices_S])
        g_batch = np.mean(grads, axis=0)

        # H_k indipendente da S_k: si ricampiona secondo max_hessian_reuse
        if need_resample_H or indices_H is None or (max_hessian_reuse is not None and used_H >= max_hessian_reuse):
            n_h = max(1, int(round(R * n)))
            n_h = min(n_h, N)
            indices_H = np.random.choice(indices_S, size=n_h, replace=False)
            used_H = 0
            need_resample_H = False

        # CCV sul campione RIUSATO al punto corrente (gratis: usa i gradienti del passo)
        if n > 1:
            var_vec = np.var(grads, axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(g_batch)
        V_norm1 = np.sum(var_vec)
        gg = np.dot(g_batch, g_batch)
        if gg > 1e-16:
            if V_norm1 / n > theta**2 * gg:
                resize_points.append(w.copy().tolist())
                n_new = int(np.ceil(V_norm1 / (theta**2 * gg))) + 1
                n = min(n_new, N)
                need_resample_S = True
                used_S = 0
                need_resample_H = True
                used_H = 0
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

            # Hessiana esplicita (versione precedente)
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
        used_S += 1
        used_H += 1
        history.append(w.copy().tolist())
        batch_sizes.append(n)
        if np.linalg.norm(grad_full(w)) < 1e-6:
            break
    return history, batch_sizes, resize_points, resample_pts

max_consec = 10  # k = max iterazioni consecutive sullo stesso mini-batch (None = illimitato)
reuse_hessian = False  # True: H_k legato a S_k; False: H_k indipendente da S_k
max_hessian_reuse = None  # max riusi consecutivi di H_k (None = illimitato)

history, batch_sizes, resize_points, resample_pts = newton_l1(w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg, eta, max_consec, reuse_hessian, max_hessian_reuse)
