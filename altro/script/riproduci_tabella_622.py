# =====================================================================
# Riproduzione della Tabella 6.22 (tab:riuso_valid_confronto) della tesi
# "Selezione Dinamica della Dimensione del Campione in Metodi di
# Ottimizzazione per il Machine Learning".
#
# CODICE ESATTO DELL'APP:
#  - preset: LOSS_PRESETS di visualizzazione.html (costruzione dataset)
#  - algoritmi: codici Python generati dall'app (varianti *Validation),
#    estratti in altro/script/validation_codice_generato/:
#      Dynamic GD    -> gd_wolfe.py       (line search di Wolfe)
#      BB-CCV        -> bb_armijo.py      (line search di Armijo)
#      Newton-CG     -> newton_cg_tied.py (H_k legato a S_k)
#      Newton-CG L1  -> newton_l1_tied.py (H_k legato a S_k)
#
# Le 16 combinazioni (4 problemi x 4 algoritmi) e le 5 colonne della
# Tabella 6.22 (base, M=inf, def., P=1,p=0.1,dyn, P=3,p=0.1,dyn) sono
# generate esattamente come in altro/script/gen_tabelle_riuso_validation.py.
# =====================================================================
import numpy as np

# ---- Preambolo dell'app (runAlgorithm): parametri globali a livello
# ---- di modulo, letti dal codice generato ----------------------------
N = 200
W0 = [2.0, -3.0]
ALPHA = 0.1
THETA = 0.5
BATCH0 = 5
MAX_ITER = 30
SEED = 42
R_ = 0.2
MAXCG = 10
NU = 0.1
SIGMA = 0.1
ETA = 0.5

# =====================================================================
# PRESET (codice esatto dell'app, LOSS_PRESETS)
# =====================================================================
def _make_preset_well(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    raw_c = 0.1 + 0.05 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    c_i = raw_c - np.mean(raw_c) + 0.1
    def J(w):
        x, y = w[0], w[1]
        return (x - 1)**2 + (y + 2)**2 + 0.1*x*y
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([2*(x - 1) + 0.1*y, 2*(y + 2) + 0.1*x])
    def hessJ(w):
        return np.array([[2.0, 0.1], [0.1, 2.0]])
    def loss_i(w, i):
        x, y = w[0], w[1]
        return (x - a_i[i])**2 + (y - b_i[i])**2 + c_i[i] * x * y
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([2*(x - a_i[i]) + c_i[i] * y, 2*(y - b_i[i]) + c_i[i] * x])
    def hess_i(w, i):
        return np.array([[2.0, c_i[i]], [c_i[i], 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="ben condizionata")

def _make_preset_ill(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    def J(w):
        x, y = w[0], w[1]
        return 20*(x - 1)**2 + (y + 2)**2
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([40*(x - 1), 2*(y + 2)])
    def hessJ(w):
        return np.array([[40.0, 0.0], [0.0, 2.0]])
    def loss_i(w, i):
        x, y = w[0], w[1]
        return 20*(x - a_i[i])**2 + (y - b_i[i])**2
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([40*(x - a_i[i]), 2*(y - b_i[i])])
    def hess_i(w, i):
        return np.array([[40.0, 0.0], [0.0, 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="mal condizionata")

def _make_preset_very_ill(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    def J(w):
        x, y = w[0], w[1]
        return 100*(x - 1)**2 + (y + 2)**2
    def gradJ(w):
        x, y = w[0], w[1]
        return np.array([200*(x - 1), 2*(y + 2)])
    def hessJ(w):
        return np.array([[200.0, 0.0], [0.0, 2.0]])
    def loss_i(w, i):
        x, y = w[0], w[1]
        return 100*(x - a_i[i])**2 + (y - b_i[i])**2
    def grad_i(w, i):
        x, y = w[0], w[1]
        return np.array([200*(x - a_i[i]), 2*(y - b_i[i])])
    def hess_i(w, i):
        return np.array([[200.0, 0.0], [0.0, 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="molto mal condizionata")

def _make_preset_offdiag(seed=SEED):
    np.random.seed(seed)
    raw_a = 1.0 + 0.2 * np.random.randn(N)
    raw_b = -2.0 + 0.2 * np.random.randn(N)
    raw_c = 0.5 + 0.05 * np.random.randn(N)
    a_i = raw_a - np.mean(raw_a) + 1.0
    b_i = raw_b - np.mean(raw_b) - 2.0
    c_i = raw_c - np.mean(raw_c) + 0.5
    def J(w):
        x, y = w[0] - 1, w[1] + 2
        return x*x + y*y + 0.5*x*y
    def gradJ(w):
        x, y = w[0] - 1, w[1] + 2
        return np.array([2*x + 0.5*y, 2*y + 0.5*x])
    def hessJ(w):
        return np.array([[2.0, 0.5], [0.5, 2.0]])
    def loss_i(w, i):
        x, y = w[0] - a_i[i], w[1] - b_i[i]
        return x*x + y*y + c_i[i]*x*y
    def grad_i(w, i):
        x, y = w[0] - a_i[i], w[1] - b_i[i]
        return np.array([2*x + c_i[i]*y, 2*y + c_i[i]*x])
    def hess_i(w, i):
        return np.array([[2.0, c_i[i]], [c_i[i], 2.0]])
    def hessvec_i(w, i, v):
        return hess_i(w, i) @ v
    def grad_full(w):
        return np.mean([grad_i(w, i) for i in range(N)], axis=0)
    return dict(N=N, J=J, gradJ=gradJ, hessJ=hessJ, loss_i=loss_i,
                grad_i=grad_i, hess_i=hess_i, hessvec_i=hessvec_i,
                grad_full=grad_full, W_STAR=np.array([1.0, -2.0]),
                label="termine incrociato")

PRESET_MAKERS = {
    "quad_well": _make_preset_well,
    "quad_ill": _make_preset_ill,
    "quad_very_ill": _make_preset_very_ill,
    "quad_offdiag": _make_preset_offdiag,
}
PRESET_LATEX = {
    "quad_well": "kappa~1.1", "quad_ill": "kappa~20",
    "quad_very_ill": "kappa~100", "quad_offdiag": "incrociato (kappa~1.67)",
}
ALGO_LATEX = {"gd": "Dynamic GD", "bb": "BB-CCV",
              "newton_cg": "Newton-CG", "newton_l1": "Newton-CG L1"}


# =====================================================================
# ALGORITMI: codice ESATTO generato dall'app (varianti *Validation).
# I sorgenti sono quelli estratti in validation_codice_generato/ (sono le
# stesse righe che compaiono nel pannello codice di visualizzazione.html).
# =====================================================================
GD_SRC = r'''
def dynamic_gd(w0, theta, max_iter, alpha, batch0, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
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
                    need_resample = True
                    patience = 0
        if np.linalg.norm(grad_full(w)) < 1e-6:
            history.append(w.copy().tolist())
            batch_sizes.append(n)
            break
        history.append(w.copy().tolist())
        batch_sizes.append(n)
    return history, batch_sizes, resize_points, m_actual, val_hist
'''


BB_SRC = r'''
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
'''


NEWTON_CG_SRC = r'''
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
'''


NEWTON_L1_SRC = r'''
def newton_l1(w0, theta, max_iter, alpha, batch0, nu, sigma, maxcg, eta, val_pct, val_tol, val_patience, val_freq, val_min_abs, val_strategy):
    w = np.array(w0, dtype=float)
    n = max(batch0, 2)
    history     = [w.copy().tolist()]
    batch_sizes = [n]
    resize_points = []
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
                n = min(n_new, len(train_idx))
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
'''


# =====================================================================
# CARICAMENTO ALGORITMI (stessa logica di make_functions di
# gen_tabelle_riuso_validation.py): si esegue il sorgente generato
# con le globali dell'app. Per la colonna 'base' si forza il
# ricampionamento a ogni iterazione (guard -> 'if True:').
# =====================================================================
ALGOS = {
    "gd":        ("dynamic_gd",     [],              GD_SRC),
    "bb":        ("bb_dynamic_gd",  [],              BB_SRC),
    "newton_cg": ("newton_cg",      [R_, MAXCG],     NEWTON_CG_SRC),
    "newton_l1": ("newton_l1",      [NU, SIGMA, MAXCG, ETA], NEWTON_L1_SRC),
}

def make_functions(p, algo, force_base=False):
    entry, extra, src = ALGOS[algo]
    if force_base:
        # base: ricampionamento a ogni iterazione (guard -> if True:)
        if algo in ("gd", "bb"):
            src = src.replace("if need_resample or indices is None:",
                              "if True:  # base: ricampionamento a ogni iterazione")
        else:
            src = src.replace("if need_resample_S or indices_S is None:",
                              "if True:  # base: ricampionamento a ogni iterazione")
    ns = {
        "np": np, "N": p["N"], "loss_i": p["loss_i"], "grad_i": p["grad_i"],
        "hess_i": p["hess_i"], "hessvec_i": p["hessvec_i"],
        "grad_full": p["grad_full"],
        "w0": W0, "alpha": ALPHA, "max_iter": MAX_ITER, "theta": THETA,
        "batch0": BATCH0, "R": R_, "maxcg": MAXCG, "nu": NU, "sigma": SIGMA,
        "eta": ETA,
    }
    exec(src, ns)
    return ns[entry], extra

def run_config(p, algo, hp, force_base=False):
    f, extra = make_functions(p, algo, force_base=force_base)
    args = [W0, THETA, MAX_ITER, ALPHA, BATCH0] + extra + [
        hp["val_pct"], hp["val_tol"], hp["val_patience"],
        hp["val_freq"], hp["val_min_abs"], hp["val_strategy"]]
    hist, batch_sizes, resize_points, m_actual, val_hist = f(*args)
    e = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
    resamples = sum(1 for x in m_actual if x == 1) - 1
    return e[-1], max(resamples, 0)

DEFAULT_HP = dict(val_pct=0.2, val_tol=1e-4, val_patience=3, val_freq=1,
                  val_min_abs=0.0, val_strategy="fixed")

# 5 colonne della Tabella 6.22
CONFIG_COLS = ["base", "M=inf", "def.", "P=1,p=0.1,dyn", "P=3,p=0.1,dyn"]

def config_hp(name):
    if name == "base":
        return dict(DEFAULT_HP, val_patience=9999), True
    if name == "M=inf":
        return dict(DEFAULT_HP, val_patience=9999), False
    if name == "def.":
        return dict(DEFAULT_HP), False
    if name == "P=1,p=0.1,dyn":
        return dict(DEFAULT_HP, val_patience=1, val_pct=0.1,
                    val_strategy="dynamic"), False
    if name == "P=3,p=0.1,dyn":
        return dict(DEFAULT_HP, val_patience=3, val_pct=0.1,
                    val_strategy="dynamic"), False

def fmt(x):
    s = f"{x:.2e}"
    m, e = s.split("e")
    return f"{m}x10^{int(e)}"

def marker(base, v):
    if v < base - 1e-12:
        return "▲"
    if v > base + 1e-12:
        return "▼"
    return "="

print("=" * 108)
print("TABELLA 6.22 riprodotta (e30 = ||w30-w*||2, seed 42; in parentesi i")
print("ricampionamenti; ▲/▼/= relativi alla colonna 'base')")
print("=" * 108)
hdr = f"{'Metodo':34s} {'base':>13s} {'M=inf':>13s} {'def.':>13s} {'P=1,p=0.1,dyn':>13s} {'P=3,p=0.1,dyn':>13s}"
print(hdr)
for pname in ("quad_well", "quad_ill", "quad_very_ill", "quad_offdiag"):
    for algo in ("gd", "bb", "newton_cg", "newton_l1"):
        base30 = None
        vals = []
        for c in CONFIG_COLS:
            hp, force = config_hp(c)
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            e30, res = run_config(p, algo, hp, force_base=force)
            if c == "base":
                base30 = e30
            vals.append((e30, res))
        cells = []
        for i, (e30, res) in enumerate(vals):
            mk = "" if i == 0 else marker(base30, e30)
            cells.append(f"{fmt(e30)}{mk}({res})")
        label = f"{PRESET_LATEX[pname]} - {ALGO_LATEX[algo]}"
        print(f"{label:34s} " + " ".join(f"{c:>13s}" for c in cells))

