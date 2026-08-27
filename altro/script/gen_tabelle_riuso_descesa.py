#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep degli iperparametri del criterio "Riuso per discesa della loss sul
batch" (Sezione 6.7, sottosezione sul riuso per discesa). Gli algoritmi sono
ESATTAMENTE i codici Python generati da visualizzazione.html (varianti con
il criterio di discesa) salvati in descent_codice_generato/ (estrazione con
harness Deno + DOM fittizio, estrai_codice_descesa.mjs):

  - Dynamic GD   -> gd_wolfe.py       (line search di Wolfe, come Sez. 6)
  - BB-CCV       -> bb_armijo.py      (line search di Armijo, come Appendice E)
  - Newton-CG    -> newton_cg_tied.py (H_k legato a S_k, default teoria)
  - Newton-CG L1 -> newton_l1_tied.py (H_k legato a S_k)

Il criterio di discesa (checkbox "Usa la discesa della loss sul batch per lo
stop del riuso" dell'app): dopo w_{k+1} si valuta ogni f iterazioni la loss
media J_batch(w_k) sul mini-batch CORRENTE (versione firmata: si confronta
con l'iterato precedente sullo STESSO batch); se per P valutazioni
consecutive non risulta J_batch(w_{k-1}) - J_batch(w_k) >= tau*|J_batch(w_{k-1})|
+ min_abs, il batch viene ricampionato all'iterazione successiva. Nessun
validation set: il mini-batch si campiona dall'INTERO dataset (tetto CCV = N).

Griglia di iperparametri testata (quelli che l'app espone quando la checkbox
del riuso per discesa e' attiva):

  desc_tol      (tolleranza relativa)   : 1e-5, 1e-4, 1e-3 (default 1e-4)
  desc_min_abs  (soglia assoluta)       : 0                  (default 0)
  desc_patience (pazienza)              : 1, 3, 8            (default 1)
  desc_freq     (frequenza valutazione) : 1, 3               (default 1)

Uso:
  python3 gen_tabelle_riuso_descesa.py --data [OUT_JSON]
      Esegue lo sweep completo, salva i risultati in OUT_JSON (default
      descent_sweep.json) e stampa l'analisi (effetti marginali).
  python3 gen_tabelle_riuso_descesa.py --analisi [SWEEP_JSON]
      Stampa l'analisi dai dati gia' salvati.
  python3 gen_tabelle_riuso_descesa.py --robustezza [OUT_JSON]
      Robustezza su 5 seed delle configurazioni candidate.
  python3 gen_tabelle_riuso_descesa.py --tex [SWEEP_JSON] [ROB_JSON]
      Genera le tabelle LaTeX per la sottosezione sul riuso per discesa.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.join(HERE, "descent_codice_generato")

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

SEEDS = [42, 7, 123, 2024, 999]

# ----------------------------------------------------------------------
# PRESET (identici a gen_tabelle_riuso.py / LOSS_PRESETS dell'app)
# ----------------------------------------------------------------------
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
    "quad_well": r"$\kappa\approx1.1$",
    "quad_ill": r"$\kappa\approx20$",
    "quad_very_ill": r"$\kappa\approx100$",
    "quad_offdiag": r"incrociato ($\kappa\approx1.67$)",
}

# ----------------------------------------------------------------------
# ALGORITMI (codice generato dall'app, in descent_codice_generato/)
# ----------------------------------------------------------------------
# (nome, file, funzione, argomenti extra oltre a w0/theta/max_iter/alpha/batch0)
ALGOS = {
    "gd":        ("gd_wolfe.py",       "dynamic_gd",    []),
    "bb":        ("bb_armijo.py",      "bb_dynamic_gd", []),
    "newton_cg": ("newton_cg_tied.py", "newton_cg",     [R_, MAXCG]),
    "newton_l1": ("newton_l1_tied.py", "newton_l1",     [NU, SIGMA, MAXCG, ETA]),
}
ALGO_LATEX = {
    "gd": "Dynamic GD",
    "bb": "BB-CCV",
    "newton_cg": "Newton-CG",
    "newton_l1": "Newton-CG $L_1$",
}

# ----------------------------------------------------------------------
# SWEEP
# ----------------------------------------------------------------------
DESC_TOLS = [1e-5, 1e-4, 1e-3]
DESC_MIN_ABS = [0.0]
DESC_PATIENCE = [1, 3, 8]
DESC_FREQ = [1, 3]

DEFAULT_HP = dict(desc_tol=1e-4, desc_min_abs=0.0, desc_patience=1,
                  desc_freq=1)


def make_functions(p, algo):
    fname, entry, _ = ALGOS[algo]
    src = open(os.path.join(CODEGEN, fname), encoding="utf-8").read()
    src = src.split("\ndesc_tol = ")[0]      # toglie la chiamata a livello di modulo
    ns = {
        "np": np,
        "N": p["N"],
        "loss_i": p["loss_i"],
        "grad_i": p["grad_i"],
        "hess_i": p["hess_i"],
        "hessvec_i": p["hessvec_i"],
        "grad_full": p["grad_full"],
        # Preambolo dell'app (visualizzazione.html, runAlgorithm): i parametri
        # sono globali a livello di modulo; il codice generato li legge da lì.
        "w0": W0, "alpha": ALPHA, "max_iter": MAX_ITER, "theta": THETA,
        "batch0": BATCH0, "R": R_, "maxcg": MAXCG, "nu": NU, "sigma": SIGMA,
        "eta": ETA,
    }
    exec(src, ns)
    return ns[entry]


def make_functions_base(p, algo):
    """Come make_functions ma con il campionamento FORZATO a ogni iterazione
    (riferimento 'base': ricampionamento a ogni passo dall'intero dataset,
    come nelle Tabelle del riuso). Modifica solo il guard del campionamento."""
    fname, entry, _ = ALGOS[algo]
    src = open(os.path.join(CODEGEN, fname), encoding="utf-8").read()
    src = src.split("\ndesc_tol = ")[0]
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
    return ns[entry]


def run_once(p, algo, hp):
    """Esegue una run con iperparametri hp; ritorna dict di metriche."""
    fname, entry, extra = ALGOS[algo]
    f = make_functions(p, algo)
    args = [W0, THETA, MAX_ITER, ALPHA, BATCH0] + extra + [
        hp["desc_tol"], hp["desc_min_abs"], hp["desc_patience"],
        hp["desc_freq"]]
    hist, batch_sizes, resize_points, resample_pts, m_actual, desc_hist = f(*args)
    e = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
    resamples = sum(1 for x in m_actual if x == 1) - 1   # ricampionamenti dopo il primo
    return {
        "e30": e[-1],
        "e_min": min(e),
        "iterations": len(hist) - 1,
        "final_batch": batch_sizes[-1],
        "resamples": max(resamples, 0),
        "ccv_resizes": len(resize_points),
        "desc_resamples": len(resample_pts),
        "desc_last": desc_hist[-1] if desc_hist else None,
        "desc_min": min(desc_hist) if desc_hist else None,
    }


def run_base(p, algo):
    """Riferimento: ricampionamento a ogni iterazione dall'intero dataset
    (stesso flusso di RNG del criterio di discesa, campionamento forzato)."""
    fname, entry, extra = ALGOS[algo]
    f = make_functions_base(p, algo)
    hp = dict(DEFAULT_HP)
    hp["desc_patience"] = 9999          # irrilevante: il campionamento è forzato
    args = [W0, THETA, MAX_ITER, ALPHA, BATCH0] + extra + [
        hp["desc_tol"], hp["desc_min_abs"], hp["desc_patience"],
        hp["desc_freq"]]
    hist, batch_sizes, resize_points, resample_pts, m_actual, desc_hist = f(*args)
    e = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
    resamples = sum(1 for x in m_actual if x == 1) - 1
    return {
        "e30": e[-1], "e_min": min(e), "iterations": len(hist) - 1,
        "final_batch": batch_sizes[-1], "resamples": max(resamples, 0),
        "ccv_resizes": len(resize_points),
    }


def run_minf(p, algo):
    """Riferimento: riuso illimitato sull'intero dataset — la pazienza non
    viene mai raggiunta (patience=9999 > 30 iterazioni), si ricampiona solo
    quando la CCV aumenta il batch (come M=∞)."""
    hp = dict(DEFAULT_HP)
    hp["desc_patience"] = 9999
    return run_once(p, algo, hp)


# ----------------------------------------------------------------------
# ESPERIMENTO
# ----------------------------------------------------------------------
def build_sweep_grid():
    grid = []
    for tol in DESC_TOLS:
        for min_abs in DESC_MIN_ABS:
            for pat in DESC_PATIENCE:
                for freq in DESC_FREQ:
                    hp = dict(DEFAULT_HP)
                    hp.update(desc_tol=tol, desc_min_abs=min_abs,
                              desc_patience=pat, desc_freq=freq)
                    grid.append(hp)
    return grid


def run_sweep(presets=("quad_well", "quad_ill", "quad_very_ill", "quad_offdiag"),
              algos=("gd", "bb", "newton_cg", "newton_l1")):
    """Ritorna data[(preset, algo, hp_key)] = metriche e
    refs[(preset, algo, nome_riferimento)] = metriche."""
    grid = build_sweep_grid()
    data = {}
    refs = {}
    for pname in presets:
        for algo in algos:
            for hp in grid:
                np.random.seed(SEED)
                p = PRESET_MAKERS[pname]()
                m = run_once(p, algo, hp)
                key = hp_key(hp)
                data[(pname, algo, key)] = m
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "base")] = run_base(p, algo)
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "minf")] = run_minf(p, algo)
    return data, refs


def hp_key(hp):
    return (f"tol={hp['desc_tol']}|minabs={hp['desc_min_abs']}"
            f"|pat={hp['desc_patience']}|freq={hp['desc_freq']}")


def key_to_hp(key):
    parts = dict(kv.split("=") for kv in key.split("|"))
    return dict(desc_tol=float(parts["tol"]),
                desc_min_abs=float(parts["minabs"]),
                desc_patience=int(parts["pat"]),
                desc_freq=int(parts["freq"]))


def serialize(data, refs=None):
    out = {}
    for (pname, algo, key), m in data.items():
        out[f"{pname}|{algo}|{key}"] = m
    if refs:
        for (pname, algo, refname), m in refs.items():
            out[f"REF|{pname}|{algo}|{refname}"] = m
    return out


def load_serialized(obj):
    data = {}
    refs = {}
    for k, m in obj.items():
        parts = k.split("|")
        if parts[0] == "REF":
            refs[(parts[1], parts[2], parts[3])] = m
        else:
            pname, algo, key = k.split("|", 2)
            data[(pname, algo, key)] = m
    return data, refs


def print_analysis(data, refs=None):
    presets = sorted({k[0] for k in data})
    algos = sorted({k[1] for k in data})
    grid_keys = [hp_key(hp) for hp in build_sweep_grid()]

    print("=" * 78)
    print("EFFETTI MARGINALI MEDI su e30 (media su 4 problemi x 4 algoritmi)")
    print("=" * 78)
    dims = [
        ("desc_patience", "pat", "Pazienza"),
        ("desc_tol", "tol", "Tolleranza"),
        ("desc_freq", "freq", "Frequenza"),
    ]
    for key_name, short, label in dims:
        print(f"\n-- {label} --")
        values = sorted({key_to_hp(hk)[key_name] for hk in grid_keys})
        for v in values:
            es, rs = [], []
            for (pname, algo, hk), m in data.items():
                if key_to_hp(hk)[key_name] == v:
                    es.append(m["e30"])
                    rs.append(m["resamples"])
            print(f"  {short}={v!s:8}  media e30={np.mean(es):.4e}  "
                  f"media ricampionamenti={np.mean(rs):6.2f}  "
                  f"min e30={np.min(es):.4e}")

    print("\n" + "=" * 78)
    print("EFFETTI MARGINALI PER PROBLEMA (media su 4 algoritmi)")
    print("=" * 78)
    for pname in presets:
        print(f"\n-- {pname} --")
        for key_name, short, label in dims:
            values = sorted({key_to_hp(hk)[key_name] for hk in grid_keys})
            line = f"  {label:16s} "
            for v in values:
                es = [m["e30"] for (pp, aa, hk), m in data.items()
                      if pp == pname and key_to_hp(hk)[key_name] == v]
                line += f"{short}={v!s:8}->{np.mean(es):.3e}   "
            print(line)

    print("\n" + "=" * 78)
    print("RIFERIMENTI (dataset intero, seed 42) vs DEFAULT e BEST dello sweep")
    print("=" * 78)
    for pname in presets:
        for algo in algos:
            bt = refs[(pname, algo, "base")]
            mi = refs[(pname, algo, "minf")]
            default = data[(pname, algo, hp_key(DEFAULT_HP))]
            best = min(
                ((hk, data[(pname, algo, hk)]) for hk in grid_keys),
                key=lambda t: t[1]["e30"])
            print(f"  {pname:14s} {algo:10s} "
                  f"base e30={bt['e30']:.4e} res={bt['resamples']:2d} | "
                  f"minf e30={mi['e30']:.4e} res={mi['resamples']:2d} | "
                  f"default e30={default['e30']:.4e} res={default['resamples']:2d} | "
                  f"best e30={best[1]['e30']:.4e} res={best[1]['resamples']:2d} "
                  f"({best[0]})")
    return 0


# ----------------------------------------------------------------------
# ROBUSTEZZA (5 seed) sulle configurazioni candidate
# ----------------------------------------------------------------------
# Candidate: default dell'app (P=1, tau=1e-4, f=1), le pazienze 3 e 8, gli
# estremi della tolleranza e la frequenza 3. La migliore dello sweep (media
# e30 su 16 caselle, seed 42) viene aggiunta dinamicamente da best_configs().
CANDIDATES = {
    "default":         dict(desc_tol=1e-4, desc_min_abs=0.0, desc_patience=1,
                            desc_freq=1),
    "pat3":            dict(desc_tol=1e-4, desc_min_abs=0.0, desc_patience=3,
                            desc_freq=1),
    "pat8":            dict(desc_tol=1e-4, desc_min_abs=0.0, desc_patience=8,
                            desc_freq=1),
    "tol1e-5-pat1":    dict(desc_tol=1e-5, desc_min_abs=0.0, desc_patience=1,
                            desc_freq=1),
    "tol1e-3-pat1":    dict(desc_tol=1e-3, desc_min_abs=0.0, desc_patience=1,
                            desc_freq=1),
    "freq3-pat1":      dict(desc_tol=1e-4, desc_min_abs=0.0, desc_patience=1,
                            desc_freq=3),
}


def best_configs(data, top=2):
    """Le top-2 configurazioni dello sweep per media di e30 su 16 caselle."""
    grid_keys = [hp_key(hp) for hp in build_sweep_grid()]
    ranked = sorted(grid_keys, key=lambda hk: np.mean(
        [data[(p, a, hk)]["e30"] for p in PRESET_MAKERS for a in ALGOS]))
    return ranked[:top]


def run_robustness(presets=("quad_well", "quad_ill", "quad_very_ill",
                            "quad_offdiag"),
                   algos=("gd", "bb", "newton_cg", "newton_l1"),
                   seeds=SEEDS, extra=None):
    """Per ogni (preset, algo, config) con config in CANDIDATES (+extra,
    es. le migliori dello sweep) + riferimenti, ritorna la lista degli e30
    sui 5 seed."""
    configs = dict(CANDIDATES)
    if extra:
        configs.update(extra)
    res = {}
    for pname in presets:
        for algo in algos:
            for name, hp in configs.items():
                es = []
                for s in seeds:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    m = run_once(p, algo, dict(DEFAULT_HP, **hp))
                    es.append(m["e30"])
                res[(pname, algo, name)] = es
            for ref in ("base", "minf"):
                es = []
                for s in seeds:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    m = (run_base(p, algo) if ref == "base"
                         else run_minf(p, algo))
                    es.append(m["e30"])
                res[(pname, algo, ref)] = es
    return res


def print_robustness(res):
    print("=" * 78)
    print("ROBUSTEZZA 5 SEED: e30 medio per configurazione")
    print("=" * 78)
    names = list(CANDIDATES) + ["base", "minf"]
    for pname in sorted({k[0] for k in res}):
        print(f"\n-- {pname} --")
        for algo in sorted({k[1] for k in res}):
            line = f"  {algo:10s}"
            for name in names:
                es = res[(pname, algo, name)]
                line += f"  {name:15s} {np.mean(es):.3e}"
            print(line)
    print("\n" + "=" * 78)
    print("CONFRONTO GLOBALE: e30 medio (16 caselle x 5 seed) per configurazione")
    print("=" * 78)
    for name in names:
        all_es = [np.mean(res[k]) for k in res if k[2] == name]
        print(f"  {name:15s} media={np.mean(all_es):.4e}  "
              f"min={np.min(all_es):.4e}  max={np.max(all_es):.4e}")
    print("\nVITTORIE vs base (16 caselle, e30 medio su 5 seed):")
    for name in names:
        if name == "base":
            continue
        wins = 0
        for pname in sorted({k[0] for k in res}):
            for algo in sorted({k[1] for k in res}):
                bt = np.mean(res[(pname, algo, "base")])
                me = np.mean(res[(pname, algo, name)])
                if me < bt - 1e-12:
                    wins += 1
        print(f"  {name:15s} {wins:2d}/16 caselle migliori di base")
    return 0


# ----------------------------------------------------------------------
# GENERAZIONE TABELLE LaTeX (confronto, sensibilità, robustezza)
# ----------------------------------------------------------------------
def fmt3(x):
    s = f"{x:.2e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def marker(base, v):
    if v < base - 1e-12:
        return "$\\blacktriangle$"
    if v > base + 1e-12:
        return "$\\blacktriangledown$"
    return "$=$"


def hp_key_label(key):
    """hp_key -> etichetta LaTeX compatta della configurazione."""
    import math
    hp = key_to_hp(key)
    e = int(round(math.log10(hp["desc_tol"])))
    return (f"$P{{=}}{hp['desc_patience']},\\tau{{=}}10^{{{e}}},"
            f"f{{=}}{hp['desc_freq']}$")


def fmt_hp_value(key_name, v):
    import math
    if key_name == "desc_tol":
        e = int(round(math.log10(v)))
        return f"$10^{{{e}}}$"
    return f"{v}"


def gen_tables_tex(data, refs, rob, best_keys):
    presets = ("quad_well", "quad_ill", "quad_very_ill", "quad_offdiag")
    algos = ("gd", "bb", "newton_cg", "newton_l1")
    out = []
    out.append("% ======================================================================")
    out.append("% Tabelle riuso per discesa della loss sul batch (confronto, sensibilità,")
    out.append("% robustezza) - Sezione 6.7")
    out.append("% Generate da gen_tabelle_riuso_descesa.py --tex - NON modificare a mano.")
    out.append("% ======================================================================")
    out.append("")

    CONFIG_COLS = ["base", "minf", "default"] + list(best_keys)
    CONFIG_HEADERS = {
        "base": "\\emph{base}",
        "minf": "$M{=}\\infty$",
        "default": "def.",
    }
    for k in best_keys:
        CONFIG_HEADERS[k] = hp_key_label(k)

    # ---- confronto (seed 42) ----
    out.append("% Tabella confronto - riuso per discesa vs riferimenti (seed 42)")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\tiny")
    out.append("\\renewcommand{\\arraystretch}{1.0}")
    out.append("\\setlength{\\tabcolsep}{1.5pt}")
    best_lbl = " e ".join(CONFIG_HEADERS[k] for k in best_keys)
    out.append(
        "\\caption{Riuso per discesa della loss sul batch: errore finale "
        f"$e_{{30}}=\\|w_{{30}}-w_*\\|_2$ (seed 42) per i quattro problemi e i "
        "quattro algoritmi. Colonne: \\emph{base} = ricampionamento a ogni "
        "iterazione; $M{=}\\infty$ = riuso illimitato; \\emph{def.} = valori di "
        "default del criterio di discesa ($\\tau=10^{-4}$, $P=1$, $f=1$, soglia "
        f"assoluta 0); {best_lbl} = migliori configurazioni dello sweep (media di "
        "$e_{30}$ su 16 combinazioni problema$\\times$algoritmo). Tra parentesi il "
        "numero di ricampionamenti del mini-batch nelle 30 iterazioni. Tutte le "
        "colonne campionano dall'intero dataset (nessun validation set). Simboli "
        "relativi a \\emph{base}: $\\blacktriangle$ = migliora, "
        "$\\blacktriangledown$ = peggiora, $=$ = invariato.}")
    out.append("\\label{tab:riuso_desc_confronto}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    headers = ["Metodo"] + [CONFIG_HEADERS[c] for c in CONFIG_COLS]
    out.append(" & ".join(headers) + "\\\\")
    out.append("\\midrule")
    for pname in presets:
        for algo in algos:
            base30 = refs[(pname, algo, "base")]["e30"]
            cells = [f"{PRESET_LATEX[pname]} - {ALGO_LATEX[algo]}"]
            for c in CONFIG_COLS:
                if (pname, algo, c) in refs:
                    m = refs[(pname, algo, c)]
                else:
                    key = c if c != "default" else hp_key(DEFAULT_HP)
                    m = data[(pname, algo, key)]
                e = m["e30"]
                res = m["resamples"]
                mk = "" if c == "base" else marker(base30, e)
                cells.append(f"{fmt3(e)}{mk} ({res})")
            out.append(" & ".join(cells) + "\\\\")
        out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    # ---- sensibilità marginale (seed 42, sweep) ----
    out.append("% Tabella sensibilità - iperparametri del riuso per discesa")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append("\\setlength{\\tabcolsep}{5pt}")
    out.append(
        "\\caption{Sensibilità dell'errore finale $e_{30}$ agli iperparametri del "
        "riuso per discesa: media di $e_{30}$ e del numero di ricampionamenti sulle "
        "16 combinazioni problema$\\times$algoritmo (seed 42), fissando uno "
        "iperparametro al valore indicato e mediando sugli altri. $P$ = pazienza, "
        "$\\tau$ = tolleranza relativa (criterio firmato "
        "$J_{\\mathrm{batch}}(w_{k-1})-J_{\\mathrm{batch}}(w_k) \\ge "
        "\\tau\\lvert J_{\\mathrm{batch}}(w_{k-1})\\rvert + "
        "\\epsilon_{\\mathrm{abs}}$, con $\\epsilon_{\\mathrm{abs}}=0$), "
        "$f$ = frequenza di valutazione.}")
    out.append("\\label{tab:riuso_desc_iper}")
    out.append("\\begin{tabular}{@{}llrr@{}}")
    out.append("\\toprule")
    out.append("Iperparametro & Valore & $\\overline{e}_{30}$ & ricampionamenti medi\\\\")
    out.append("\\midrule")
    dims = [
        ("Pazienza $P$", "desc_patience", DESC_PATIENCE),
        ("Tolleranza $\\tau$", "desc_tol", DESC_TOLS),
        ("Frequenza $f$", "desc_freq", DESC_FREQ),
    ]
    for label, key_name, values in dims:
        for v in values:
            es, rs = [], []
            for (pname, algo, hk), m in data.items():
                if key_to_hp(hk)[key_name] == v:
                    es.append(m["e30"])
                    rs.append(m["resamples"])
            vs = fmt_hp_value(key_name, v)
            row = (f"{label} & {vs} & {fmt3(float(np.mean(es)))} "
                   f"& {np.mean(rs):.1f}\\\\")
            out.append(row)
        out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    # ---- robustezza 5 seed ----
    rob_cols = ["base", "minf", "default"] + list(best_keys)
    out.append("% Tabella robustezza su 5 seed")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.1}")
    out.append("\\setlength{\\tabcolsep}{3.5pt}")
    out.append(
        "\\caption{Robustezza su 5 seed indipendenti ($\\{42,7,123,2024,999\\}$) "
        "del riuso per discesa della loss sul batch. Per ogni configurazione: "
        "$\\overline{e}_{30}$ = media di $e_{30}$ su 16 caselle $\\times$ 5 seed; "
        "\\emph{vitt.} = numero di caselle (su 16) in cui la media su 5 seed è "
        "minore di quella di \\emph{base}; le ultime quattro colonne riportano la "
        "media su 5 seed per problema (mediata sui 4 algoritmi). Configurazioni "
        "come nella Tabella~\\ref{tab:riuso_desc_confronto}.}")
    out.append("\\label{tab:riuso_desc_robustezza}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Configurazione & $\\overline{e}_{30}$ & \\emph{vitt.} & "
               "$\\kappa{\\approx}1.1$ & $\\kappa{\\approx}20$ & "
               "$\\kappa{\\approx}100$ & incr. ($\\kappa{\\approx}1.67$)\\\\")
    out.append("\\midrule")
    for c in rob_cols:
        es = [np.mean(rob[(p, a, c)]) for p in presets for a in algos]
        wins = 0
        if c != "base":
            for p in presets:
                for a in algos:
                    if np.mean(rob[(p, a, c)]) < np.mean(rob[(p, a, "base")]) - 1e-12:
                        wins += 1
        per_problem = [np.mean([np.mean(rob[(p, a, c)]) for a in algos])
                       for p in presets]
        row = ([f"{CONFIG_HEADERS[c]}"] + [fmt3(float(np.mean(es)))] +
               [f"{wins}/16"] + [fmt3(float(v)) for v in per_problem])
        out.append(" & ".join(row) + "\\\\")
        if c == "minf":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")
    return out


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mode = args[0]
    if mode == "--data":
        out_path = args[1] if len(args) > 1 else "descent_sweep.json"
        print("sweep in corso (18 configurazioni x 4 problemi x 4 algoritmi + riferimenti)...")
        data, refs = run_sweep()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialize(data, refs), f, indent=1)
        print(f"salvato: {out_path}")
        print_analysis(data, refs)
        return 0
    if mode == "--analisi":
        path = args[1] if len(args) > 1 else "descent_sweep.json"
        with open(path, encoding="utf-8") as f:
            data, refs = load_serialized(json.load(f))
        print_analysis(data, refs)
        return 0
    if mode == "--robustezza":
        out_path = args[1] if len(args) > 1 else "descent_robustezza.json"
        sweep_path = args[2] if len(args) > 2 else "descent_sweep.json"
        with open(sweep_path, encoding="utf-8") as f:
            data, _ = load_serialized(json.load(f))
        extra = {k: key_to_hp(k) for k in best_configs(data, top=2)}
        print(f"robustezza su 5 seed in corso (candidate + best {list(extra)} )...")
        res = run_robustness(extra=extra)
        ser = {f"{p}|{a}|{name}": es for (p, a, name), es in res.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ser, f, indent=1)
        print(f"salvato: {out_path}")
        print_robustness(res)
        return 0
    if mode == "--tex":
        sweep_path = args[1] if len(args) > 1 else "descent_sweep.json"
        rob_path = args[2] if len(args) > 2 else "descent_robustezza.json"
        with open(sweep_path, encoding="utf-8") as f:
            data, refs = load_serialized(json.load(f))
        with open(rob_path, encoding="utf-8") as f:
            rob_obj = json.load(f)
        rob = {}
        for k, es in rob_obj.items():
            p, a, name = k.split("|", 2)
            rob[(p, a, name)] = es
        best = best_configs(data, top=2)
        sys.stdout.write("\n".join(gen_tables_tex(data, refs, rob, best)) + "\n")
        return 0
    print(f"modalità sconosciuta: {mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())










