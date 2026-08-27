#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep degli iperparametri dello stop adattivo con validation set (Appendice E,
sottosezione E.6). Gli algoritmi sono ESATTAMENTE i codici Python generati da
visualizzazione.html (varianti *Validation) salvati in
validation_codice_generato/ (estrazione con harness Deno + DOM fittizio):

  - Dynamic GD   -> gd_wolfe.py       (line search di Wolfe, come Sez. 6)
  - BB-CCV       -> bb_armijo.py      (line search di Armijo, come Appendice E)
  - Newton-CG    -> newton_cg_tied.py (H_k legato a S_k, default teoria)
  - Newton-CG L1 -> newton_l1_tied.py (H_k legato a S_k)

La griglia di iperparametri testata (quelli che l'app espone quando la checkbox
"Usa validation set per stop adattivo" è attiva):

  val_pct      (percentuale validation)  : 0.1, 0.2, 0.3   (default 0.2)
  val_tol      (tolleranza relativa)     : 1e-5, 1e-4, 1e-3 (default 1e-4)
  val_patience (pazienza)                : 1, 3, 8          (default 3)
  val_freq     (frequenza valutazione)   : 1, 3             (default 1)
  val_strategy (fixed/dynamic)           : fixed, dynamic   (default fixed)

Uso:
  python3 gen_tabelle_riuso_validation.py --data [OUT_JSON]
      Esegue lo sweep completo, salva i risultati in OUT_JSON (default
      validation_sweep.json) e stampa l'analisi (effetti marginali).
  python3 gen_tabelle_riuso_validation.py --tex [APPENDICE_TEX] [OUT]
      (implementato dopo l'analisi dei dati)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.join(HERE, "validation_codice_generato")

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
# ALGORITMI (codice generato dall'app, in validation_codice_generato/)
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
VAL_PCTS = [0.1, 0.2, 0.3]
VAL_TOLS = [1e-5, 1e-4, 1e-3]
VAL_PATIENCE = [1, 3, 8]
VAL_FREQ = [1, 3]
VAL_STRATEGY = ["fixed", "dynamic"]

DEFAULT_HP = dict(val_pct=0.2, val_tol=1e-4, val_patience=3, val_freq=1,
                  val_min_abs=0.0, val_strategy="fixed")


def make_functions(p, algo):
    fname, entry, _ = ALGOS[algo]
    src = open(os.path.join(CODEGEN, fname), encoding="utf-8").read()
    src = src.split("\nval_pct = ")[0]      # toglie la chiamata a livello di modulo
    ns = {
        "np": np,
        "N": p["N"],
        "loss_i": p["loss_i"],
        "grad_i": p["grad_i"],
        "hess_i": p["hess_i"],
        "hessvec_i": p["hessvec_i"],
        "grad_full": p["grad_full"],
        # Preambolo dell'app (visualizzazione.html, runAlgorithm): i parametri
        # sono globali a livello di modulo; il codice generato (es. il blocco
        # di campionamento della Hessiana usa `R`) li legge da lì.
        "w0": W0, "alpha": ALPHA, "max_iter": MAX_ITER, "theta": THETA,
        "batch0": BATCH0, "R": R_, "maxcg": MAXCG, "nu": NU, "sigma": SIGMA,
        "eta": ETA,
    }
    exec(src, ns)
    return ns[entry]


def make_functions_base_train(p, algo):
    """Come make_functions ma con il campionamento FORZATO a ogni iterazione
    (riferimento 'base sul training set': stessi dati di accesso dello stop
    adattivo — split train/validation fisso — ma senza il criterio di
    validazione). Modifica solo il guard del campionamento."""
    fname, entry, _ = ALGOS[algo]
    src = open(os.path.join(CODEGEN, fname), encoding="utf-8").read()
    src = src.split("\nval_pct = ")[0]
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
        hp["val_pct"], hp["val_tol"], hp["val_patience"],
        hp["val_freq"], hp["val_min_abs"], hp["val_strategy"]]
    hist, batch_sizes, resize_points, m_actual, val_hist = f(*args)
    e = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
    resamples = sum(1 for x in m_actual if x == 1) - 1   # ricampionamenti dopo il primo
    return {
        "e30": e[-1],
        "e_min": min(e),
        "iterations": len(hist) - 1,
        "final_batch": batch_sizes[-1],
        "resamples": max(resamples, 0),
        "ccv_resizes": len(resize_points),
        "val_last": val_hist[-1] if val_hist else None,
        "val_min": min(val_hist) if val_hist else None,
    }


def run_base_train(p, algo):
    """Riferimento: ricampionamento a ogni iterazione dal SOLO training set
    (split fisso pct=0.2, stesso flusso di RNG dello stop adattivo)."""
    fname, entry, extra = ALGOS[algo]
    f = make_functions_base_train(p, algo)
    hp = dict(DEFAULT_HP)
    hp["val_patience"] = 9999          # irrilevante: il campionamento è forzato
    args = [W0, THETA, MAX_ITER, ALPHA, BATCH0] + extra + [
        hp["val_pct"], hp["val_tol"], hp["val_patience"],
        hp["val_freq"], hp["val_min_abs"], hp["val_strategy"]]
    hist, batch_sizes, resize_points, m_actual, val_hist = f(*args)
    e = [float(np.linalg.norm(np.array(w) - p["W_STAR"])) for w in hist]
    resamples = sum(1 for x in m_actual if x == 1) - 1
    return {
        "e30": e[-1], "e_min": min(e), "iterations": len(hist) - 1,
        "final_batch": batch_sizes[-1], "resamples": max(resamples, 0),
        "ccv_resizes": len(resize_points),
    }


def run_minf_train(p, algo):
    """Riferimento: riuso illimitato sul SOLO training set — la pazienza non
    viene mai raggiunta (patience=9999 > 30 iterazioni), si ricampiona solo
    quando la CCV aumenta il batch (come M=∞, ma dal training set)."""
    hp = dict(DEFAULT_HP)
    hp["val_patience"] = 9999
    return run_once(p, algo, hp)



# ----------------------------------------------------------------------
# ESPERIMENTO
# ----------------------------------------------------------------------
def build_sweep_grid():
    grid = []
    for pct in VAL_PCTS:
        for tol in VAL_TOLS:
            for pat in VAL_PATIENCE:
                for freq in VAL_FREQ:
                    for strat in VAL_STRATEGY:
                        hp = dict(DEFAULT_HP)
                        hp.update(val_pct=pct, val_tol=tol, val_patience=pat,
                                  val_freq=freq, val_strategy=strat)
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
            refs[(pname, algo, "base_train")] = run_base_train(p, algo)
            np.random.seed(SEED)
            p = PRESET_MAKERS[pname]()
            refs[(pname, algo, "minf_train")] = run_minf_train(p, algo)
    return data, refs


def hp_key(hp):
    return (f"pct={hp['val_pct']}|tol={hp['val_tol']}|pat={hp['val_patience']}"
            f"|freq={hp['val_freq']}|strat={hp['val_strategy']}")


def key_to_hp(key):
    parts = dict(kv.split("=") for kv in key.split("|"))
    return dict(val_pct=float(parts["pct"]),
                val_tol=float(parts["tol"]),
                val_patience=int(parts["pat"]),
                val_freq=int(parts["freq"]),
                val_strategy=parts["strat"],
                val_min_abs=0.0)


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
        ("val_patience", "pat", "Pazienza"),
        ("val_tol", "tol", "Tolleranza"),
        ("val_pct", "pct", "Percentuale val."),
        ("val_freq", "freq", "Frequenza"),
        ("val_strategy", "strat", "Strategia"),
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
    print("RIFERIMENTI (training set, seed 42) vs DEFAULT e BEST dello sweep")
    print("=" * 78)
    for pname in presets:
        for algo in algos:
            bt = refs[(pname, algo, "base_train")]
            mi = refs[(pname, algo, "minf_train")]
            default = data[(pname, algo, hp_key(DEFAULT_HP))]
            best = min(
                ((hk, data[(pname, algo, hk)]) for hk in grid_keys),
                key=lambda t: t[1]["e30"])
            print(f"  {pname:14s} {algo:10s} "
                  f"base_train e30={bt['e30']:.4e} res={bt['resamples']:2d} | "
                  f"minf_train e30={mi['e30']:.4e} res={mi['resamples']:2d} | "
                  f"default e30={default['e30']:.4e} res={default['resamples']:2d} | "
                  f"best e30={best[1]['e30']:.4e} res={best[1]['resamples']:2d}")
    return 0


# ----------------------------------------------------------------------
# ROBUSTEZZA (5 seed) sulle configurazioni candidate
# ----------------------------------------------------------------------
CANDIDATES = {
    "default":           dict(val_pct=0.2, val_tol=1e-4, val_patience=3,
                              val_freq=1, val_strategy="fixed"),
    "pat1":              dict(val_pct=0.2, val_tol=1e-4, val_patience=1,
                              val_freq=1, val_strategy="fixed"),
    "pat1-pct1-dyn":     dict(val_pct=0.1, val_tol=1e-4, val_patience=1,
                              val_freq=1, val_strategy="dynamic"),
    "pat3-pct1-dyn":     dict(val_pct=0.1, val_tol=1e-4, val_patience=3,
                              val_freq=1, val_strategy="dynamic"),
}


def run_robustness(presets=("quad_well", "quad_ill", "quad_very_ill",
                            "quad_offdiag"),
                   algos=("gd", "bb", "newton_cg", "newton_l1"),
                   seeds=SEEDS):
    """Per ogni (preset, algo, config) con config in CANDIDATES + riferimenti,
    ritorna la lista degli e30 sui 5 seed."""
    res = {}
    for pname in presets:
        for algo in algos:
            for name, hp in CANDIDATES.items():
                es = []
                for s in seeds:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    m = run_once(p, algo, dict(DEFAULT_HP, **hp))
                    es.append(m["e30"])
                res[(pname, algo, name)] = es
            for ref in ("base_train", "minf_train"):
                es = []
                for s in seeds:
                    np.random.seed(s)
                    p = PRESET_MAKERS[pname](seed=s)
                    m = (run_base_train(p, algo) if ref == "base_train"
                         else run_minf_train(p, algo))
                    es.append(m["e30"])
                res[(pname, algo, ref)] = es
    return res


def print_robustness(res):
    print("=" * 78)
    print("ROBUSTEZZA 5 SEED: e30 medio per configurazione")
    print("=" * 78)
    names = list(CANDIDATES) + ["base_train", "minf_train"]
    for pname in sorted({k[0] for k in res}):
        print(f"\n-- {pname} --")
        for algo in sorted({k[1] for k in res}):
            line = f"  {algo:10s}"
            for name in names:
                es = res[(pname, algo, name)]
                line += f"  {name:15s} {np.mean(es):.3e}"
            print(line)
    # Sintesi globale: media su 16 caselle del e30 medio (5 seed) di default vs pat1-pct1-dyn
    print("\n" + "=" * 78)
    print("CONFRONTO GLOBALE: e30 medio (16 caselle x 5 seed) per configurazione")
    print("=" * 78)
    for name in names:
        all_es = [np.mean(res[k]) for k in res if k[2] == name]
        print(f"  {name:15s} media={np.mean(all_es):.4e}  "
              f"min={np.min(all_es):.4e}  max={np.max(all_es):.4e}")
    # Win/loss vs base_train
    print("\nVITTORIE vs base_train (16 caselle, e30 medio su 5 seed):")
    for name in names:
        if name == "base_train":
            continue
        wins = 0
        for pname in sorted({k[0] for k in res}):
            for algo in sorted({k[1] for k in res}):
                bt = np.mean(res[(pname, algo, "base_train")])
                me = np.mean(res[(pname, algo, name)])
                if me < bt - 1e-12:
                    wins += 1
        print(f"  {name:15s} {wins:2d}/16 caselle migliori di base_train")
    return 0


# ----------------------------------------------------------------------
# GENERAZIONE TABELLE LaTeX (E.19, E.20, E.21)
# ----------------------------------------------------------------------
CONFIG_COLS = ["base_train", "minf_train", "default",
               "pat1-pct1-dyn", "pat3-pct1-dyn"]
CONFIG_HEADERS = {
    "base_train": "\\emph{base}",
    "minf_train": "$M{=}\\infty$",
    "default": "def.",
    "pat1-pct1-dyn": "$P{=}1,p{=}0.1$,dyn",
    "pat3-pct1-dyn": "$P{=}3,p{=}0.1$,dyn",
}
CONFIG_HP = {
    "default": dict(DEFAULT_HP),
    "pat1": dict(DEFAULT_HP, val_patience=1),
    "pat1-pct1-dyn": dict(DEFAULT_HP, val_patience=1, val_pct=0.1,
                          val_strategy="dynamic"),
    "pat3-pct1-dyn": dict(DEFAULT_HP, val_patience=3, val_pct=0.1,
                          val_strategy="dynamic"),
}


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


def fmt_hp_value(key_name, v):
    if key_name == "val_tol":
        import math
        e = int(round(math.log10(v)))
        return f"$10^{{{e}}}$"
    if key_name == "val_pct":
        return f"{int(round(v * 100))}\\%"
    if key_name == "val_strategy":
        return "\\emph{dinamico}" if v == "dynamic" else "\\emph{fisso}"
    return f"{v}"


def gen_tables_tex(data, refs, rob):
    presets = ("quad_well", "quad_ill", "quad_very_ill", "quad_offdiag")
    algos = ("gd", "bb", "newton_cg", "newton_l1")
    out = []
    out.append("% =====================================================================")
    out.append("% Tabelle E.19-E.21 - stop adattivo con validation set")
    out.append("% Generate da gen_tabelle_riuso_validation.py --tex - NON modificare a mano.")
    out.append("% =====================================================================")
    out.append("")

    # ---- E.19: confronto (seed 42) ----
    out.append("% Tabella E.19 - confronto stop adattivo vs riferimenti (seed 42)")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\tiny")
    out.append("\\renewcommand{\\arraystretch}{1.0}")
    out.append("\\setlength{\\tabcolsep}{1.5pt}")
    out.append("\\caption{Stop adattivo con validation set: errore finale $e_{30}=\\|w_{30}-w_*\\|_2$ (seed 42) per i quattro problemi e i quattro algoritmi. Colonne: \\emph{base} = ricampionamento a ogni iterazione; $M{=}\\infty$ = riuso illimitato; \\emph{def.} = valori di default dello stop adattivo ($p=0.2$, $\\tau=10^{-4}$, $P=3$, $f=1$, split fisso); $P{=}1,p{=}0.1$,dyn = pazienza 1, validation $10\\%$ e split dinamico; $P{=}3,p{=}0.1$,dyn = pazienza 3, validation $10\\%$ e split dinamico. Tra parentesi il numero di ricampionamenti del mini-batch nelle 30 iterazioni. Tutte le colonne campionano dal solo training set ($p$ percentuale in validation): \\emph{base} e $M{=}\\infty$ si riferiscono al training set (senza criterio di validazione). Simboli relativi a \\emph{base}: $\\blacktriangle$ = migliora, $\\blacktriangledown$ = peggiora, $=$ = invariato.}")
    out.append("\\label{tab:riuso_valid_confronto}")
    out.append("\\begin{tabular}{@{}lrrrrr@{}}")
    out.append("\\toprule")
    headers = ["Metodo"] + [CONFIG_HEADERS[c] for c in CONFIG_COLS]
    out.append(" & ".join(headers) + "\\\\")
    out.append("\\midrule")
    for pname in presets:
        for algo in algos:
            base30 = refs[(pname, algo, "base_train")]["e30"]
            cells = [f"{PRESET_LATEX[pname]} - {ALGO_LATEX[algo]}"]
            for c in CONFIG_COLS:
                if (pname, algo, c) in refs:
                    m = refs[(pname, algo, c)]
                else:
                    k = hp_key(dict(DEFAULT_HP, **CONFIG_HP[c]))
                    m = data[(pname, algo, k)]
                e = m["e30"]
                res = m["resamples"]
                mk = "" if c == "base_train" else marker(base30, e)
                cells.append(f"{fmt3(e)}{mk} ({res})")
            out.append(" & ".join(cells) + "\\\\")
        out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    # ---- E.20: sensibilità marginale (seed 42, sweep) ----
    out.append("% Tabella E.20 - sensibilità agli iperparametri (media su 16 caselle, seed 42)")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append("\\setlength{\\tabcolsep}{5pt}")
    out.append("\\caption{Sensibilità dell'errore finale $e_{30}$ agli iperparametri dello stop adattivo: media di $e_{30}$ e del numero di ricampionamenti sulle 16 combinazioni problema$\\times$algoritmo (seed 42), fissando uno iperparametro al valore indicato e mediando sugli altri. $P$ = pazienza, $\\tau$ = tolleranza relativa, $p$ = percentuale di validation, $f$ = frequenza di valutazione.}")
    out.append("\\label{tab:riuso_valid_iper}")
    out.append("\\begin{tabular}{@{}llrr@{}}")
    out.append("\\toprule")
    out.append("Iperparametro & Valore & $\\overline{e}_{30}$ & ricampionamenti medi\\\\")
    out.append("\\midrule")
    dims = [
        ("Pazienza $P$", "val_patience", VAL_PATIENCE),
        ("Tolleranza $\\tau$", "val_tol", VAL_TOLS),
        ("Percentuale $p$", "val_pct", VAL_PCTS),
        ("Frequenza $f$", "val_freq", VAL_FREQ),
        ("Strategia di split", "val_strategy", VAL_STRATEGY),
    ]
    for label, key_name, values in dims:
        for v in values:
            es, rs = [], []
            for (pname, algo, hk), m in data.items():
                if key_to_hp(hk)[key_name] == v:
                    es.append(m["e30"])
                    rs.append(m["resamples"])
            vs = fmt_hp_value(key_name, v)
            row = f"{label} & {vs} & {fmt3(float(np.mean(es)))} & {np.mean(rs):.1f}\\\\"
            out.append(row)
        out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")


    # ---- E.21: robustezza 5 seed ----
    out.append("% Tabella E.21 - robustezza su 5 seed")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\footnotesize")
    out.append("\\renewcommand{\\arraystretch}{1.1}")
    out.append("\\setlength{\\tabcolsep}{3.5pt}")
    out.append("\\caption{Robustezza su 5 seed indipendenti ($\\{42,7,123,2024,999\\}$). Per ogni configurazione: $\\overline{e}_{30}$ = media di $e_{30}$ su 16 caselle $\\times$ 5 seed; \\emph{vitt.} = numero di caselle (su 16) in cui la media su 5 seed è minore di quella di \\emph{base}; le ultime quattro colonne riportano la media su 5 seed per problema (mediata sui 4 algoritmi). Configurazioni come nella Tabella~\\ref{tab:riuso_valid_confronto}.}")
    out.append("\\label{tab:riuso_valid_robustezza}")
    out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    out.append("\\toprule")
    out.append("Configurazione & $\\overline{e}_{30}$ & \\emph{vitt.} & $\\kappa{\\approx}1.1$ & $\\kappa{\\approx}20$ & $\\kappa{\\approx}100$ & incr. ($\\kappa{\\approx}1.67$)\\\\")
    out.append("\\midrule")
    for c in CONFIG_COLS:
        es = [np.mean(rob[(p, a, c)]) for p in presets for a in algos]
        wins = 0
        if c != "base_train":
            for p in presets:
                for a in algos:
                    if np.mean(rob[(p, a, c)]) < np.mean(rob[(p, a, "base_train")]) - 1e-12:
                        wins += 1
        per_problem = [np.mean([np.mean(rob[(p, a, c)]) for a in algos])
                       for p in presets]
        row = ([f"{CONFIG_HEADERS[c]}"] + [fmt3(float(np.mean(es)))] +
               [f"{wins}/16"] + [fmt3(float(v)) for v in per_problem])
        out.append(" & ".join(row) + "\\\\")
        if c == "minf_train":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mode = args[0]
    if mode == "--data":
        out_path = args[1] if len(args) > 1 else "validation_sweep.json"
        print("sweep in corso (108 configurazioni x 4 problemi x 4 algoritmi + riferimenti)...")
        data, refs = run_sweep()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialize(data, refs), f, indent=1)
        print(f"salvato: {out_path}")
        print_analysis(data, refs)
        return 0
    if mode == "--analisi":
        path = args[1] if len(args) > 1 else "validation_sweep.json"
        with open(path, encoding="utf-8") as f:
            data, refs = load_serialized(json.load(f))
        print_analysis(data, refs)
        return 0
    if mode == "--robustezza":
        out_path = args[1] if len(args) > 1 else "validation_robustezza.json"
        print("robustezza su 5 seed in corso...")
        res = run_robustness()
        ser = {f"{p}|{a}|{name}": es for (p, a, name), es in res.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ser, f, indent=1)
        print(f"salvato: {out_path}")
        print_robustness(res)
        return 0
    if mode == "--tex":
        # Genera le tabelle E.19-E.21 (confronto, iperparametri, robustezza)
        sweep_path = args[1] if len(args) > 1 else "validation_sweep.json"
        rob_path = args[2] if len(args) > 2 else "validation_robustezza.json"
        with open(sweep_path, encoding="utf-8") as f:
            data, refs = load_serialized(json.load(f))
        with open(rob_path, encoding="utf-8") as f:
            rob_obj = json.load(f)
        rob = {}
        for k, es in rob_obj.items():
            p, a, name = k.split("|")
            rob[(p, a, name)] = es
        sys.stdout.write(gen_tables_tex(data, refs, rob))
        return 0
    print(f"modalità sconosciuta: {mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

