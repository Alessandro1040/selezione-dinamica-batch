#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selezione delle configurazioni consigliate per ciascun metodo (Sez. 6.5.8
"Iperparametri consigliati per ciascun metodo"), su TUTTI i problemi della
batteria: i quattro quadratici + la funzione di Rosenbrock (non quadratica).

Criterio (come descritto in tesi/tesi.tex):
  1. candidati per ciascun metodo: riuso del mini-batch per M in
     {inf, 10, 5, 3, 2, 1} e stop adattivo con validation set con P in {1, 3},
     f = 1, p in {10%, 20%}, strategia di split fissa o dinamica
     (tolleranza tau = 1e-4 di default);
  2. sono ammissibili le configurazioni che MIGLIORANO la mediana di e30
     (su 5 seed) rispetto alla base su TUTTI i problemi della batteria;
  3. tra le ammissibili vince quella con la migliore (piu' piccola) media
     geometrica dei rapporti e30_cfg / e30_base sulle
     len(PRESETS) * len(SEEDS) combinazioni problema x seed;
  4. se nessuna configurazione e' ammissibile, la consigliata e' la base.

Gli esperimenti eseguono il codice ESATTO generato da visualizzazione.html
(riuso: run_base_reuse; validation: run_validation di
riproduci_tutte_le_tabelle.py, che usa i generatori *_Validation dell'app).

Uso:
  python3 selezione_consigliati.py          # stampa la scelta e i dettagli
"""
import numpy as np

import riproduci_tutte_le_tabelle as R

M_LABEL = {"inf": r"\infty", "10": "10", "5": "5", "3": "3", "2": "2", "1": "1"}


def cfg_id(cfg):
    kind, p = cfg
    if kind == "riuso":
        return f"riuso M={M_LABEL[p]}"
    pat, freq, pct, strat = p
    return (f"val P={pat} f={freq} p={pct:.0%} strat={strat}")


def val_hp(cfg):
    pat, freq, pct, strat = cfg[1]
    return dict(R.VALID_DEFAULT_HP, val_patience=pat, val_freq=freq,
                val_pct=pct, val_strategy=strat)


def run_cfg(p, algo, cfg, seed):
    np.random.seed(seed)
    pp = R.PRESET_MAKERS[p](seed=seed)
    kind, param = cfg
    if kind == "riuso":
        e, _bs, _rp = R.run_base_reuse(pp, algo,
                                       max_consec=R.M_VALUES[param],
                                       reuse=True)
        return e[-1]
    e, _res = R.run_validation(pp, algo, val_hp(cfg))
    return e[-1]


def run_base(p, algo, seed):
    np.random.seed(seed)
    pp = R.PRESET_MAKERS[p](seed=seed)
    e, _bs, _rp = R.run_base_reuse(pp, algo, None, reuse=False)
    return e[-1]


def candidates(algo):
    out = []
    for m in ("inf", "10", "5", "3", "2", "1"):
        out.append(("riuso", m))
    for pat in (1, 3):
        for pct in (0.1, 0.2):
            for strat in ("fixed", "dynamic"):
                out.append(("val", (pat, 1, pct, strat)))
    return out


def main():
    results = {}
    for algo in R.ALGOS4:
        base = {(p, s): run_base(p, algo, s)
                for p in R.PRESETS for s in R.SEEDS}
        med_base = {p: float(np.median([base[(p, s)] for s in R.SEEDS]))
                    for p in R.PRESETS}
        cand_stats = []
        for cfg in candidates(algo):
            e30 = {(p, s): run_cfg(p, algo, cfg, s)
                   for p in R.PRESETS for s in R.SEEDS}
            med = {p: float(np.median([e30[(p, s)] for s in R.SEEDS]))
                   for p in R.PRESETS}
            # ammissibile: mediana migliore della base su OGNI problema
            admissible = all(med[p] < med_base[p] - 1e-12
                             for p in R.PRESETS)
            ratios = [e30[(p, s)] / base[(p, s)]
                      for p in R.PRESETS for s in R.SEEDS]
            # media geometrica dei rapporti
            geomean = float(np.exp(np.mean(np.log(ratios))))
            cand_stats.append((cfg_id(cfg), admissible, geomean,
                               {p: med[p] for p in R.PRESETS}))
        admissible = [c for c in cand_stats if c[1]]
        print(f"=== {algo} ===")
        print("mediana base per problema:")
        for p in R.PRESETS:
            print(f"   {p:14s} {med_base[p]:.4e}")
        if not admissible:
            print("  -> nessuna configurazione ammissibile: consigliata = BASE")
            results[algo] = ("base", "")
            continue
        admissible.sort(key=lambda c: c[2])
        winner = admissible[0]
        print("  candidati ammissibili (ordinati per media geometrica):")
        for c in admissible:
            med = ", ".join(f"{p}={v:.2e}" for p, v in c[3].items())
            print(f"    {c[0]:40s} geomean={c[2]:.4f}  [{med}]")
        kind, param = winner[0].split(" ", 1)
        if kind == "riuso":
            results[algo] = ("riuso", param)
        else:
            # 'val P=1 f=1 p=10% strat=fixed' -> ("validation", "P=1;f=1;p=0.1;fixed")
            part = dict(x.split("=", 1) for x in param.split(" "))
            pct = float(part["p"].rstrip("%")) / 100.0
            results[algo] = ("validation",
                             f"P={part['P']};f={part['f']};"
                             f"p={pct};strat={part['strat']}")
        print(f"  -> VINCE: {winner[0]}  (geomean={winner[2]:.4f})")
        print()
    print("=== RISULTATO FINALE (da copiare in RECOMMENDED) ===")
    for algo, (kind, param) in results.items():
        print(f'    "{algo}": ("{kind}", "{param}"),')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
