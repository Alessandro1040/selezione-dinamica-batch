#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Esperimento 'nota' su NSynth: riconoscimento della classe di altezza
(pitch class, 12 note) con i quattro metodi della tesi + riferimento sklearn.

Riutilizza il motore di run_benchmark.py (LogReg, Dynamic GD, Newton-CG,
Newton-CG L1, BB-CCV) con C=12 e feature chroma 24D.
"""
import os, sys, time, json
import numpy as np
import run_benchmark as rb

SEED, THETA, ALPHA, BATCH0, MAX_ITER = 42, 0.5, 1.0, 64, 300
R, MAXCG, LAM, NU = 0.1, 50, 1e-4, 1e-3
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DATA = "/Users/alessandrolocurcio/Downloads/tesi/data/nsynth"
OUT = "/Users/alessandrolocurcio/Downloads/tesi/simulazione-dinamica-batch/tesi/figure_nsynth_nota"


def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr = np.load(os.path.join(DATA, "feat_nota_valid_X.npy"))
    Ytr = np.load(os.path.join(DATA, "feat_nota_valid_y.npy"))
    Xte = np.load(os.path.join(DATA, "feat_nota_test_X.npy"))
    Yte = np.load(os.path.join(DATA, "feat_nota_test_y.npy"))
    Xtr, Xte = rb.standardize(Xtr, Xte)
    C = 12
    N = Xtr.shape[0]
    prob = rb.LogReg(Xtr, Ytr, C, LAM)
    prob0 = rb.LogReg(Xtr, Ytr, C, 0.0)
    prob_te = rb.LogReg(Xte, Yte, C, 0.0)
    w0 = np.zeros(prob.p)
    print(f"train={N}, test={Xte.shape[0]}, classi={C}, parametri={prob.p}, d={Xtr.shape[1]}")

    methods = [
        ("Dynamic GD", lambda: rb.dynamic_gd(prob, N, w0, MAX_ITER, ALPHA, BATCH0, THETA)),
        ("Newton-CG", lambda: rb.newton_cg(prob, N, w0, MAX_ITER, ALPHA, BATCH0, THETA, R, MAXCG)),
        ("Newton-CG L1", lambda: rb.newton_l1(prob0, N, w0, MAX_ITER, ALPHA, BATCH0, NU, 1e-4, MAXCG, R, THETA)),
        ("BB-CCV", lambda: rb.bb_ccv(prob, N, w0, MAX_ITER, ALPHA, BATCH0, THETA)),
    ]

    results, accs, batches = {}, {}, {}
    for name, fn in methods:
        np.random.seed(SEED)
        t0 = time.time()
        hw, hn, hg = fn()
        dt = time.time() - t0
        wf = hw[-1]
        if name == "Newton-CG L1":
            loss_f = prob0.loss_full(wf) + NU * np.sum(np.abs(wf))
            nz = int(np.sum(wf != 0))
        else:
            loss_f = prob.loss_full(wf)
            nz = None
        acc = rb.accuracy(prob_te, wf, Xte, Yte)
        results[name] = dict(iter=len(hw) - 1, acc=acc, loss=loss_f,
                             gnorm=float(np.linalg.norm(hg[-1])), batch=max(hn),
                             batch0=hn[0], time=dt, nnz=nz)
        accs[name] = [rb.accuracy(prob_te, np.array(w), Xte, Yte) for w in hw]
        batches[name] = [int(nn) for nn in hn]
        print(f"{name:14s} iter={results[name]['iter']:3d} "
              f"acc={acc*100:5.2f}%  ||grad||={results[name]['gnorm']:.1e}  "
              f"tempo={dt:5.1f}s" + (f"  nnz={nz}" if nz is not None else ""))

    # riferimento sklearn (L-BFGS, stesse features gia' standardizzate)
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(Xtr, Ytr)
    acc_ref = m.score(Xte, Yte)
    print(f"Riferimento sklearn (L-BFGS): acc = {acc_ref*100:.2f}%")
    results["sklearn (L-BFGS)"] = dict(acc=acc_ref)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"risultati": results, "note": NOTE}, f, indent=2)
    np.savez(os.path.join(OUT, "curves.npz"), accs=accs, batches=batches)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Dynamic GD": "#1f77b4", "Newton-CG": "#d62728",
              "Newton-CG L1": "#2ca02c", "BB-CCV": "#9467bd"}
    ks = np.arange(0, MAX_ITER + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, _ in methods:
        a = np.asarray(accs[name]) * 100
        ax.plot(ks[:len(a)], a, lw=1.8, label=name, color=colors[name])
    ax.axhline(100.0 / 12, color="gray", ls="--", lw=1.2, label="Casuale (1/12)")
    ax.set_xlabel(r"Iterazione $k$")
    ax.set_ylabel(r"Accuratezza sul test (\%)")
    ax.set_title("NSynth nota: accuratezza vs iterazioni")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nota_accuracy.pdf"))
    fig.savefig(os.path.join(OUT, "nota_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, _ in methods:
        b = np.asarray(batches[name])
        ax.step(np.arange(len(b)), b, where="mid", lw=1.6, label=name, color=colors[name])
    ax.set_xlabel(r"Iterazione $k$")
    ax.set_ylabel(r"Dimensione del batch $n_k$")
    ax.set_title("NSynth nota: dinamica del batch (CCV)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nota_batch.pdf"))
    fig.savefig(os.path.join(OUT, "nota_batch.png"), dpi=150)
    plt.close(fig)

    print("\nOK:", OUT)


if __name__ == "__main__":
    main()
