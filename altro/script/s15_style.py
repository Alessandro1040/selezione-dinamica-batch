#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S15: footnote, stile, citazioni."""
PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

def rep(old, new, label):
    global s
    if old in s:
        s = s.replace(old, new, 1)
        print(f"  [ok] {label}")
    else:
        print(f"  [--] {label}")

# 1) footnote per la regola di aggiornamento del batch
rep(r"""    n_{k}^{\mathrm{new}} = \left\lceil \frac{\|\widehat{\mathcal{V}}\|_1}{\theta^2 \|\nabla J_{\mathcal{S}_k}(w_k)\|_2^2} \right\rceil\textsuperscript{1}""",
    r"""    n_{k}^{\mathrm{new}} = \left\lceil \frac{\|\widehat{\mathcal{V}}\|_1}{\theta^2 \|\nabla J_{\mathcal{S}_k}(w_k)\|_2^2} \right\rceil""",
    "rimozione textsuperscript")
rep("\\noindent \\textsuperscript{1} Nell'implementazione si aggiunge 1 per evitare che, quando il rapporto è esattamente intero, il batch non cresca, causando ripetute violazioni della CCV per fluttuazioni numeriche.\n",
    "",
    "rimozione nota textsuperscript")
rep("(ovvero la stima della varianza per predire la dimensione del batch k viene fatta con la varianza del batch k-1) si ottiene:",
    "(ovvero la stima della varianza per predire la dimensione del batch $k$ viene fatta con la varianza del batch $k-1$) si ottiene:\\footnote{Nell'implementazione si aggiunge 1 per evitare che, quando il rapporto \\`e esattamente intero, il batch non cresca, causando ripetute violazioni della CCV per fluttuazioni numeriche.}",
    "footnote regola batch")

# 2) stile: passaggi colloquiali nel Background
rep("Il problema? \\textbf{Calcolare il gradiente su 100 milioni di esempi ad ogni passo \\`e proibitivo.}",
    "Tuttavia, il calcolo del gradiente su $N$ dell'ordine di $10^8$ esempi ad ogni passo risulta proibitivo.",
    "stile 'Il problema?'")
rep("veloce ma rumoroso, come navigare a vista nella nebbia.",
    "veloce ma fortemente rumoroso.",
    "stile metafora nebbia")
rep("preciso ma lentissimo.",
    "preciso ma computazionalmente proibitivo.",
    "stile 'lentissimo'")

# 3) citazioni formali
rep("\\textbf{Confronto con il risultato di Nocedal (Theorem 4.1).}",
    "\\textbf{Confronto con il risultato di Nocedal e Wright~\\cite{nocedal2006}.}",
    "cita Nocedal 1")
rep("mentre Nocedal dimostra $1 - \\dfrac{(1-\\theta)\\lambda}{2L}$ (eq.~4.9 del paper).",
    "mentre Nocedal e Wright~\\cite{nocedal2006} dimostrano $1 - \\dfrac{(1-\\theta)\\lambda}{2L}$ (eq.~4.9).",
    "cita Nocedal 2")
rep("(eq.~4.4--4.5) utilizzata da Nocedal.",
    "(eq.~4.4--4.5) utilizzata da Nocedal e Wright~\\cite{nocedal2006}.",
    "cita Nocedal 3")
rep("in analogia con quanto fa il paper originale.",
    "in analogia con quanto fatto nel lavoro originale~\\cite{byrd2012}.",
    "cita paper originale")
rep("rispetto a quello ottenuto da Byrd et al. (2012), che presenta un fattore $1/2$ aggiuntivo",
    "rispetto a quello ottenuto da Byrd et al.~\\cite{byrd2012}, che presentano un fattore $1/2$ aggiuntivo",
    "cita Byrd et al 1")
rep("\\textbf{Differenza rispetto al paper (Byrd et al. 2012, Teorema 4.2).}",
    "\\textbf{Differenza rispetto al lavoro originale~\\cite{byrd2012} (Teorema 4.2).}",
    "cita Byrd et al 2")
rep("(operazione costosa), il paper propone di calcolarla \\emph{una sola volta}",
    "(operazione costosa), il lavoro originale~\\cite{byrd2012} propone di calcolarla \\emph{una sola volta}",
    "cita Byrd et al 3")

open(PATH, "w", encoding="utf-8").write(s)
print("S15 FATTO")
