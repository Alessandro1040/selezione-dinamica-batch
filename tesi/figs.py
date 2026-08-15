#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Innesto Figure 5.1-5.3 e Tabella 5.1 nella sezione Algoritmi."""
PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

fig51 = open("contenuti/fig5_1.tex").read()
tab51 = open("contenuti/tab5_1.tex").read()

FIG52 = r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.78\textwidth]{figure/batch_size.pdf}
\caption{Andamento qualitativo della dimensione del batch $n_k$ in funzione
dell'iterazione $k$ per il metodo a campione dinamico (linea blu, con gradini
in corrispondenza delle violazioni della CCV) confrontato con un batch fisso
$n_k = 16$ (linea rossa tratteggiata). La strategia dinamica concentra la
risorsa computazionale nelle iterazioni finali, dove il gradiente \`e piccolo
e serve precisione.}
\label{fig:batch_qualitativa}
\end{figure}
"""

FIG53 = r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.68\textwidth]{figure/cono_discesa.pdf}
\caption{Rappresentazione geometrica della condizione di discesa: se il
gradiente stimato $g_k$ giace nel cono di tolleranza attorno al gradiente
esatto $\nabla J(w_k)$, con errore $\|e_k\| = \|g_k - \nabla J(w_k)\| \le
\theta\|g_k\|$, allora $-g_k$ \`e una direzione di discesa per $J$.}
\label{fig:cono}
\end{figure}
"""

def rep(old, new, label):
    global s
    if old in s and label not in s:
        s = s.replace(old, new, 1)
        print(f"  [ok] {label}")
    elif label in s:
        print(f"  [--] {label} gia presente")
    else:
        print(f"  [--] {label}: ancoraggio mancante")

# Figura 5.3 (cono) dopo il paragrafo sulla condizione di discesa
rep("dunque garantisce la discesa.\n\\newpage",
    "dunque garantisce la discesa.\n" + FIG53 + "\n\\newpage",
    "Fig 5.3 cono discesa")

# Figura 5.2 (batch size) dopo il notebox sull'aumento del batch
rep("\\underline{Il batch cresce automaticamente dove serve precisione.}\n\\end{notebox}",
    "\\underline{Il batch cresce automaticamente dove serve precisione.}\n\\end{notebox}\n" + FIG52,
    "Fig 5.2 batch size")

# Figura 5.1 (diagramma a blocchi) dopo lo pseudocodice
rep("\\end{actionbox}\n\\newpage\n\\subsubsection{Convergenza deterministica}",
    "\\end{actionbox}\n\n" + fig51 + "\n\\newpage\n\\subsubsection{Convergenza deterministica}",
    "Fig 5.1 diagramma a blocchi")

# Tabella 5.1 prima della sezione Esperimenti
rep("\\section{Esperimenti e Visualizzazione Interattiva}\\label{sec:esperimenti}",
    tab51 + "\n" + "\\section{Esperimenti e Visualizzazione Interattiva}\\label{sec:esperimenti}",
    "Tabella 5.1")

open(PATH, "w", encoding="utf-8").write(s)
print("FATTO")
