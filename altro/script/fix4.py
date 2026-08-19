#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix S4 (box norma matrice) e S6 (transizione) su main.tex."""
import sys

PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

NUOVO4 = (
    " \\begin{mnbox}\n"
    "  \\textbf{Numero di condizionamento e velocit\\`a di convergenza.}\n"
    "  Le norme di matrice sono fondamentali per analizzare la stabilit\\`a e la\n"
    "  convergenza degli algoritmi. Per una matrice $A$ simmetrica definita\n"
    "  positiva, il \\textbf{numero di condizionamento} \\`e definito come\n"
    "  \\[\n"
    "  \\kappa(A) = \\frac{\\lambda_{\\max}(A)}{\\lambda_{\\min}(A)}.\n"
    "  \\]\n"
    "  Questo numero misura la \\textbf{sensibilit\\`a} del problema alla\n"
    "  propagazione degli errori: pi\\`u $\\kappa(A)$ \\`e grande, pi\\`u piccole\n"
    "  perturbazioni nei dati o negli arrotondamenti possono amplificarsi nella\n"
    "  soluzione. Un problema con $\\kappa(A)$ grande si dice \\textbf{mal\n"
    "  condizionato}: la soluzione \\`e numericamente instabile e gli algoritmi\n"
    "  iterativi (come il metodo del gradiente) convergono lentamente. Per una\n"
    "  funzione quadratica $f(w) = \\frac12 w^T A w$, con $A$ simmetrica\n"
    "  definita positiva, $\\kappa = \\lambda_{\\max}(A)/\\lambda_{\\min}(A)$, e\n"
    "  nel metodo del gradiente con passo ottimale l'errore in valore di\n"
    "  funzione decresce come\n"
    "  \\[\n"
    "    f(w_k) - f(w_*) \\;\\le\\; \\left(\\frac{\\kappa - 1}{\\kappa + 1}\\right)^{2k}\n"
    "    \\bigl(f(w_0) - f(w_*)\\bigr).\n"
    "  \\]\n"
    "  Pi\\`u $\\kappa$ \\`e grande, pi\\`u lenta \\`e la convergenza: il fattore di\n"
    "  contrazione $(\\kappa-1)/(\\kappa+1)$ tende a $1$ quando\n"
    "  $\\kappa\\to\\infty$, e per un problema mal condizionato servono\n"
    "  moltissime iterazioni per ridurre l'errore.\n"
    "\n"
    "  Nel nostro contesto, la matrice Hessiana $\\nabla^2 J(w)$ ha autovalori\n"
    "  compresi tra $\\lambda$ e $L$ (per le ipotesi di convessit\\`a forte e\n"
    "  Lipschitzianit\\`a), quindi il suo numero di condizionamento \\`e\n"
    "  $\\kappa = L/\\lambda$. Nel Machine Learning $\\kappa$ pu\\`o essere enorme,\n"
    "  da qui la necessit\\`a di metodi pi\\`u avanzati come Newton o il gradiente\n"
    "  coniugato (Sezione~\\ref{sec:algoritmi}).\n"
    "\\end{mnbox}"
)

if "Numero di condizionamento e velocit" not in s:
    ini = s.find("\\begin{mnbox}")
    fine = s.find("\\end{mnbox}", ini)
    blk = s[ini:fine + len("\\end{mnbox}")]
    if "Metodi Numerici" in blk:
        s = s.replace(blk, NUOVO4, 1)
        print("[ok] S4 applicato")
    else:
        print("[--] S4: primo mnbox non contiene 'Metodi Numerici'")
        sys.exit(1)
else:
    print("[--] S4 gia applicato")

TRANS = (
    "\\subsection{Collegamento con la formulazione del problema}\n"
    "\n"
    "I concetti introdotti in questo capitolo -- convessit\\`a, gradiente,\n"
    "norme, condizionamento e varianza campionaria -- sono gli strumenti con\n"
    "cui il prossimo capitolo formalizza il problema di \\emph{ottimizzazione\n"
    "empirica su larga scala}: dato un dataset di $N$ esempi, si tratta di\n"
    "minimizzare la perdita media $J(w)$ su un insieme di parametri\n"
    "$w\\in\\R^m$, sapendo che il calcolo esatto del gradiente su tutti gli\n"
    "esempi \\`e troppo costoso. La varianza della popolazione $\\mathcal{V}$ e\n"
    "la sua stima campionaria forniranno il criterio per decidere,\n"
    "dinamicamente, quanti dati usare ad ogni iterazione.\n"
)

if "Collegamento con la formulazione" not in s:
    # trova il \newpage subito dopo la VBOX (ultimo \end{notebox} prima di Formulazione)
    marker = "\\section{Formulazione del Problema}"
    cut = s.find(marker)
    if cut == -1:
        print("[--] S6: sezione Formulazione non trovata")
        sys.exit(1)
    area = s[:cut]
    iend = area.rfind("\\end{notebox}")
    if iend == -1:
        print("[--] S6: nessun notebox in Background")
        sys.exit(1)
    inew = s.find("\\newpage", iend)
    if inew == -1:
        print("[--] S6: nessun \\newpage dopo la VBOX")
        sys.exit(1)
    s = s[:inew] + TRANS + "\\newpage" + s[inew + len("\\newpage"):]
    print("[ok] S6 applicato")
else:
    print("[--] S6 gia applicato")

open(PATH, "w", encoding="utf-8").write(s)
print("FATTO")
