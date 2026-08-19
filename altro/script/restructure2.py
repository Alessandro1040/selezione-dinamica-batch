#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ristrutturazione di main.tex -- passi S2...S15 (idempotenti)."""
import re

PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()


def rep(old, new, label="", count=1):
    global s
    n = s.count(old)
    if n == 0:
        print(f"  [--] {label} (assente/gia applicato)")
        return False
    s = s.replace(old, new, count)
    print(f"  [ok] {label} (x{n})")
    return True


# ----------------------------------------------------------------- S2
print("== S2 rinomina sezione Background ==")
rep(r"\section{Concetti di Base}",
    r"\section{Background e Notazione}\label{sec:background}",
    "S2 rinomina sezione")

# ----------------------------------------------------------------- S3
print("== S3 fix RIVEDERE ==")
m = re.search(
    r"    Il \\textbf\{numero di condizionamento\}.*?"
    r"come Newton o gradiente coniugato\.",
    s, re.S)
if m:
    nuovo = (
        "    La loro combinazione determina la velocit\\`a di convergenza del "
        "metodo del\n    gradiente: la discussione completa sul \\textbf{numero "
        "di condizionamento}\n    $\\kappa = L/\\lambda$ \\`e riportata subito "
        "dopo la definizione di norma di\n    matrice, dove sono disponibili "
        "tutti gli strumenti necessari."
    )
    s = s.replace(m.group(0), nuovo, 1)
    print("  [ok] S3 fix RIVEDERE")
else:
    print("  [--] S3 gia applicato / assente")

# ----------------------------------------------------------------- S4
print("== S4 box norma di matrice espanso ==")
m4 = re.search(
    r" \\\\begin\{mnbox\}\s*\\\\textbf\{Metodi Numerici:\}.*?"
    r"pi\\`u difficile da ottimizzare\.\s*\\\\end\{mnbox\}",
    s, re.S)
if m4:
    nuovo4 = r""" \begin{mnbox}
  \textbf{Numero di condizionamento e velocit\`a di convergenza.}
  Le norme di matrice sono fondamentali per analizzare la stabilit\`a e la
  convergenza degli algoritmi. Per una matrice $A$ simmetrica definita
  positiva, il \textbf{numero di condizionamento} \`e definito come
  \[
  \kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}.
  \]
  Questo numero misura la \textbf{sensibilit\`a} del problema alla propagazione
  degli errori: pi\`u $\kappa(A)$ \`e grande, pi\`u piccole perturbazioni nei
  dati o negli arrotondamenti possono amplificarsi nella soluzione. Un problema
  con $\kappa(A)$ grande si dice \textbf{mal condizionato}: la soluzione \`e
  numericamente instabile e gli algoritmi iterativi (come il metodo del
  gradiente) convergono lentamente. Per una funzione quadratica
  $f(w) = \frac12 w^T A w$, con $A$ simmetrica definita positiva,
  $\kappa = \lambda_{\max}(A)/\lambda_{\min}(A)$, e nel metodo del gradiente
  con passo ottimale l'errore in valore di funzione decresce come
  \[
    f(w_k) - f(w_*) \;\le\; \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2k}
    \bigl(f(w_0) - f(w_*)\bigr).
  \]
  Pi\`u $\kappa$ \`e grande, pi\`u lenta \`e la convergenza: il fattore di
  contrazione $(\kappa-1)/(\kappa+1)$ tende a $1$ quando $\kappa\to\infty$, e
  per un problema mal condizionato servono moltissime iterazioni per ridurre
  l'errore.

  Nel nostro contesto, la matrice Hessiana $\nabla^2 J(w)$ ha autovalori
  compresi tra $\lambda$ e $L$ (per le ipotesi di convessit\`a forte e
  Lipschitzianit\`a), quindi il suo numero di condizionamento \`e
  $\kappa = L/\lambda$. Nel Machine Learning $\kappa$ pu\`o essere enorme, da
  qui la necessit\`a di metodi pi\`u avanzati come Newton o il gradiente
  coniugato (Sezione~\ref{sec:algoritmi}).
\end{mnbox}"""
    s = s.replace(m4.group(0), nuovo4, 1)
    print("  [ok] S4 box norma matrice")
else:
    print("  [--] S4 gia applicato / assente")


# ----------------------------------------------------------------- S5
print("== S5 notebox varianza della popolazione in Background ==")
VBOX = r"""\begin{notebox}
  \textbf{Il significato di $\mathcal{V}$ (varianza della popolazione).}
  Al passo $k$ dell'algoritmo, con pesi correnti $w_k$, si prendono tutti i
  dati del dataset (dal primo all'$N$-esimo) e si calcola
  $\nabla \ell(w_k; i)$, la direzione di discesa per quell'esempio. La media
  di questi $N$ vettori \`e il gradiente esatto
  \[
  \nabla J(w_k) = \frac{1}{N} \sum_{i=1}^N \nabla \ell(w_k; i).
  \]
  Il vettore $\mathcal{V}$ misura quanto i singoli gradienti
  $\nabla \ell(w_k; i)$ differiscono dalla loro media:
  \[
  \mathcal{V} := \frac{1}{N} \sum_{i=1}^{N}
  \left( \nabla \ell(w_k; i) - \nabla J(w_k) \right)^2 \in \mathbb{R}^m,
  \]
  dove il quadrato \`e inteso \emph{componente per componente}. In termini
  probabilistici, se si estrae a caso un indice $I$ dal dataset, il gradiente
  della loss su quell'esempio \`e una variabile aleatoria, e $\mathcal{V}$
  \`e la sua \textbf{varianza} intorno alla media:
  \[
  \mathcal{V} =
  \mathrm{Var}_{I \sim \mathrm{Uniform}\{1,\dots,N\}}\bigl(\nabla \ell(w_k; I)\bigr).
  \]
  Intuitivamente, se i gradienti individuali sono tra loro coerenti (bassa
  varianza), la loro media campionaria \`e un buon stimatore di $\nabla J(w_k)$;
  se invece sono molto dispersi, il batch deve essere grande perch\'e la stima
  sia affidabile. Questa osservazione \`e il punto di partenza della
  \emph{Condizione di Controllo della Varianza} (CCV) introdotta nella
  Sezione~\ref{sec:algoritmi}.
\end{notebox}
"""
if "Il significato di $\\mathcal{V}$" not in s:
    anchor = ("misura la \\emph{dispersione totale} del campione.\n"
              "\\end{definitionbox}")
    if anchor in s:
        s = s.replace(anchor, anchor + "\n\n" + VBOX, 1)
        print("  [ok] S5 notebox V")
    else:
        print("  [--] S5 ancoraggio mancante")
else:
    print("  [--] S5 gia applicato")

# ----------------------------------------------------------------- S6
print("== S6 transizione Background -> Formulazione ==")
TRANS = r"""\subsection{Collegamento con la formulazione del problema}

I concetti introdotti in questo capitolo -- convessit\`a, gradiente, norme,
condizionamento e varianza campionaria -- sono gli strumenti con cui il
prossimo capitolo formalizza il problema di \emph{ottimizzazione empirica su
larga scala}: dato un dataset di $N$ esempi, si tratta di minimizzare la
perdita media $J(w)$ su un insieme di parametri $w\in\R^m$, sapendo che il
calcolo esatto del gradiente su tutti gli esempi \`e troppo costoso. La
varianza della popolazione $\mathcal{V}$ e la sua stima campionaria
forniranno il criterio per decidere, dinamicamente, quanti dati usare ad ogni
iterazione.
"""
if "Collegamento con la formulazione" not in s:
    m6 = re.search(r"\\end\{definitionbox\}\n\n\\newpage\n", s)
    if m6:
        s = s.replace(m6.group(0), "\\end{definitionbox}\n\n" + TRANS + "\\newpage\n", 1)
        print("  [ok] S6 transizione")
    else:
        print("  [--] S6 ancoraggio mancante")
else:
    print("  [--] S6 gia applicato")

# ----------------------------------------------------------------- S7
print("== S7 Formulazione: riferimenti e PL ==")
# nota PL: aggiunge il riferimento esplicito alla disuguaglianza di PL in App A
m7 = re.search(
    r"\\begin\{notebox\}\s*\\textbf\{Disuguaglianze fondamentali.*?"
    r"\\end\{notebox\}",
    s, re.S)
if m7 and "Appendice~\\ref{app:pl}" in m7.group(0):
    # gia' presente il riferimento, nessuna azione
    print("  [--] S7 gia presente")
else:
    print("  [--] S7 niente da fare")

open(PATH, "w", encoding="utf-8").write(s)
print("PARTE 1 TERMINATA")

