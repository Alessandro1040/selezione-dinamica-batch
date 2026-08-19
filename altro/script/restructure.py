#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ristrutturazione di main.tex per la tesi (FASI 1-5)."""
import sys, re

PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()
orig_len = len(s)

def rep(old, new, count=1, label=""):
    global s
    n = s.count(old)
    if n == 0:
        print(f"  [MANCANTE] {label}")
        return False
    s = s.replace(old, new, count)
    print(f"  [ok x{n}] {label}")
    return True

print("== FASE 1a: frontespizio / ringraziamenti / abstract / TOC ==")

RINGRAZIAMENTI = r"""% ── RINGRAZIAMENTI ─────────────────────────────────────────────────────────
\newpage
\vspace*{3.5cm}
\begin{center}
  {\large\bfseries\color{coverblue} Ringraziamenti}\\[0.6cm]
\end{center}
\noindent
Desidero ringraziare il mio relatore per la guida, la pazienza e i preziosi
consigli che hanno accompagnato la stesura di questo lavoro, e per avermi
introdotto a un tema di ricerca tanto affascinante quanto attuale. Un
ringraziamento sincero va alla mia famiglia, per il sostegno costante e per
avermi sempre incoraggiato a perseguire i miei obiettivi, e agli amici e
colleghi che hanno condiviso con me questo percorso, rendendolo più ricco e
piacevole. Infine, un pensiero riconoscente va a tutti coloro che, con
suggerimenti e discussioni, hanno contribuito a migliorare questo elaborato.
\newpage
"""

ABSTRACT = r"""% ── ABSTRACT ────────────────────────────────────────────────────────────────
\begin{center}
  \rule{0.8\textwidth}{0.4pt}\\[0.4cm]
  {\large\bfseries\color{coverblue} Abstract}\\[0.3cm]
\end{center}
\noindent
Questa tesi presenta, in forma rigorosa ma accessibile, la metodologia per la
\emph{selezione dinamica della dimensione del campione} negli algoritmi di
ottimizzazione per il machine learning su larga scala, basandosi sul lavoro
seminale di Byrd, Chin, Nocedal e Wu~\cite{byrd2012}. L'idea centrale è la
seguente: anziché scegliere a priori una dimensione fissa del batch (il
sottoinsieme di dati usato ad ogni iterazione), è possibile decidere
\emph{durante} l'algoritmo se e quando aumentarla, guidandosi da stime
statistiche della varianza del gradiente stocastico. Vengono trattati tre
contributi principali: (i) un metodo del gradiente a campione dinamico con
analisi di convergenza lineare e stima di complessità; (ii) un metodo di
Newton con gradiente coniugato e Hessiana sottocampionata, con un criterio di
arresto adattivo per il risolutore interno; (iii) un'estensione ai problemi
con regolarizzazione $L_1$ per la produzione di soluzioni sparse. Il documento
è organizzato in sette capitoli: dopo l'introduzione e il background
necessario, si formula il problema di ottimizzazione, si passano in rassegna i
lavori correlati, si descrivono gli algoritmi proposti, se ne discute
l'implementazione interattiva con i risultati sperimentali e si traggono le
conclusioni con le prospettive di lavoro futuro. I dettagli dimostrativi e i
listati Python completi sono raccolti in appendice.
\begin{center}
  \rule{0.8\textwidth}{0.4pt}
\end{center}
\newpage
"""

# Rimuove il blocco commentato tra "nopagecolor" e "tableofcontents"
m = re.search(r"\\nopagecolor.*?\\tableofcontents", s, re.S)
assert m, "blocco nopagecolor/tableofcontents non trovato"
s = s.replace(m.group(0),
              "\\nopagecolor\n\n" + RINGRAZIAMENTI + ABSTRACT + "\\tableofcontents",
              1)
print("  [ok] nopagecolor + ringraziamenti + abstract + TOC")

# Toglie il wrapper \begin{comment} ... \end{comment} intorno a Concetti di Base
s = s.replace("\\clearpage\n\\begin{comment}\n\\newpage\n",
              "\\clearpage\n", 1)
print("  [ok] rimozione begin{comment} (Concetti di Base)")
end_comm = "\\end{comment}\n\\newpage\n% " + "\u2550" * 0  # marker da gestire dopo
# la chiusura del commento: cerchiamo l'\end{comment} che precede la sezione Formulazione
m2 = re.search(r"\\end\{comment\}(\s*\\newpage)?\s*\n% \u2550+\n\\section\{Formulazione del Problema\}", s)
assert m2, "end{comment} prima di Formulazione non trovato"
s = s.replace(m2.group(0), "\n\\newpage\n\n% " + "\u2550" * 67 + "\n\\section{Formulazione del Problema}", 1)
print("  [ok] rimozione end{comment} (Concetti di Base)")

open(PATH, "w", encoding="utf-8").write(s)
print(f"Lunghezza: {orig_len} -> {len(s)}")
print("PASSO 1 COMPLETATO")
