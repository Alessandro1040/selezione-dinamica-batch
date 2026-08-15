#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A6-A9: appendice A/B/C, bibliografia, pulizia blocchi commentati."""
import re

PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

def load(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

appB = load("contenuti/appendice_b.tex")
appC = load("contenuti/appendice_c.tex")
biblio = load("contenuti/biblio.tex")

print("== A6: struttura appendice A ==")
if "app:dimostrazioni" not in s:
    # section Appendice + label app:pl -> \appendix\section{Dimostrazioni}
    old = "\\section{Appendice}\n\\label{app:pl}"
    new = "\\appendix\n\\section{Dimostrazioni}\\label{app:dimostrazioni}"
    assert old in s, "sezione Appendice non trovata"
    s = s.replace(old, new, 1)
    # il label app:pl passa alla sottosezione PL
    old2 = ("\\subsection{Dimostrazione delle disuguaglianze di "
            "\\newline Polyak--Lojasiewicz}")
    new2 = old2 + "\\label{app:pl}"
    assert old2 in s, "sottosezione PL non trovata"
    s = s.replace(old2, new2, 1)
    # label app:corr
    old3 = ("\\subsection{Dimostrazione del fattore di correzione per "
            "\\newline popolazione finita}")
    new3 = old3 + "\\label{app:corr}"
    assert old3 in s, "sottosezione correzione non trovata"
    s = s.replace(old3, new3, 1)
    # label app:wolfe
    old4 = "\\subsection{Spiegazione delle condizioni sul passo $\\alpha_k$}"
    new4 = old4 + "\\label{app:wolfe}"
    assert old4 in s, "sottosezione Wolfe non trovata"
    s = s.replace(old4, new4, 1)
    print("  [ok] appendice A strutturata")
else:
    print("  [--] gia strutturata")

print("== A7: inserimento appendici B e C ==")
if "\section{Listati Python completi}" not in s:
    anchor = "\\begin{thebibliography}{99}"
    assert anchor in s, "thebibliography non trovata"
    s = s.replace(anchor, appB.strip() + "\n\n" + appC.strip() + "\n\n" + anchor, 1)
    print("  [ok] appendici B e C")
else:
    print("  [--] gia inserite")

print("== A8: sostituzione bibliografia ==")
if "bibitem{byrd2011}" not in s:
    start = s.find("\\begin{thebibliography}")
    end = s.find("\\end{thebibliography}")
    assert start != -1 and end != -1 and start < end
    s = s[:start] + biblio.strip() + "\n" + s[end + len("\\end{thebibliography}"):]
    print("  [ok] bibliografia sostituita")
else:
    print("  [--] bibliografia gia sostituita")

open(PATH, "w", encoding="utf-8").write(s)
print("PARTE 2 OK")
