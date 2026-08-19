#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Innesto dei contenuti e ristrutturazione finale di main.tex."""
import re

PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

def load(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

intro = load("contenuti/intro.tex")
lavori = load("contenuti/lavori.tex")
conclusioni = load("contenuti/conclusioni.tex")
sez6 = load("contenuti/sezione6.tex")
appB = load("contenuti/appendice_b.tex")
appC = load("contenuti/appendice_c.tex")
biblio = load("contenuti/biblio.tex")

# ---------------------------------------------------------------- A1
print("== A1: label sezioni ==")
s = s.replace("\\section{Formulazione del Problema}",
              "\\section{Formulazione del Problema}\\label{sec:formulazione}", 1)
s = s.replace("\\section{Algoritmi Proposti}",
              "\\section{Algoritmi Proposti}\\label{sec:algoritmi}", 1)
print("  [ok] label sec:formulazione, sec:algoritmi")

# ---------------------------------------------------------------- A2
print("== A2: inserimento Introduzione dopo TOC ==")
if "sec:introduzione" not in s:
    anchor = "\\tableofcontents\n\\clearpage\n"
    if anchor in s:
        s = s.replace(anchor, anchor + "\n" + intro + "\n", 1)
        print("  [ok] introduzione")
    else:
        print("  [--] ancoraggio TOC mancante")
else:
    print("  [--] gia inserita")

# ---------------------------------------------------------------- A3
print("== A3: inserimento Lavori Correlati prima di Algoritmi ==")
if "\section{Lavori Correlati}" not in s:
    anchor = "\\section{Algoritmi Proposti}\\label{sec:algoritmi}"
    if anchor in s:
        s = s.replace(anchor, lavori + "\n" + anchor, 1)
        print("  [ok] lavori correlati")
    else:
        print("  [--] ancoraggio Algoritmi mancante")
else:
    print("  [--] gia inseriti")

# ---------------------------------------------------------------- A4
print("== A4: sostituzione sezione Visualizzazione -> Esperimenti ==")
if "\section{Esperimenti e Visualizzazione Interattiva}" not in s:
    start = s.find("\\section{Visualizzazione Interattiva}")
    end = s.find("\\section{Appendice}")
    assert start != -1 and end != -1 and start < end
    s = s[:start] + sez6 + "\n" + s[end:]
    print("  [ok] sezione 6 sostituita")
else:
    print("  [--] gia sostituita")

# ---------------------------------------------------------------- A5
print("== A5: conclusione prima dell'appendice ==")
if "\section{Conclusioni e Lavoro Futuro}" not in s:
    anchor = "\\section{Appendice}"
    if anchor in s:
        s = s.replace(anchor, conclusioni + "\n" + anchor, 1)
        print("  [ok] conclusioni")
    else:
        print("  [--] ancoraggio appendice mancante")
else:
    print("  [--] gia inserite")

open(PATH, "w", encoding="utf-8").write(s)
print("PARTE 1 OK")
