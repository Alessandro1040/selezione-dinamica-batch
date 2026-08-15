#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S9: rinumerazione delle equazioni nella sezione Algoritmi Proposti."""
import re

PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

A = s.find("\\section{Algoritmi Proposti}")
B = s.find("\\section{Visualizzazione Interattiva}")
assert A != -1 and B != -1 and A < B
sec = s[A:B]
rest = s[:A] + s[B:]

# --- 1) tag 3.N -> 5.N
for n in range(1, 31):
    sec = sec.replace(f"\\tag{{3.{n}}}", f"\\tag{{5.{n}}}")
# --- 2) tag 4.N -> 5.30+N
for n in range(1, 11):
    sec = sec.replace(f"\\tag{{4.{n}}}", f"\\tag{{5.{30 + n}}}")
sec = sec.replace("\\tag{4.5a}", "\\tag{5.35a}")
sec = sec.replace("\\tag{4.5b}", "\\tag{5.35b}")
# --- 3) tag 6.N -> 5.40+N
for n in range(1, 12):
    sec = sec.replace(f"\\tag{{6.{n}}}", f"\\tag{{5.{40 + n}}}")

# --- 4) riferimenti testuali (3.N) -> (5.N)
for n in range(1, 31):
    sec = sec.replace(f"({3}.{n})" if False else f"(3.{n})", f"(5.{n})")
# --- 5) riferimenti testuali (4.N) -> (5.30+N)
for n in range(1, 11):
    sec = sec.replace(f"(4.{n})", f"(5.{30 + n})")
sec = sec.replace("(4.5a)", "(5.35a)")
sec = sec.replace("(4.5b)", "(5.35b)")
# --- 6) riferimenti testuali (6.N) -> (5.40+N)
for n in range(1, 12):
    sec = sec.replace(f"(6.{n})", f"(5.{40 + n})")

# --- riferimenti nominali
sec = sec.replace("Corollario 3.2", "Corollario 5.2")
sec = sec.replace("Corollario~3.2", "Corollario~5.2")
sec = sec.replace("Teorema 3.1", "Teorema 5.1")
sec = sec.replace("sez.~3.2.2", "sez.~5.1.2")
sec = sec.replace("sez.~4.2.2", "sez.~5.2.2")
sec = sec.replace("Lemma~2", "Lemma~\\ref{lem:accuratezza}")
sec = sec.replace("Equation (6.11)", "\\eqref{eq:cg_l1}")
sec = sec.replace("la (6.4)", "la (5.44)")

s = s[:A] + sec + s[B:]

# --- aggiunge label al Lemma e label al criterio CG L1
if "\\begin{lemma}" in s and "\\label{lem:accuratezza}" not in s:
    s = s.replace("\\begin{lemma}", "\\begin{lemma}\\label{lem:accuratezza}", 1)
    print("[ok] label lemma")
if "\\label{eq:cg_l1}" not in s:
    s = s.replace(
        "\\|r_k\\| \\le \\eta \\, \\|Y_k^T \\widetilde{\\nabla}F_{\\mathcal{S}_k}(w_k)\\|, \\tag{5.51}",
        "\\|r_k\\| \\le \\eta \\, \\|Y_k^T \\widetilde{\\nabla}F_{\\mathcal{S}_k}(w_k)\\|, \\tag{5.51}\\label{eq:cg_l1}",
        1)
    print("[ok] label eq cg_l1")

# --- 'definita in (1)' -> eqref empirical risk
s = s.replace("$J(w)$ è la funzione di perdita empirica definita in (1).",
              "$J(w)$ è la funzione di perdita empirica definita in \\eqref{eq:empirical_risk}.")

# --- condizione di discesa: assegna un label all'equazione della CCV-accettazione
# la condizione ||e_k||<=theta||g_k|| mostrata come equazione
if "\\label{eq:discesa}" not in s:
    old_d = ("\\|g_k - \\nabla J(w_k)\\|_2 \\;\\leq\\; \\theta \\|g_k\\|_2, \\quad \\theta \\in [0,1).\n"
             "\\]\n"
             "\n"
             "Per capire perch\\'e questo garantisce la discesa")
    new_d = ("\\|g_k - \\nabla J(w_k)\\|_2 \\;\\leq\\; \\theta \\|g_k\\|_2, \\quad \\theta \\in [0,1).\n"
             "\\label{eq:discesa}\n"
             "\\]\n"
             "\n"
             "Per capire perch\\'e questo garantisce la discesa")
    if old_d in s:
        s = s.replace(old_d, new_d, 1)
        print("[ok] label eq:discesa")
    else:
        print("[--] eq:discesa non trovata (la gestisco dopo)")

open(PATH, "w", encoding="utf-8").write(s)
print("S9 COMPLETATO")
