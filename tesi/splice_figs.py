#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Innesto dei 3 diagrammi TikZ (Dynamic GD, Newton-CG, Newton-L1)."""
PATH = "main.tex"
s = open(PATH, encoding="utf-8").read()

fig51 = open("contenuti/fig5_1.tex").read()
figcg = open("contenuti/fig5_2_newtoncg.tex").read()
figl1 = open("contenuti/fig5_3_l1.tex").read()

# --- 1) sostituisce la vecchia Figura 5.1 con la versione migliorata
old51_start = s.find("% Figura 5.1 - schema a blocchi dell'algoritmo Dynamic GD")
if old51_start != -1:
    old51_end = s.find("\\end{figure}", old51_start)
    assert old51_end != -1
    old51_end += len("\\end{figure}")
    if "\\label{fig:block_diagram}" in s[old51_start:old51_end]:
        s = s[:old51_start] + fig51.strip() + "\n" + s[old51_end:]
        print("[ok] Fig 5.1 sostituita con versione migliorata")
    else:
        print("[--] Fig 5.1 non trovata (label mancante)")
else:
    print("[--] Fig 5.1 non trovata")

# --- 2) inserisce il diagramma Newton-CG dopo il suo pseudocodice
anchor_cg = ("\\end{actionbox}\n\\begin{mnbox}\n"
             "L'algoritmo del gradiente coniugato serve a minimizzare")
if anchor_cg in s and "fig:newton_cg" not in s:
    s = s.replace(anchor_cg,
                  "\\end{actionbox}\n\n" + figcg.strip() + "\n\n\\begin{mnbox}\n"
                  "L'algoritmo del gradiente coniugato serve a minimizzare", 1)
    print("[ok] Fig Newton-CG inserita")
else:
    print("[--] Fig Newton-CG: gia presente o ancoraggio mancante")

# --- 3) inserisce il diagramma Newton-L1 dopo il suo pseudocodice
anchor_l1 = ("\\end{actionbox}\n\n\\begin{notebox}\n"
             "  L'algoritmo appena descritto è particolarmente efficace")
if anchor_l1 in s and "fig:newton_l1" not in s:
    s = s.replace(anchor_l1,
                  "\\end{actionbox}\n\n" + figl1.strip() + "\n\n\\begin{notebox}\n"
                  "  L'algoritmo appena descritto è particolarmente efficace", 1)
    print("[ok] Fig Newton-L1 inserita")
else:
    print("[--] Fig Newton-L1: gia presente o ancoraggio mancante")

open(PATH, "w", encoding="utf-8").write(s)
print("FATTO")
