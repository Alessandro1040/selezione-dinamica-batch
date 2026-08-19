#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera le tabelle LaTeX della Sezione 6 (Risultati Numerici) a partire dai
file OCR prodotti da ocr.swift (framework Vision di macOS). Le tabelle
riportano ||w_k - w_*||_2 a ogni iterazione per i quattro algoritmi
(Dynamic GD, Newton-CG, Newton-CG L1, BB-CCV); i file OCR dei metodi rimossi
dal documento possono essere semplicemente omessi dalla cartella.

Uso:
  python3 gen_tabelle_f.py <dir_ocr> <output_tex>

Formato file OCR:
  __ROW__ y=...
  0.050  0          <- colonna iter (x < 0.15)
  0.256  1.4142e+0   <- colonna errore (0.15 <= x < 0.45)
  0.625  1.0100e+2   <- colonna J(w) (x >= 0.45)
"""
import os, re, sys

METHODS = ["dgd", "ncg", "ncgl1", "bbccv"]
METHOD_NAMES = {
    "dgd":   "Dynamic GD",
    "ncg":   "Newton-CG",
    "ncgl1": "Newton-CG $L_1$",
    "bbccv": "BB-CCV",
}

PROBLEMS = [
    ("bencond",   "quadratico ben condizionato ($\\kappa \\approx 1.1$)", "tab:test_bencond"),
    ("malcond",   "quadratico molto mal condizionato ($\\kappa \\approx 100$)", "tab:test_malcond"),
    ("incrociato", "quadratico con termine incrociato", "tab:test_incrociato"),
]

def clean(tok):
    """Ripulisce un token OCR: spazi interni, e cirillica, spazi."""
    tok = tok.replace(" ", "")
    tok = tok.replace("\u0435", "e")  # e cirillica
    tok = tok.replace("\u2013", "-")
    return tok

def parse_file(fname):
    rows = []          # list of dict(col -> value)
    cur = None
    for line in open(fname, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("__ROW__"):
            if cur is not None:
                rows.append(cur)
            cur = {}
            continue
        parts = line.split(None, 1)
        if len(parts) < 2 or cur is None:
            continue
        try:
            x = float(parts[0])
        except ValueError:
            continue
        text = clean(parts[1])
        if x < 0.15:
            cur["iter"] = text
        elif x < 0.45:
            cur["err"] = text
        else:
            cur["J"] = text
    if cur is not None:
        rows.append(cur)
    return rows

def build_data(rows):
    """Produce lista (k, err) validando la sequenza di iterazioni."""
    data = []
    prev = -1
    for r in rows:
        if "iter" in r and not r["iter"].replace(".", "").isdigit():
            continue
        if "err" not in r:
            continue
        if "iter" in r and r["iter"].replace(".", "").isdigit():
            k = int(float(r["iter"]))
        else:
            k = prev + 1
        if k != prev + 1:
            raise ValueError(f"iterazione non sequenziale: atteso {prev+1}, trovato {k}")
        data.append((k, r["err"]))
        prev = k
    return data

def latex_num(s):
    """1.4142e+0 -> $1.4142\\times10^{0}$"""
    m, _, e = s.lower().partition("e")
    exp = int(e)
    return f"{m}\\times10^{{{exp}}}"

def main():
    if len(sys.argv) != 3:
        sys.exit("uso: gen_tabelle_f.py <dir_ocr> <output_tex>")
    base, outfile = sys.argv[1], sys.argv[2]
    out = []
    for prefix, desc, label in PROBLEMS:
        data_by_method = {}
        for m in METHODS:
            fname = os.path.join(base, f"{prefix}_{m}.txt")
            data = build_data(parse_file(fname))
            data_by_method[m] = data
            last = data[-1][0]
            print(f"{prefix}_{m}: {len(data)} righe, k 0..{last}")
        subtables = []
        for j, m in enumerate(METHODS):
            data = data_by_method[m]
            body = "\n".join(f"{k} & ${latex_num(err)}$\\\\" for k, err in data)
            sub = (
                "\\begin{subtable}{0.24\\textwidth}\n"
                "\\centering\n"
                f"\\caption{{{METHOD_NAMES[m]}}}\n"
                "\\begin{tabular}{@{}rl@{}}\n"
                "\\toprule\n"
                "$k$ & $\\|w_k-w_*\\|_2$\\\\\n"
                "\\midrule\n"
                f"{body}\n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "\\end{subtable}"
            )
            sep = "\\hfill" if j < len(METHODS) - 1 else ""
            subtables.append(sub + sep)
        out.append(f"% Tabella F - {desc}")
        out.append("\\begin{table}[H]")
        out.append("\\centering")
        out.append("\\footnotesize")
        out.append("\\renewcommand{\\arraystretch}{0.85}")
        out.append("\\setlength{\\tabcolsep}{3pt}")
        out.append(f"\\caption{{Errore $\\|w_k-w_*\\|_2$ a ogni iterazione sul problema "
                   f"{desc}, per i quattro algoritmi. I valori sono quelli riportati "
                   f"dal pannello \\emph{{Analisi}} dell'applicazione "
                   f"(Sezione~\\ref{{sec:visualizzazione}}).}}")
        out.append(f"\\label{{{label}}}")
        out.append("\n".join(subtables))
        out.append("\\end{table}")
        out.append("")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"OK -> {outfile}")

if __name__ == "__main__":
    main()

