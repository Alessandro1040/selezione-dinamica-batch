#!/bin/bash
# =====================================================================
# Compila il PDF DEFINITIVO della tesi dal documento unico sapthesis.
#
#   ./compila_tesi.sh
#     - compila tesi_sapthesis.tex (documento unico in classe sapthesis:
#       frontespizio \maketitle + Ringraziamenti + Abstract + Indice +
#       7 capitoli + appendici A-E + bibliografia)
#     - copia il risultato in tesi_finale.pdf (PDF definitivo)
#
# Il vecchio flusso (frontespizio.tex + tesi.tex con merge pypdf) non serve
# piu': il frontespizio e' ora integrato nel documento unico.
# tesi.tex (article) resta come documento di lavoro; bozza.tex/bozza.pdf
# sono la versione bozza.
# =====================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "[1/2] Compilazione tesi_sapthesis.tex (latexmk)..."
latexmk -pdf -shell-escape -interaction=nonstopmode tesi_sapthesis.tex

echo "[2/2] tesi_sapthesis.pdf -> tesi_finale.pdf ..."
cp tesi_sapthesis.pdf tesi_finale.pdf
echo "OK: tesi_finale.pdf aggiornato"

