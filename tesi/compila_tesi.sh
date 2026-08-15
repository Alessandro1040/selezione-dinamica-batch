#!/bin/bash
# =====================================================================
# Compila il PDF finale della tesi con il frontespizio istituzionale.
#
# Uso (dalla cartella che contiene i sorgenti):
#   ./compila_tesi.sh [main | tesi]
#     - main : documento = main.tex  (copia di lavoro in Downloads/tesi)
#     - tesi : documento = tesi.tex  (documento della repo)
#
# Produce tesi_finale.pdf = frontespizio.pdf (2 pp: frontespizio + verso)
# + documento senza la vecchia copertina custom (pag. 1 di main/tesi.pdf).
# La copertina custom resta nel sorgente; per ripristinarla nel PDF
# basta rimuovere lo slicing "body.pages[1:]" qui sotto.
# =====================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

DOC="${1:-}"
if [ -z "$DOC" ]; then
  if   [ -f main.tex ]; then DOC=main
  elif [ -f tesi.tex ]; then DOC=tesi
  else echo "Nessun documento trovato (main.tex o tesi.tex)."; exit 1
  fi
fi
case "$DOC" in
  main|main.tex) DOC=main ;;
  tesi|tesi.tex) DOC=tesi ;;
  *) echo "Argomento non valido: usa 'main' o 'tesi'."; exit 1 ;;
esac

echo "[1/3] Frontespizio (sapthesis)..."
pdflatex -interaction=nonstopmode -halt-on-error frontespizio.tex >/dev/null

echo "[2/3] Documento $DOC.tex (latexmk)..."
latexmk -pdf -shell-escape -interaction=nonstopmode "$DOC.tex" >/dev/null

echo "[3/3] Merge: frontespizio + documento..."
python3 - "$DOC" <<'PY'
import sys
from pypdf import PdfReader, PdfWriter
doc = sys.argv[1]
fe = PdfReader('frontespizio.pdf')
body = PdfReader(doc + '.pdf')
out = PdfWriter()
for p in fe.pages[:1]:            # solo il frontespizio (il verso non è generato)
    out.add_page(p)
for p in body.pages[1:]:            # salta la vecchia copertina custom (pag. 1)
    out.add_page(p)
out.write('tesi_finale.pdf')
print("OK: tesi_finale.pdf (%d pagine)" % len(out.pages))
PY
