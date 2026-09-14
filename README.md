# Selezione Dinamica della Dimensione del Campione

Tesi e materiale per *"Selezione Dinamica della Dimensione del Campione in
Metodi di Ottimizzazione per il Machine Learning"* — Corso di Laurea in
Scienze Matematiche per l'Intelligenza Artificiale, Sapienza Università di
Roma (A.A. 2025–2026).

## Contenuto del repository

```
.
├── README.md                      questo file
├── README_visualizzazione.md      guida utente dell'applicazione web (visualizzazione.html)
├── visualizzazione.html           applicazione web interattiva (Pyodide + Plotly)
├── simulazione_batch.py           simulazione autonoma della Figura 5.3 (n_k vs k)
├── presentazione_riformattata.pptx  presentazione della tesi in PowerPoint (23 slide, riformattata)
├── presentazione_riformattata.pdf   PDF della presentazione (23 slide, 16:9) - stessa versione del .pptx
├── figure_sim/                    figure generate da simulazione_batch.py
├── tesi/                          SOLO ciò che serve a compilare tesi_finale.pdf
│   ├── tesi.tex                   documento di lavoro (article, con copertina; contiene anche la Sez. 6.7 sul riuso del mini-batch)
│   ├── tesi.pdf                   PDF compilato del documento di lavoro
│   ├── tesi_finale.pdf            PDF DEFINITIVO: frontespizio + documento
│   ├── frontespizio.tex           frontespizio istituzionale (sapthesis, 1 p)
│   ├── compila_tesi.sh            compila frontespizio + tesi.tex -> tesi_finale.pdf
│   ├── conodiscesa2.jpeg          figura del cono di discesa (copertina)
│   ├── figure_sim/                solo batch_size_app.png (Fig. 5.x usata in tesi.tex)
│   ├── figure_nsynth_nota/        SOLO i PDF usati in tesi.tex (nota_accuracy, nota_batch)
│   ├── figure_nsynth_net/         SOLO i PDF usati in tesi.tex (nsynth_accuracy_net, nsynth_batch_net)
│   └── nsynth/                    SOLO i notebook Colab citati nella tesi
├── colab_risorse/                 risorse per far funzionare i Colab
│   ├── pesi_net.npz               pesi dei 4 modelli (rete neurale NSynth)
│   ├── scaler_net.npz             RobustScaler 5–95 del training
│   ├── pesi_nota.npz              pesi dei 4 modelli (riconoscimento della nota) + mu/sd
│   ├── features_nota.npz          features del test set NSynth (Xte standardizzato, Yte, nomi clip)
│   ├── features_opt_net_test.npz  features della rete, split test (4 096 clip, X/y/names)
│   ├── features_opt_net_valid.npz features della rete, split valid (12 678 clip, X/y/names)
│   └── figure/                    figure prodotte dai Colab
│       ├── nota/                  esperimento riconoscimento della nota
│       ├── net/                   esperimento rete neurale
│       └── famiglia/              esperimento famiglia strumentale (tagliato dalla tesi)
└── altro/                         materiale storico/non usato (non serve a compilare)
    ├── tesi_sapthesis.tex/.pdf    documento unico sapthesis (riferimento storico)
    ├── bozza.tex/.pdf             versione bozza/draft (riferimento storico)
    ├── appendice_riuso.tex + appendice_riuso_estratto.tex/.pdf
    │                              ex Appendice E (riuso mini-batch), ora Sez. 6.7 di tesi.tex (riferimento storico)
    ├── metodinumerici.tex         vecchia bozza completa (riferimento storico)
    ├── contenuti/                 frammenti LaTeX dei capitoli (già incorporati in tesi.tex)
    ├── figure/                    vecchie figure non più usate in tesi.tex
    ├── figure_test/               PNG di test (analisi OCR)
    ├── figure_sim/                vecchia copia di batch_size.pdf
    ├── ocr_f/ + ocr_appendix_f.swift   testi estratti con OCR dall'Appendice F
    ├── script/                    script di supporto (bbccv.py, rapg.py, sim_exp.py, ...)
    ├── tabelle/                   frammenti di tabelle (tabella6_1, tabella6_2)
    └── nsynth/                    notebook non citati + script degli esperimenti NSynth
```

## Applicazione web interattiva

Apri `visualizzazione.html` in un browser moderno (Chrome, Firefox, Safari,
Edge). L'app esegue Python nel browser tramite **Pyodide** e consente di:
- scegliere/modificare la funzione obiettivo (preset 1D/2D o codice custom),
- eseguire i quattro algoritmi: Dynamic GD, Newton-CG, Newton-CG $L_1$,
  BB-CCV (più l'algoritmo personalizzato, in Python, scritto dall'utente),
- osservare il percorso su Plotly, la dimensione del batch ($n_k$ vs $a^k$)
  e l'analisi di convergenza.

👉 **Guida utente completa: [`README_visualizzazione.md`](README_visualizzazione.md)** —
pannelli, ogni controllo, tutti i grafici, gli esperimenti guidati e la mappa
delle sezioni della tesi.

## Simulazione (come l'applicazione web)

```bash
python3 simulazione_batch.py
```

Riproduce **fedelmente l'implementazione dell'app `visualizzazione.html`**:
preset *Quadratica ben condizionata (κ≈1.1)*, dataset sintetico "centrato"
(media campionaria dei coefficienti = coefficienti esatti di $J$), algoritmo
**Dynamic GD** con CCV e **line search di Wolfe** (default dell'app).

Genera in `figure_sim/`:
- `batch_size.pdf/png` — $n_k$ vs $k$ (dinamico CCV, fit $a^k$, batch fisso),
- `convergenza.pdf/png` — $\|w_k-w_*\|$ vs $k$ (scala log, metrica `errs` dell'app).

Parametri (default dell'app): preset `quad_well` ($J=(w_1-1)^2+(w_2+2)^2+0.1w_1w_2$),
$N=200$, $w_0=[2,-3]$, $\alpha=0.1$, $\theta=0.5$, batch$_0=5$, 30 iterazioni, seed 42.

## Tesi LaTeX

Documento di lavoro (article, con copertina):

```bash
cd tesi
latexmk -pdf -shell-escape tesi.tex    # serve pygments per i listati minted
```

**PDF definitivo** (frontespizio istituzionale sapthesis come prima pagina,
seguito dal documento senza copertina):

```bash
cd tesi
./compila_tesi.sh    # frontespizio.tex + tesi.tex (merge pypdf) -> tesi_finale.pdf
```

`tesi_finale.pdf` è il PDF definitivo: **frontespizio istituzionale**
(`frontespizio.tex`, sapthesis) come prima pagina + il documento di lavoro
`tesi.tex` (article) senza la copertina, uniti con un **merge pypdf** (i due
pezzi sono documenti separati). `bozza.tex`/`bozza.pdf` sono la versione
bozza. `tesi_sapthesis.tex`/`tesi_sapthesis.pdf` (il periodo in cui il PDF
definitivo era un documento unico in classe `sapthesis`) sono spostati in
`altro/` come riferimento storico.

## Note operative e stato corrente (29/08/2026)

Da tenere presente nelle sessioni di lavoro successive:

- **File in lavorazione.** La copia di lavoro è la cartella sul Desktop
  (NON è un clone git):
  `/Users/alessandrolocurcio/Desktop/Selezione Dinamica della Dimensione del
  Campione in Metodi di Ottimizzazione per il Machine Learning/`. Le modifiche
  si fanno in `tesi/tesi.tex` lì (documento di lavoro, article); il PDF
  definitivo si rigenera con `./compila_tesi.sh` (frontespizio + `tesi.tex`,
  merge pypdf -> `tesi_finale.pdf`). I PDF si rigenerano sul posto e
  prima del commit i file vanno copiati nella repo
  (`cp <Desktop>/.../tesi/<file> tesi/<file>`), verificando con `md5` che
  copia Desktop e repo coincidano. La vecchia copia
  `/Users/alessandrolocurcio/Downloads/tesi/main.tex` resta solo come
  riferimento storico. **Struttura della repo (19/08/2026):** `tesi/` contiene
  solo ciò che serve a compilare `tesi_finale.pdf` (documento, frontespizio,
  PDF usati, notebook Colab citati nella tesi); le figure
  complete prodotte dai Colab (PDF/PNG/npz/json) sono in `colab_risorse/figure/`
  (sottocartelle `nota/`, `net/`, `famiglia/`) con i pesi dei modelli in
  `colab_risorse/` (`pesi_net.npz`, `scaler_net.npz`, `pesi_nota.npz`,
  `features_nota.npz`) e le features già estratte della rete
  (`features_opt_net_test.npz`, `features_opt_net_valid.npz`); tutto il materiale
  storico/non usato è in `altro/` (tesi_sapthesis, bozza, contenuti/,
  figure_test/, script di supporto, notebook non citati, ecc.).
- **Documento autocontenuto.** `tesi.tex` NON usa `\input` per i capitoli: i frammenti in
  `altro/contenuti/` sono già incorporati nel file. Non tentare di ricostruire
  il documento a partire da `altro/contenuti/`. L'ex Appendice E (riuso del
  mini-batch) è stata eliminata come appendice (24/08/2026): il contenuto è ora
  integrato in `tesi.tex` come sottosezione **6.7** "Riuso del mini-batch:
  iterazioni consecutive sullo stesso campione" (sottosezioni 6.7.1–6.7.7, 34
  tabelle). I vecchi sorgenti autonomi (`appendice_riuso.tex` e
  `appendice_riuso_estratto.tex/.pdf`) sono in `altro/` come riferimento
  storico e NON vanno ricompilati per `tesi_finale.pdf`.
- **Tabelle del riuso (metodi di Newton).** Nelle tabelle dei metodi del
  secondo ordine (8 principali `tab:riuso_*_ncg/_l1` e 8 consigliati
  `tab:riuso_cons_*_ncg/_nl1`) l'intestazione è a riga unica con nomi
  descrittivi (`base`, `$M{=}\infty$`, `$M{=}10$`, `$M{=}5$`, `$M{=}2$`,
  `H ind. $M_H{=}\infty$`); le didascalie spiegano l'ultima colonna
  (modalità *Indipendente da $S_k$*) e, per i consigliati, la configurazione
  consigliata (stop adattivo `$P{=}1$, $f{=}1$, $p{=}10\%$, split fissa` per
  Newton-CG; riuso `$M{=}3$` per Newton-CG $L_1$). Il generatore
  `altro/script/gen_tabelle_riuso.py` (`--tex`) scrive le 8 tabelle Newton in
  questo formato.
- **Figura di copertina.** La compilazione della copia di lavoro richiede
  `conodiscesa2.jpeg` nella stessa cartella di `tesi.tex` (nella copia sul
  Desktop e in `tesi/` della repo). Se manca, copiarlo dalla repo.
- **Compilazione.** `cd <dir> && latexmk -pdf -shell-escape <nome>.tex`.
  Compilare solo il documento modificato per risparmiare tempo. Su macOS
  `setsid` NON esiste: per lanciare in background usare
  `(nohup latexmk -pdf -shell-escape -interaction=nonstopmode <nome>.tex > /tmp/<nome>.log 2>&1 < /dev/null &)`.
- **Bozza.** `altro/bozza.tex` (+ `altro/bozza.pdf`) è la versione bozza storica:
  numerazione ed equazioni diverse. Non serve a compilare `tesi_finale.pdf`
  (che usa solo `tesi.tex`); è in `altro/` come riferimento.
- **Niente changelog nel README (13/09/2026).** Il README non contiene più la
  cronistoria degli interventi ("Ultimo intervento" e "Intervento precedente"):
  le 245 voci sono state rimosse su richiesta. **Non riscriverle** e non
  aggiungere nuove voci di changelog: la storia delle modifiche resta nei
  messaggi di commit di git (`git log`, già dettagliati). Qui vanno tenute solo
  le note operative utili alle sessioni successive.

## Riferimento

R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu, *Sample size selection in
optimization methods for machine learning*, Mathematical Programming, 2012.
