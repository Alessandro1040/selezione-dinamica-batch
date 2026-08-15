# Selezione Dinamica della Dimensione del Campione

Tesi e materiale per *"Selezione Dinamica della Dimensione del Campione in
Metodi di Ottimizzazione per il Machine Learning"* — Corso di Laurea in
Scienze Matematiche per l'Intelligenza Artificiale, Sapienza Università di
Roma (A.A. 2025–2026).

## Contenuto del repository

```
.
├── README.md                      questo file
├── visualizzazione.html           applicazione web interattiva (Pyodide + Plotly)
├── simulazione_batch.py           simulazione autonoma della Figura 5.3 (n_k vs k)
├── figure_sim/                    figure generate da simulazione_batch.py
└── tesi/                          sorgenti LaTeX e PDF della tesi
    ├── main.tex                   documento principale
    ├── main.pdf                   PDF compilato
    ├── contenuti/                 frammenti LaTeX dei capitoli (intro, lavori, sezione6, ...)
    ├── figure/                    figure usate nella tesi (PDF/PNG)
    ├── sim_exp.py                 suite completa di esperimenti (rigenera tabelle e figure)
    └── ... (script di supporto, frammenti di tabelle)
```

## Applicazione web interattiva

Apri `visualizzazione.html` in un browser moderno (Chrome, Firefox, Safari,
Edge). L'app esegue Python nel browser tramite **Pyodide** e consente di:
- scegliere/modificare la funzione obiettivo (preset 1D/2D o codice custom),
- eseguire i tre algoritmi: Dynamic GD, Newton-CG, Newton-CG $L_1$,
- osservare il percorso su Plotly, la dimensione del batch ($n_k$ vs $a^k$)
  e l'analisi di convergenza.

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

Per compilare:

```bash
cd tesi
latexmk -pdf -shell-escape main.tex    # serve pygments per i listati minted
```

## Riferimento

R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu, *Sample size selection in
optimization methods for machine learning*, Mathematical Programming, 2012.
