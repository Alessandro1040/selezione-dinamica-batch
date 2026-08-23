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
├── tesi/                          SOLO ciò che serve a compilare tesi_finale.pdf
│   ├── tesi.tex                   documento di lavoro (article, con copertina)
│   ├── appendice_riuso.tex        Appendice E: riuso del mini-batch (inclusa da tesi.tex)
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

## Note operative e stato corrente (23/08/2026)

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
  il documento a partire da `altro/contenuti/`. **Unica eccezione (21/08/2026):**
  l'Appendice E (riuso del mini-batch) vive nel file `tesi/appendice_riuso.tex`
  (contenuto puro, incluso con `\input{appendice_riuso}` SOLO nel documento
  autonomo `tesi/appendice_riuso_estratto.tex`); in `tesi.tex` c'è una sezione
  segnaposto `(PLACEHOLDER)`. Se si modifica `appendice_riuso.tex` va
  ricompilato solo `appendice_riuso_estratto.pdf` (non più
  `tesi.pdf`/`tesi_finale.pdf`).
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
- **Ultimo intervento (23/08/2026).** **App web: teoria dinamica aggiornata —
  `M_H` negli pseudocodici di Newton-CG e Newton-CG $L_1$, e definita la
  variabile "servono dati nuovi".** In `visualizzazione.html` (file solo nella
  repo, non presente nella copia Desktop: modificato direttamente qui):
  (1) gli pseudocodici di riuso di Newton-CG e Newton-L1 ora includono `M_H`
  (limite dei riusi consecutivi di $H_k$) e il modo Hessiana *Legato a $S_k$*
  (default, $M_H$ inattivo: $H_k$ ricampionato insieme a $S_k$) /
  *Indipendente* ($H_k$ ricampionato dopo $M_H$ riusi; $M_H=\infty$: solo
  quando la CCV su $S_k$ è violata), con flag e contatori separati
  `da_ricampionare_S`/`da_ricampionare_H` e $j_{\mathcal S}$/$j_{\mathcal H}$
  (la rinomina risolve anche il conflitto del contatore di riuso $j$ col
  contatore interno del CG); estese le note "Mini-batch riusato" dei due
  Newton. (2) In TUTTI gli pseudocodici di riuso (GD, BB-CCV, Newton-CG,
  Newton-L1) la frase-variabile "servono dati nuovi" (mai definita) è
  sostituita dal booleano `da_ricampionare`, definito e inizializzato a `vero`
  prima del loop, azzerato a `falso` subito dopo il ricampionamento (prima
  mancava il reset: lo pseudocodice ricampionava a ogni iterazione) e
  riattivato a `vero` su CCV violata o a limite $M$/$M_H$ raggiunto — coerente
  con `need_resample`/`used` del codice generato. In BB aggiunta anche la
  definizione di $\widehat{\mathcal V}_k$ (usata ma mai definita). Validazione:
  i 4 blocchi pseudocodice modificati compilano con pdflatex (0 errori),
  `\[`/`\]` bilanciati (56/56). Nessun PDF coinvolto (l'app non è inclusa in
  `tesi.tex`).

- **Ultimo intervento (22/08/2026).** **Appendice E: aggiunta la colonna per il
  riuso dell'Hessiana indipendente da $S_k$ ($M_H=\infty$) e CORRETTA
  un'inconsistenza delle tabelle Newton.** (1) Scoperto che le tabelle
  E.1--E.16 erano state generate con `Sottocampionamento Hessiana` = "Intero
  dataset" (`hkSubset=False`, NON il default dell'app): le colonne base dei
  metodi di Newton non coincidevano con le Tabelle 6.3--6.5 della tesi (che
  usano `hkSubset=True` = Subset, validate da `riproduci_tabelle.py`).
  Rigenerate le 8 tabelle Newton (E.2, E.3, E.6, E.7, E.10, E.11, E.14, E.15)
  con `hkSubset=True` (default app): ora le colonne base coincidono
  esattamente con le Tabelle 6.3--6.5 (worst|diff|=0.00). GD e BB non usano
  l'Hessiana: invariati. (2) Aggiunta a queste 8 tabelle la colonna
  "H ind. $M_H=\infty$" (modalità *Indipendente da $S_k$* dell'app, batch
  $M=10$): a parità di riuso del batch l'Hessiana resta in uso finché la CCV è
  soddisfatta. Risultato: migliora su ben condizionato (Newton-CG) e termine
  incrociato (Newton-CG $L_1$), peggiora sui mal condizionati di Newton-CG.
  (3) Aggiornate le righe Newton della sintesi E.17 e della robustezza E.18
  (5 seed, ricalcolate con `hkSubset=True`). (4) Prosa aggiornata (E.1:
  descrizione modalità *Legato a $S_k$*/*Indipendente da $S_k$* e parametro
  $M_H$; E.2: valori corretti; E.3: setup con le colonne e i parametri
  dell'app; E.4/E.5: risultati) e didascalie delle 8 tabelle. (5) Nuovo script
  riproducibile `altro/script/gen_tabelle_riuso.py` (preset e algoritmi del
  codice generato dall'app; `--diagnosi` verifica la riproduzione con
  `subset=False`, `--data` produce i dati, `--tex` riscrive le tabelle).
  Ricompilato `appendice_riuso_estratto.pdf`: 21 -> 22 pp, 0 errori, 0
  overfull. I `tesi.pdf`/`tesi_finale.pdf` NON sono coinvolti (in `tesi.tex`
  l'Appendice E è un PLACEHOLDER).

- **Ultimo intervento (22/08/2026).** **App web: riuso dell'Hessiana
  indipendente da $S_k$ con iperparametro $M_H$.** Nuova opzione "Riuso
  Hessiana (H_k)" per Newton-CG e Newton-L1 (visibile solo con "Iterazioni
  consecutive sullo stesso mini-batch" attivo): modalità **Legato a S_k**
  (default teoria: $H_k$ segue $S_k$ finché la CCV è soddisfatta) e
  **Indipendente da S_k** ($H_k$ riusato per $M_H$ iterazioni consecutive; se
  $M_H=\infty$, $H_k$ resta legato alla CCV su $S_k$ ma senza il contatore di
  $S_k$). Aggiunti i controlli HTML `hessianReuseMode`/`mhMode`/`mhValue`, i
  campi `reuseHessian`/`maxHessianReuse` in `getAlgoOptions()` e la logica in
  `generateNewtonCG`/`generateNewtonL1` con contatori separati
  `used_S`/`used_H` e `need_resample_S`/`need_resample_H` (la CCV violata
  ricampiona sia $S_k$ sia $H_k$; `maxHessianReuse` ignorato in modalità
  legata). Validato con jsc + py_compile: 16/16 codici generati con sintassi
  OK; regressione: output dei casi base e GD/BB byte-identici a HEAD.

- **Ultimo intervento (22/08/2026).** **Appendice E: prosa ristrutturata
  ("Meccanismo: vantaggi e limiti del riuso"), rimossi grassetti e corsivi,
  nuova figura con i punti blu delle violazioni CCV.** In
  `tesi/appendice_riuso.tex` la prosa è stata riscritta: le sottosezioni "Un
  esempio geometrico per capire il meccanismo" e "Perché il riuso può
  migliorare/peggiorare" sono confluite nella nuova "Meccanismo: vantaggi e
  limiti del riuso" (label `app:riuso-meccanismo`, E.2; Descrizione E.1,
  Setup E.3, Risultati numerici E.4, Sintesi E.5) con le formule esplicite di
  $g_k=\nabla J_{\mathcal{S}_k}(w_k)$ e $w^*_{\mathcal{S}}$. Rimossi TUTTI i
  `\textbf` e gli `\emph` dalla prosa. Figura `download.png` aggiornata (la
  nuova versione da `~/Downloads/download (1).png`: **puntini blu sulle
  iterazioni in cui la condizione CCV viene violata**); caption estesa di
  conseguenza. **Tabelle E.1--E.18 INVARIATE**. `appendice_riuso_estratto.pdf`
  ricompilato (**21 pp** da 24, 0 errori, 0 undefined, 0 overfull, 1 underfull
  preesistente). `tesi.tex` non toccata (l'appendice non è più inclusa):
  nessuna ricompilazione di tesi.pdf/tesi_finale.pdf. Sincronizzati nella repo
  `tesi/appendice_riuso.tex` + `tesi/appendice_riuso_estratto.pdf` +
  `tesi/download.png` (md5 verificati).

- **Ultimo intervento (22/08/2026).** **Appendice E: prima parte (prosa)
  riscritta con l'esempio geometrico e la figura delle traiettorie.**
  In `tesi/appendice_riuso.tex` la prosa iniziale (righe 1--264) è stata
  sostituita con la versione nuova (da `~/Downloads/appendice_riuso.tex`):
  aggiunta la sottosezione "Un esempio geometrico per capire il meccanismo"
  (E.2) con la quadratica $J(w)=\kappa w_1^2+w_2^2$ ($\kappa=1$ ben
  condizionato, $\kappa=20$ mal condizionato), i paragrafi "Il gradiente vero
  vs. il gradiente del batch" e "Perché il condizionamento cambia tutto", e la
  figura `download.png` (label `fig:riuso_traiettorie`, Fig. E.1: traiettorie
  di Dynamic GD con ricampionamento vs riuso del batch sui due problemi).
  Sottosezioni successive rinumerate: Perché il riuso può migliorare (E.3),
  Perché il riuso può peggiorare (E.4), Setup sperimentale (E.5), Risultati
  numerici (E.6), Sintesi (E.7). **Tabelle E.1--E.18 INVARIATE**.
  `appendice_riuso_estratto.pdf` ricompilato (**24 pp** da 20, 0 errori, 0
  undefined, 0 overfull, 1 underfull preesistente); nuova figura
  `tesi/download.png`. `tesi.tex` non toccata (l'appendice non è più inclusa):
  nessuna ricompilazione di tesi.pdf/tesi_finale.pdf. Sincronizzati nella repo
  `tesi/appendice_riuso.tex` + `tesi/appendice_riuso_estratto.pdf` +
  `tesi/download.png` (md5 verificati).

- **Ultimo intervento (21/08/2026).** **Appendice E: riscritta la prima parte
  (prosa) e aggiunta la sottosezione "Perché il riuso può peggiorare i
  risultati".** In `tesi/appendice_riuso.tex` testo introduttivo e sottosezioni
  riscritti e riordinati: Descrizione (E.1), Perché il riuso può migliorare
  (E.2), **Perché il riuso può peggiorare (E.3, nuova)**, Setup sperimentale
  (E.4), Risultati numerici (E.5, ora con paragrafi per metodo: primo ordine
  ben/mal condizionato, Newton-CG, Newton-CG $L_1$), Sintesi (E.6, con i
  paragrafi "Quando il riuso aiuta", "Quando il riuso non aiuta" e
  "Osservazione sul costo computazionale"). Tabelle E.1--E.18 invariate.
  `appendice_riuso_estratto.pdf` ricompilato (**20 pp**, 0 errori, 0
  undefined, 0 overfull); sincronizzati nella repo `tesi/appendice_riuso.tex`
  + `tesi/appendice_riuso_estratto.pdf` (md5 verificati). `tesi.tex` non
  toccata (l'appendice non è più inclusa): nessuna ricompilazione di
  tesi.pdf/tesi_finale.pdf.
- **Ultimo intervento (21/08/2026).** **Tesi: Appendice E estratta in un
  documento autonomo e ridotta a segnaposto.** In `tesi/tesi.tex` il comando
  `\input{appendice_riuso}` è stato sostituito dalla sezione segnaposto
  `\section{Riuso del mini-batch: iterazioni consecutive sullo stesso
  campione}\label{app:riuso}` seguita da `(PLACEHOLDER)` (titolo e label
  conservati, così l'Appendice E resta nell'indice). Il contenuto integrale
  dell'appendice (incluso via `\input{appendice_riuso}`) vive ora nel nuovo
  documento autonomo `tesi/appendice_riuso_estratto.tex` (preambolo minimale +
  `\appendix` con `\setcounter{section}{4}` per restare numerata **E**, tabelle
  E.1--E.18 invariate; riferimenti esterni risolti: `sec:algoritmi` → 5,
  `sec:setup` → 6.1, `sec:risultati` → 6.3, `sec:visualizzazione` → 6.4),
  compilato in `tesi/appendice_riuso_estratto.pdf` (**20 pp**). Ricompilati
  `tesi.pdf` (**103 → 83 pp**) e `tesi_finale.pdf` (**83 pp**), 0 errori, 0
  riferimenti undefined. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf/appendice_riuso_estratto.tex/
  appendice_riuso_estratto.pdf (md5 verificati). `appendice_riuso.tex`
  invariata; `bozza.tex` non toccata.
- **Ultimo intervento (21/08/2026).** **Tesi: sottosezioni 6.5 e 6.6 estratte in
  un documento separato e ridotte a segnaposto.** In `tesi/tesi.tex` il testo
  delle sottosezioni 6.5 ("Validazione su un benchmark musicale: riconoscimento
  della nota con NSynth") e 6.6 ("Estensione oltre le ipotesi: implementazione
  vettorizzata per la rete neurale") è stato sostituito con `(PLACEHOLDER)`,
  mantenendo le `\label` (`sec:nsynth`, `sec:nsynth-net`) e i titoli abbreviati
  in modo che entrino in una riga: **"Riconoscimento Note"** (6.5) e
  **"Riconoscimento Strumenti"** (6.6). Il contenuto integrale delle due
  sottosezioni è stato spostato nel nuovo documento autonomo
  `tesi/sezioni_65_66.tex` (preambolo minimale + le sole due sottosezioni,
  figure da `figure_nsynth_nota/` e `figure_nsynth_net/`; riferimenti esterni
  risolti: `sec:l1` → 5.3 e `sec:setup` → 6.1 via label segnaposto, citazione
  `engel2017` con bibliografia minima), compilato in `tesi/sezioni_65_66.pdf`
  (**14 pp**). Ricompilati `tesi.pdf` (**117 → 103 pp**) e `tesi_finale.pdf`
  (**103 pp**), 0 errori, 0 riferimenti undefined. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf/sezioni_65_66.tex/sezioni_65_66.pdf (md5
  verificati). `bozza.tex` non toccata.
- **Ultimo intervento (21/08/2026).** **Tesi: nuova Appendice E "Riuso del
  mini-batch: iterazioni consecutive sullo stesso campione".** Nuovo file
  `tesi/appendice_riuso.tex` (incluso da `tesi.tex` con `\input` prima della
  bibliografia) che analizza l'opzione "Iterazioni consecutive" dell'app web
  introdotta nel commit precedente. Contiene: descrizione della variante
  (riuso degli indici del campione, CCV come test di esaurimento, M=1 coincide
  con la base per GD/BB), spiegazione del meccanismo (passi coerenti su
  sottoproblema fisso → meno rumore, CCV come salvaguardia), setup sperimentale
  (4 preset × 4 algoritmi × M∈{∞,10,5,3,2,1}, seed 42 e 5 seed per la
  robustezza), 16 tabelle per-iterazione (E.1–E.16, errore ||w_k−w_*|| a ogni
  k=0..30, base vs M=∞/10/5/2, formato identico alle Tabelle 6.1–6.3) e 2
  tabelle di sintesi: E.17 (errore finale e_30 per tutte le M, con ▲/▼/= rispetto
  alla base) ed E.18 (robustezza su 5 seed, M=∞ e M=10). Dati generati da
  replica esatta del codice dell'app (`/tmp/verifica_reuse2.py` +
  `/tmp/gen_tabelle2.py`), valori base coincidenti con le Tabelle 6.1–6.3.
  **Risultato dell'analisi**: il riuso aiuta sistematicamente GD/BB sui problemi
  ben condizionati (5/5 seed su κ≈1.1) e spesso su quello incrociato, ma non è
  una garanzia universale: peggiora GD/BB sul mal condizionato (3/5 seed su
  κ≈20 con M=∞) e Newton-CG quasi sempre (0/5 con M=10 su κ≈20, κ≈100,
  incrociato). Il testo spiega i "perché" (bias del sottoproblema non catturato
  dalla CCV; Hessiana riusata obsoleta; line search solo sulla loss del batch).
  Ricompilati `tesi.pdf` (96 → 117 pp) e `tesi_finale.pdf` (117 pp), 0 errori, 0
  overfull nell'appendice. Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf/appendice_riuso.tex (md5 verificati). `bozza.tex` non toccata.

- **Ultimo intervento (21/08/2026).** **App web: opzione "Iterazioni consecutive
  sullo stesso mini-batch" per tutti i metodi.** In `visualizzazione.html` (solo
  repo, la copia Desktop non contiene l'app) sono stati aggiunti due controlli
  globali nel blocco "Codice Python dell'algoritmo": (1) lo switch **"Iterazioni
  consecutive sullo stesso mini-batch"** (Attivo / Disattivato, default
  **Disattivato**); (2) l'iperparametro **"Max iterazioni consecutive per
  mini-batch (k)"** con opzione **"Illimitato"** (default) o un valore finito
  (1–200). Quando la funzione è Attiva, il codice Python generato per **tutti e 4
  i metodi** (GD, Newton-CG, Newton-CG-L1, BB-CCV) viene ristrutturato: lo stesso
  mini-batch (S_k, e H_k per i metodi di Newton) viene riusato per più iterazioni
  consecutive; a ogni iterazione interna la **CCV è valutata sul campione in uso
  al punto corrente** (gratis: usa gli stessi gradienti per-esempio del passo);
  se la CCV fallisce il batch è "esaurito" → n_k cresce con la regola della tesi
  e si ricampiona; se k è finito si ricampiona comunque dopo k iterazioni (stessa
  n_k). Con **k = 1** si ricampiona a ogni iterazione: per GD e BB il
  comportamento è **identico** all'algoritmo attuale (verificato: traiettorie
  byte-identiche a parità di seed); per Newton-CG e Newton-L1 la sola differenza
  è che la CCV è valutata sul batch in uso invece che su un nuovo campione. In
  più, quando l'opzione viene modificata si aggiorna **solo la teoria dinamica**
  (pseudocodici e formule che descrivono l'algoritmo) di tutti i metodi: ogni
  sezione teoria mostra la variante "mini-batch riusato" (blocco `.pseudo-reuse`
  + nota). Validato: 4 metodi × 8 varianti di opzioni generate e compilate
  (32/32 sintassi OK); gli output con opzione Disattivata sono **byte-identici**
  a quelli precedenti (confronto con HEAD); test funzionale su problema
  quadratico sintetico (N=200) senza errori, con riuso che riduce la crescita del
  batch (es. GD n_max 22→13, Newton-L1 41→15 con k illimitato). Nessun sorgente
  LaTeX toccato (nessun PDF da ricompilare). Commit e push della repo.

- **Ultimo intervento (21/08/2026).** **BB-CCV: la trattazione teorica si
  sposta dall'Appendice E alla Sezione 5.4.** In `tesi/tesi.tex` l'intera
  appendice "Schema dell'algoritmo BB-CCV" (passo di Barzilai--Borwein +
  schema a blocchi) è stata spostata alla fine del Capitolo 5 "Algoritmi
  Proposti" come nuova sottosezione **5.4 "Metodo BB-CCV (Barzilai--Borwein
  con campionamento dinamico)"**, con le due sotto-sottosezioni 5.4.1 "Il
  metodo di Barzilai--Borwein" e 5.4.2 "Schema dell'algoritmo BB-CCV".
  Adattamenti: le equazioni E.1–E.5 sono rinumerate (5.52)–(5.56) (continuano
  la numerazione manuale del capitolo, che arrivava a 5.51); il paragrafo
  introduttivo "Questa appendice..." è riscritto come incipit della
  sottosezione; la figura a blocchi diventa la **Figura 5.6** (caption
  aggiornata: `Sezione~\ref{sec:esperimenti}` al posto di "Sezione~6"); la
  tabella di confronto diventa la **Tabella 5.3**. Il listato Python resta in
  Appendice **B.4**, rinominata "**Codice BB-CCV (Barzilai--Borwein con
  campionamento dinamico)**" per evitare il doppione col titolo 5.4; i 4
  riferimenti `Appendice~\ref{app:bbccv}` diventano `Sezione~\ref{sec:bbccv}`
  (Sez. 6.1, 6.3 e Appendice B.4). Le appendici ora sono A–D (la bibliografia
  segue l'Appendice D). Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**97 pp
  invariate**, 0 errori, 0 riferimenti indefiniti, 14 overfull preesistenti);
  sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non allineata.

- **Ultimo intervento (21/08/2026).** **BB-CCV esteso allo script di
  simulazione (`simulazione_batch.py`).** Lo script che genera le figure di
  `figure_sim/` (n_k vs k e convergenza) eseguiva solo Dynamic GD: ora esegue
  anche **BB-CCV** (`bb_dynamic_gd`, copia del listato **Appendice B.4** di
  `tesi.tex`: passo Barzilai-Borwein con safeguard `clip(alpha/20, 5*alpha)`,
  CCV sul batch e line search di Armijo) e produce i plot con il **confronto
  tra i due metodi** (curve blu = Dynamic GD, verdi = BB-CCV, con i fit a^k).
  Su `quad_well` (seed 42, default app): Dynamic GD 31 iterazioni, BB-CCV 11
  iterazioni, batch finale 200 per entrambi. Rigenerati
  `figure_sim/batch_size.{pdf,png}` e `figure_sim/convergenza.{pdf,png}`;
  allineato anche `tesi/figure_sim/batch_size.pdf` nella copia Desktop (md5
  verificati). `tesi.tex` non toccato (usa `batch_size_app.png` dell'app, non
  gli output dello script) → nessun PDF da ricompilare. Commit e push della
  repo.
- **Ultimo intervento (21/08/2026).** **BB-CCV riaggiunto all'app web
  (`visualizzazione.html`).** Nel file `visualizzazione.html` (solo repo, la
  copia Desktop non contiene l'app) è stato ripristinato il metodo
  **Barzilai-Borwein con Campionamento Dinamico (BB-CCV)** tra gli algoritmi
  possibili. La versione con BB era presente nelle copie storiche pre-9/08 in
  Downloads (`deepseek_html_20260802_1c3ec7.html` del 2/08,
  `interactive_optimization_fixed.html` del 4/08, `prova.html` del 7/08,
  tutte con `option value="bb"` e funzione `generateBB`); è stata rimossa nella
  riscrittura successiva (9/08, "Visualizzazione (Tesi).html") e nel file
  attuale restava solo l'etichetta "Line search — GD, Newton-CG, BB". Copiata
  **esattamente** (diff byte-identico) da `interactive_optimization_fixed.html`:
  (1) `<option value="bb">` nel selettore algoritmo; (2) sezione teoria
  `theory-bb` (passo BB $\alpha_k^{\mathrm{BB}}$, safeguard
  $\mathrm{clip}(\cdot,\alpha/20,5\alpha)$, condizione di curvatura,
  aggiornamento CCV identico al GD, line search); (3) funzione
  `generateBB(opts)` (passo BB + CCV dinamico, on/off come gli altri metodi,
  line search Wolfe/Armijo); (4) `case 'bb'` nello switch `generateAlgoCode`;
  (5) parametri GD visibili anche per BB (`updateParamsVisibility`); (6) hint
  in `loadAlgoPreset`. Validato: il codice Python generato da `generateBB`
  (CCV attivo + Wolfe) converge su `quad_well` in 13 iterazioni con batch che
  cresce fino a N=200. Nessun sorgente LaTeX toccato (nessun PDF da
  ricompilare). Commit e push della repo.
- **Ultimo intervento (20/08/2026).** **"Retropropagazione" → "backpropagation";
  spiegato meglio il termine "vettorizzata".** In `tesi/tesi.tex` tutte le
  occorrenze di *retropropagazione* (5: testo del forward, un nodo della figura
  del forward, il paragrafo "Gradiente del batch", un commento e la caption di
  `grad_batch`) sono state sostituite con *backpropagation*; stesso cambio in
  `altro/tesi_sapthesis.tex` (3: testo, nodo e caption). Inoltre il termine
  *vettorizzata* è ora spiegato esplicitamente: (1) nel paragrafo "Gradiente sul
  training set completo" si definisce cosa significa — formule applicate a
  matrici che raccolgono tutto il dataset in un solo passaggio (es.
  $Z=X_{\mathrm{aug}}W^{\top}$) anziché un ciclo per-esempio — e il perché
  (kernel di algebra lineare ottimizzati di NumPy); (2) nella sottosezione sulla
  rete neurale le versioni `batch` sono dette *vettorizzate* con rinvio a
  "Sezione 6.5". Ricompilati `tesi.pdf`, `tesi_finale.pdf` (**97 pp**, +1 dal
  testo aggiunto) e `tesi_sapthesis.pdf`; copia Desktop e repo sincronizzate
  (md5 verificati).
- **Ultimo intervento (20/08/2026).** **Notebook nota aggiornati (solo repo,
  copia Desktop non toccata).** `tesi/nsynth/nsynth_nota_riproduzione.ipynb`:
  (1) la tabella stampata ora riporta il **batch finale reale** per ogni metodo
  (`{r['batch']:,}`; prima era hardcodato `12\,678`, che mostrava 12\,678 anche
  per Dynamic GD il cui batch vero è 5\,297); (2) l'esempio casuale ora usa
  `random.randrange(len(names_test))` quando `nota = None` (prima restava
  sempre l'indice 42); (3) la cella predizioni mostra l'**incertezza del
  modello**: softmax top-3 con probabilità + entropia normalizzata
  (es. `Newton-CG: A (20.2%) ✘ | 2°: A# (15.7%) | 3°: B (12.2%) | incertezza
  93%`); rimosse le celle di prova temporanee (verifica audio/f0). Anche
  `tesi/nsynth/nsynth_nota_test.ipynb` ora mostra la colonna "Inc."
  (entropia normalizzata) nelle predizioni (celle 5 e 8). Aggiunti
  `colab_risorse/pesi_nota.npz` (pesi dei 4 modelli + mu/sd) e
  `colab_risorse/features_nota.npz` (Xte/Yte/names del test set), scaricati
  dalla run Colab. Aggiunte anche le features già estratte della rete
  (`colab_risorse/features_opt_net_test.npz`, split test 4 096 clip, e
  `features_opt_net_valid.npz`, split valid 12 678 clip) per saltare
  l'estrazione librosa (~20-60 min) nei Colab della rete. Nessun sorgente
  LaTeX toccato: documento invariato.
- **Ultimo intervento (20/08/2026).** **Filtro anti-clip-anomale e incertezza
  anche per la rete.** In `tesi/nsynth/nsynth_nota_riproduzione.ipynb` la cella
  dell'esempio casuale ora evita le clip rumorose/difettose di NSynth: sceglie
  a caso finché il modello più accurato (Newton-CG) non ha entropia normalizzata
  < 0.5 (max 30 tentativi); con `nota = "C"` (o altra nota) il filtro non si
  applica e si scelgono clip di quella nota. In
  `tesi/nsynth/nsynth_nota_riproduzione` (cella 21) e
  `tesi/nsynth/nsynth_net_test.ipynb` (celle 5 e 8) le predizioni mostrano ora
  l'incertezza (entropia normalizzata, colonna "Inc." / "incertezza %").
  Nessun sorgente LaTeX toccato: documento invariato.
- **Ultimo intervento (20/08/2026).** **Corretti 3 errori nei dati della Sez. 6.5
  (nota) e 6.5.1 (rete neurale).** (1)~Caption Tab. 6.5: rimosso Newton-CG da
  "raggiungono il criterio di arresto $\|\nabla J\|_2 < 10^{-6}$": dal
  `results.json` Newton-CG termina a `iter=300` (usa tutto il budget) con
  gnorm finale $2.8\times10^{-6} > 10^{-6}$; solo Newton-CG-$L_1$ (iter=128,
  gnorm $8.5\times10^{-7}$) lo raggiunge. (2)~Testo Sez. 6.5: "azzerando il
  51% dei coefficienti (146 su 300)" era ambiguo/errato: 146 è il numero di
  coefficienti **non nulli** (nnz), gli azzerati sono 154 (51.3%); corretto in
  "(146 restano non nulli su 300)". (3)~Tab. 6.6 (rete): tempi allineati al
  `colab_risorse/figure/net/results_net.json` (file dati ufficiale, identico
  in repo e in `Documents/placeholder/nsynth_net_locale/`): Dynamic GD
  175.8 $\to$ **148.8** s, Newton-CG 269.7 $\to$ **210.2** s, Newton-CG-$L_1$
  263.3 $\to$ **196.9** s, BB-CCV 246.7 $\to$ **216.9** s (i valori 175.8/... 
  della run precedente non corrispondevano al file dati). Verificati tutti gli
  altri valori delle Tabb. 6.5 e 6.6 contro i `results.json` (accuratezze,
  norme del gradiente, batch, nnz): coerenti. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (97 pp invariati); `bozza.tex` non toccata. Sincronizzati
  nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
- **Ultimo intervento (20/08/2026).** **Sez. 6 "Osservazioni comparative":
  riscritto il paragrafo con le giustificazioni dei risultati, interamente su
  una sola pagina.** Il paragrafo dopo le Tabelle 6.1--6.3 sforava di ~4 parole
  ("BB-CCV, permette di progredire.") sulla pagina successiva. Riscritto in
  forma più compatta e con i "perché" dei risultati osservati: (i)~sul ben
  condizionato ($\kappa\approx1.1$) la curvatura quasi isotropa rende un passo
  scalare adeguato in ogni direzione, perciò Dynamic GD/Newton-CG-$L_1$/BB-CCV
  sono i più veloci e Newton-CG il più lento (paga la soluzione del sistema di
  Newton senza guadagnarne); (ii)~sul molto mal condizionato ($\kappa\approx
  100$) il numero di condizionamento entra nel tasso dei metodi del primo
  ordine (Sez. 4), quindi Dynamic GD/Newton-CG-$L_1$/Newton-CG restano sopra
  $10^{-1}$, mentre BB-CCV raggiunge la precisione macchina in 11 iterazioni
  perché il passo di Barzilai--Borwein stima la curvatura dai soli gradienti
  (App. E) e si adatta all'anisotropia; (iii)~con il termine incrociato BB-CCV
  è di nuovo il migliore. Rimossi i dettagli numerici già presenti nelle
  tabelle. Ora il paragrafo entra tutto nella pagina in cui inizia (niente
  riga orfana in cima alla successiva: lì inizia la Sez. 6.4). Documento
  invariato a **97 pp** (`tesi.pdf`, `tesi_finale.pdf`). `bozza.tex` non
  toccata. Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5
  verificati).
- **Ultimo intervento (20/08/2026).** **Corrette le Tabelle 6.1--6.3 dei
  risultati numerici (errore $\|w_k-w_*\|_2$ a ogni iterazione).** Aggiunto lo
  script di riproduzione fedele `altro/script/riproduci_tabelle.py` (codice
  esatto dell'app `visualizzazione.html` per GD/Newton-CG/Newton-CG-L1 con i
  default N=200, seed 42, w0=(2,-3), α=0.1, θ=0.5, n0=5, 30 iterazioni, R=0.2,
  maxcg=10, ν=0.1, σ=0.1, η=0.5, line search di Wolfe; BB-CCV dal codice
  `altro/script/bbccv.py`, line search di Armijo). Confronto riga-riga (5 cifre
  significative, differenze ~1e-16): le Tabb. 6.1 e 6.3 erano già identiche
  alla riproduzione; nella Tab. 6.2 (molto mal condizionato, κ≈100) le colonne
  ``Dynamic GD'' e ``Newton-CG $L_1$'' erano **scambiate** — corretto
  mantenendo l'ordine delle colonne coerente con le altre tabelle. Corretti
  anche i valori k=26..30 della colonna Newton-CG (errori da lettura OCR:
  2.6912, 2.6912, 2.2653, 2.0761, 2.0761 → 2.4956, 2.4956, 2.4956, 2.2864,
  2.0969) e l'ultima cifra di BB-CCV (1.1324e-14 → 1.0991e-14). Verificato che
  il valore ≈10^-14 di BB-CCV sul problema mal condizionato **non è un
  artefatto OCR**: si riproduce deterministicamente (il passo di
  Barzilai--Borwein salvaguardato + Armijo porta $w_{11}$ a
  $\|w-w_*\|≈1.1\times10^{-14}$, con il gradiente pieno sotto la tolleranza di
  arresto 1e-6). Ricompilati `tesi.pdf` e `tesi_finale.pdf` (97 pp, 0 errori,
  0 riferimenti indefiniti); sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non allineata.
- **Ultimo intervento (20/08/2026, follow-up).** **Caption Figura E.2 (schema
  BB-CCV): corretto ``~18 iterazioni medie'' in ``media ≈13''.** Il numero
  ``~18'' era stato scritto il 16/08 (commit `bfb7e53`, aggiunta dello schema
  BB-CCV) prima dell'allineamento delle tabelle all'app: non corrisponde agli
  esperimenti attuali. La riproduzione fedele dà BB-CCV a k=12, 11 e 16 sui
  tre problemi (media 13) contro le 30 del GD; la frase ora cita
  esplicitamente i tre esperimenti della Sezione~6. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (97 pp); sincronizzati in repo (md5 verificati).
  `bozza.tex` non allineata.
- **Ultimo intervento (20/08/2026).** **Notebook di test per il pitch: nuovo
  `tesi/nsynth/nsynth_nota_test.ipynb` + salvataggio dei pesi nel notebook di
  riproduzione.** Il notebook `nsynth_nota_riproduzione.ipynb` NON salvava i
  pesi addestrati (variabile `pesi` solo in memoria): aggiunta la cella
  ``4b. Salva i pesi...'' che salva e scarica `pesi_nota.npz` (pesi finali dei
  4 metodi + `mu`/`sd` della standardizzazione, chiavi `Dynamic_GD`,
  `Newton-CG`, `Newton-CG_L1`, `BB-CCV`) e `features_nota.npz` (Xte/Yte/nomi
  delle clip di test). Corretti anche i nomi dei file scaricati dalla cella
  opzionale finale (erano `nsynth_accuracy.*`, ora `nota_accuracy.*`/
  `nota_batch.*`, coerenti con le figure salvate). Nuovo
  **`tesi/nsynth/nsynth_nota_test.ipynb`** speculare a `nsynth_net_test.ipynb`:
  carica `pesi_nota.npz` via upload e permette di (1) testare su un
  **qualsiasi file .wav** (chroma 24-dim + standardizzazione, stessi parametri
  della Sez. 6.5), (2) ascoltare la clip, (3) calcolare l'accuratezza sul test
  set e (4) provare singole clip NSynth con la nota vera (e ascoltarle,
  scaricando una volta sola il tar di test ~350 MB). Pipeline validato con uno
  smoke test sintetico end-to-end (features chroma + standardizzazione +
  softmax: 4 note predette correttamente). Nessun sorgente LaTeX toccato.
- **Ultimo intervento (19/08/2026).** **Sez. 6.5: compattata la spiegazione sulla
  linearità del modello.** Il paragrafo "Il modello è una regressione logistica
  multinomiale…" (spiegazione di cosa significa "lineare" e perché la scelta è
  deliberata) passa da 20 a 15 righe: rimossi i giri di parole ("ogni componente
  cromatica contribuisce in modo indipendente", "trasforma i dodici punteggi",
  "dove la loss è non convessa e i milioni di parametri rendono impraticabile
  l'analisi" → "(loss non convessa, milioni di parametri)", l'enumerazione
  (i)/(ii) del confronto con la rete profonda). Contenuto invariato: formula
  $s_c(x)$, significato di "lineare", unica non-linearità = softmax, motivo
  (loss convessa e liscia → teoria applicabile), due semplificazioni vs rete
  profonda. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (97 pp, 0 errori, 0
  riferimenti indefiniti); sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non allineata.
- **Ultimo intervento (19/08/2026).** **Ripristinato font Computer Modern.** Dopo
  aver provato URW Gothic (scala 0.90, vedi nota sotto), su richiesta si torna
  al font originale: rimosso l'intero blocco `avant`/`\DeclareFontShape` da
  `tesi.tex` (la modifica era solo lì). Verificato che il layout è tornato
  **identico all'originale riga per riga** (confronto LCS: 6161/6161 righe,
  100%; 97 pp). Ricompilati `tesi.pdf` e `tesi_finale.pdf` (97 pp);
  sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati;
  `tesi.tex` e `tesi_finale.pdf` coincidono con lo stato pre-font).
  `bozza.tex` non allineata.
- **Ultimo intervento (19/08/2026).** **Font URW Gothic (Avant Garde) per tutto il
  documento, scala 0.90.** In `tesi/tesi.tex` aggiunto il blocco font
  `\usepackage{avant}` + `\renewcommand{\familydefault}{\sfdefault}` (testo
  completo in sans). URW Gothic è più largo di Computer Modern: misurate le
  righe identiche al layout CM per varie scale (sweep con confronto riga-riga
  su 6161 righe): 1.0→107 pp/21%, 0.95→105 pp/22%, **0.90→100 pp/73.5%**,
  0.85→97 pp/67%, 0.80→95 pp/41%, 0.75→95 pp/11%. Scelta la **scala 0.90**
  (miglior compromesso righe identiche, +3 pp). La scala si applica
  ridefinendo le forme del family `pag` via NFSS (`s * [0.90]`): `avant.sty`
  non supporta l'opzione `scaled`. Due trappole documentate nel sorgente:
  (1) le `\DeclareFontShape` NON vanno in una `\newcommand` (non registra le
  forme → "Font T1/pag/m/n/12 not found"); (2) serve `\input{t1pag.fd}` prima
  di ridefinire le forme, altrimenti `\DeclareFontFamily{T1}{pag}{}` marca la
  famiglia come definita e le forme `it`/`b`/`bx` (corsivo e grassetto)
  restano non definite ("Font shape 'T1/pag/m/it' undefined" → ricaduta su
  CM). Verificato nel PDF: `URWGothicL-Book`, `-Demi`, `-BookObli`; i font
  matematici restano Computer Modern. Nessun errore né riferimento
  indefinito; overfull 14→22 (conseguenza del reflow). Documento da 97 a
  **100 pp** (`tesi.pdf`, `tesi_finale.pdf`). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  allineata; frontespizio (sapthesis, documento separato) invariato.
- **Ultimo intervento (19/08/2026).** **Intestazioni e link interni in nero.**
  In `tesi/tesi.tex`: (1) rimossi i `\textcolor{coverblue}{...}` da `\lhead`
  (Alessandro Lo Curcio) e `\rhead` (Selezione Dinamica del Campione in ML):
  le intestazioni in alto a sinistra/destra sono ora nere; (2) `\hypersetup`
  passa da `linkcolor=coverblue` a `linkcolor=black`: le voci dell'indice e
  tutti i riferimenti interni nel testo (Sezione/Figura/Tabella) sono ora
  neri; citazioni e URL restano `coverblue`. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (97 pp invariati); sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  allineata.
- **Ultimo intervento (19/08/2026).** **Ringraziamenti centrati e in corsivo.**
  In `tesi/tesi.tex` la frase di ringraziamento è ora centrata e in `\emph`
  (corsivo), sotto il titolo, in parallelo con la pagina dell'Abstract (prima
  era `\noindent` allineata a sinistra in tondo). Scelta di stile: centratura
  e corsivo per una pagina dedicata di solo testo (allineare a destra avrebbe
  richiamato una firma). Ricompilati `tesi.pdf` e `tesi_finale.pdf` (97 pp
  invariati); sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5
  verificati). `bozza.tex` non allineata.
- **Ultimo intervento (19/08/2026).** **Tabella 5.2: da float `[h]` a `[H]`.** La
  tabella `tab:5_1` (confronto dei tre metodi) era dichiarata `\begin{table}[h]`
  e LaTeX la spostava in cima alla pagina successiva, compilando prima il
  titolo "6 Esperimenti e Visualizzazione Interattiva" e l'inizio del
  paragrafo introduttivo: nel PDF la caption terminava con "...m il numero di
  variabili." seguita direttamente dalla seconda metà del paragrafo ("e i
  risultati numerici ottenuti..."), come se mancasse una riga. Passata a
  `[H]` (posizionamento rigoroso, coerente con le altre figure del documento):
  la tabella resta alla fine del Cap. 5 e la Sez. 6 segue intatta. Corretto
  anche il commento interno ("Tabella 5.1" → "Tabella 5.2"). Ricompilati
  `tesi.pdf` e `tesi_finale.pdf` (97 pp invariati); sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non allineata.
- **Ultimo intervento (19/08/2026).** **Appendice Istruzioni: riscritta la
  sottosezione "Esecuzione degli esperimenti numerici".** Prima affermava che
  gli esperimenti del Capitolo 6 si riproducono eseguendo `sim_exp.py`; non è
  vero (lo script usa problemi e versioni diverse, senza `newton_l1`/BB-CCV,
  e non accetta parametri da CLI). Ora il testo spiega che gli esperimenti del
  Cap. 6 sono quelli dell'app `visualizzazione.html` (pannello Analisi,
  riproducibili a parità di seed) e che `sim_exp.py` è solo un esempio
  standalone (Dynamic GD e Newton-CG) su problemi test sintetici, con figure
  PDF/PNG e tabelle LaTeX. Conservata la frase (vera) sull'esempio di
  regressione lineare nella sottosezione successiva. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (97 pp invariati); sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non allineata.
- **Ultimo intervento (19/08/2026).** **Tabella nota: corretto "Batch finale" di
  Dynamic GD (12 678 → 5 297) e aggiunta la spiegazione del plateau in Sez.
  6.5.** I dati salvati (`colab_risorse/figure/nota/curves.npz` e
  `results.json`) mostrano che nell'esperimento nota il solo Dynamic GD porta
  il batch a 5 297 (non a 12 678 = N): la sua norma del gradiente ristagna a
  ~6.8e-2 (mai sotto la tolleranza di arresto) e la CCV si stabilizza a
  n_k ≈ 5 297. BB-CCV invece raggiunge N=12 678 più velocemente di tutti
  (iterazione 23); il plateau a ~5000 nella figura è la linea blu di Dynamic
  GD, non BB-CCV. Corretta la cella della Tab. (tab:nsynth_nota) in `tesi.tex`
  e aggiunta una frase esplicativa nella prosa della Sez. 6.5. Ricompilati
  `tesi.pdf` e `tesi_finale.pdf` (pagine invariate); sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non allineata.
- **Ultimo intervento (19/08/2026).** **Riorganizzata la struttura della repo:
  cartelle `colab_risorse/` e `altro/`; `tesi/` ridotta al minimo per
  compilare.** `tesi/` ora contiene solo ciò che serve a compilare
  `tesi_finale.pdf` (documento, frontespizio, `conodiscesa2.jpeg`, i 4 PDF di
  figure usati da `tesi.tex` — `figure_sim/batch_size_app.png`,
  `figure_nsynth_nota/{nota_accuracy,nota_batch}.pdf`,
  `figure_nsynth_net/{nsynth_accuracy_net,nsynth_batch_net}.pdf` — e i 3
  notebook Colab citati nella tesi). I PNG/npz/json delle figure prodotte dai
  Colab sono stati spostati in **`colab_risorse/figure/`** (sottocartelle
  `nota/`, `net/`, `famiglia/`); i pesi dei modelli della rete neurale ora sono
  in repo in **`colab_risorse/`** (`pesi_net.npz` copiato da
  `Documents/placeholder/nsynth_net_locale/`, `scaler_net.npz` spostato da
  `tesi/nsynth/`). Tutto il materiale non usato dalla tesi è in **`altro/`**:
  `tesi_sapthesis.tex/pdf`, `bozza.tex/pdf`, `metodinumerici.tex`, `contenuti/`
  (frammenti già incorporati), `figure/` (vecchie, non referenziate),
  `figure_test/`, `figure_sim/batch_size.pdf`, `ocr_f/` + `ocr_appendix_f.swift`,
  `script/` (bbccv.py, rapg.py, sim_exp.py, restructure*, splice*, ecc.),
  `tabelle/` (tabella6_1/6_2) e `nsynth/` con i notebook non citati
  (`nsynth_riproduzione`, `music_reco_riproduzione`) e gli script degli
  esperimenti NSynth (run_nota.py, features_nota.py, run_benchmark.py,
  features.py). Nessun sorgente LaTeX toccato: `tesi.pdf`/`tesi_finale.pdf`
  invariati, compilazione non necessaria. Verificato che tutti i file
  referenziati da `tesi.tex` esistano ancora in `tesi/`.
- **Ultimo intervento (19/08/2026).** **"Codici" al posto di "listati",
  eliminato il doppio numero B.1--B.3 nei titoli, rimossa la parola "firma"
  dalle sezioni NSynth.** In `tesi/tesi.tex`: (1) l'Appendice B si intitola ora
  "Codici Python completi" e "listati"/"listato" sono stati sostituiti con
  "codici"/"codice" in tutto il documento (incluso il label interno
  `app:listati` → `app:codici`, riferimenti aggiornati); (2) rimossi i prefissi
  manuali "B.1:"/"B.2:"/"B.3:" dai titoli delle tre sottosezioni dell'Appendice
  B, che comparivano due volte nel PDF ("B.1 B.1: …") perché il numero è già
  generato automaticamente da `\appendix`; (3) eliminata la parola "firma" nelle
  sezioni NSynth (6.5 nota e 6.5.1 rete): le 11 etichette "Firma:" nei nodi TikZ
  ora mostrano solo la chiamata di funzione con ingressi → uscite, e in prosa
  "con la *firma* (ingressi → uscite)" è diventato "(ingressi → uscite)".
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (97 pp invariati); `bozza.tex` non
  toccata (non contiene Appendice B né NSynth). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
- **Ultimo intervento (19/08/2026).** **Figure 6.2, 6.3, 6.5, 6.6 e 6.7
  rimpicciolite (layout orizzontale).** Le 5 figure delle funzioni del notebook
  nota erano a colonna verticale (5–6 nodi impilati, strette ma altissime) e il
  `\resizebox{0.72–0.8\textwidth}` le ingrandiva (scala > 1 rispetto alla
  larghezza naturale). Ristrutturate in layout orizzontale come le figure della
  sottosezione rete neurale: firma in alto, passi in fila da sinistra a destra,
  uscita in basso; `\resizebox` ~0.72\textwidth (0.88 per la Fig. 6.7
  `grad_full`, 4 colonne), box da 5.2 a 6.5 cm, distanze compatte. Compilazione
  senza errori né riferimenti indefiniti, nessun overfull nuovo (i 13 residui
  sono preesistenti). Documento da 101 a **97 pp** (`tesi.pdf`,
  `tesi_finale.pdf`). `bozza.tex` non toccata. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
- **Ultimo intervento (19/08/2026).** **Sez. 6.5 divisa in due subsection e
  schemi delle funzioni del notebook nota.** (1) La Sez. 6.5 ora è composta da
  due `\subsection`: "Validazione su un benchmark musicale: riconoscimento
  della nota con NSynth" (`sec:nsynth`) e "Estensione oltre le ipotesi:
  implementazione vettorizzata per la rete neurale" (`sec:nsynth-net`, prima
  `\subsubsection` 6.5.1, ora `\subsection` 6.6, contenuto invariato). (2)
  Rimosso il titolo in grassetto "Cosa significa 'modello lineare'." (il testo
  del paragrafo è conservato). (3) Aggiunti 7 schemi TikZ delle funzioni del
  primo notebook (`nsynth_nota_riproduzione`, riconoscimento della nota),
  strutturati esattamente come quelli della sottosezione rete neurale (firma
  ingressi→uscite e formule esatte): Fig. 6.1 mappa delle funzioni
  (scarica_estrai → estrai_chroma → costruisci_dati → loss_i/grad_i/hessvec_i/
  grad_full → algoritmi B.1–B.4 → acc_test), Fig. 6.2 estrai_chroma (clip →
  x_i in R^24: media e dev. std del chromagramma), Fig. 6.3 costruisci_dati
  (X, y, standardizzazione sul train), Fig. 6.4 loss_i (softmax stabile,
  cross-entropia, L2), Fig. 6.5 grad_i, Fig. 6.6 hessvec_i, Fig. 6.7 grad_full.
  Le figure della sottosezione rete neurale (già 6.5–6.11) sono ora 6.8–6.14.
  (4) Paragrafo "Riproducibilità": aggiunto il link Colab diretto del notebook
  `nsynth_nota_riproduzione`. Compilazione senza errori né riferimenti
  indefiniti, nessun overfull nuovo (i 13 residui sono preesistenti).
  Documento da 94 a **101 pp** (`tesi.pdf`, `tesi_finale.pdf`). `bozza.tex`
  non toccata (la sezione NSynth non c'è). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
- **Ultimo intervento (19/08/2026).** **Sez. 6.5: tagliato l'esperimento NSynth
  sulla famiglia strumentale; la sezione tratta solo il riconoscimento della
  nota (pitch class).** Riscritta in `tesi/tesi.tex` la sottosezione
  `sec:nsynth` (ora "Validazione su un benchmark musicale: riconoscimento
  della nota con NSynth"): mantiene l'introduzione sul dataset NSynth e lo
  split validation/test, e sposta qui le features (chromagramma, 24 dimensioni)
  e il problema di apprendimento (12 classi, 300 parametri) del pitch;
  eliminati il riconoscimento della famiglia strumentale (10 classi, features
  mel, 810 parametri), la tabella `tab:nsynth` (57.8--59.3%), le figure
  `fig:nsynth_acc`/`fig:nsynth_batch`, le "quattro osservazioni" e la tabella
  del trade-off $L_1$ `tab:nsynth_l1`. La famiglia resta citata in una sola
  frase di passaggio (~59% contro il 20.8% della classe maggioritaria; il
  limite è nella rappresentazione) e come compito della rete neurale
  (Sez. 6.5.1, invariata, 96.9--99.2%). Adattati il paragrafo riassuntivo, la
  "Riproducibilità" (resta solo il notebook `nsynth_nota_riproduzione`) e la
  "Precisione sui valori numerici" (rimosso l'esempio Colab della famiglia).
  Nessun altro capitolo citava i label rimossi; compilazione senza errori né
  riferimenti indefiniti. Documento da 96 a **94 pp** (`tesi.pdf`,
  `tesi_finale.pdf`). `bozza.tex` non toccata (la sezione NSynth non c'è).
  Sincronizzati tesi.tex/tesi.pdf/tesi_finale.pdf nella repo (md5 verificati).
- **Ultimo intervento (18/08/2026).** **Sez. 6.5.1: figure 6.5--6.11 distanziate
  per eliminare le sovrapposizioni tra i box.** Le 7 figure a livello di
  funzione (6.5--6.11) avevano i box dei nodi che si toccavano/sovrapponevano
  (es. le righe a 3 box delle Figg. 6.8 e 6.11 con centro distante 5.4~cm ma
  larghezza 5.55~cm). Coordinate e `text width` riviste: 6.5 mappa funzioni
  (X da $-8$ a $8$, righe a y~5.5/0/--3.5/--6.5, io/box 3.8~cm, fcn 3.4~cm);
  6.6 estrai\_features (y 8.5, 6.0, 2.8, --2.8, --6.8, --9.5; x $-8,0,8$);
  6.7 forward (`node distance` 1.2/2.0, box parametri a 2.0~cm); 6.8 loss\_batch
  (riga fwd/sm/ce a $x=\pm6$, firma y~4.5); 6.9 grad\_batch (catena verticale
  distanziata, strati a $\pm7$); 6.10 ccv\_stats (nodi a $x=\pm4$);
  6.11 hess\_batch (riga f/hvp/eq a $x=\pm6$). Verificato con analisi geometrica
  delle bounding box: nessuna sovrapposizione residua; compilazione senza
  errori, nessun `Overfull \vbox`, nessun `Float too large`. Documento invariato
  a **95 pp** (`tesi.pdf`, `tesi_finale.pdf`). `bozza.tex` non toccata.
- **Ultimo intervento (18/08/2026).** **Sez. 6.5.1: figure 6.5--6.11 riscritte
  come schemi a livello di funzione.** Le 5 figure precedenti (pipeline,
  architettura, gradiente, varianza CCV, Hessiano--vettore) sono state
  sostituite da 7 figure "a livello di funzione", con la *firma* (ingressi
  $\to$ uscite), le formule esatte implementate e i nomi identici al notebook
  `nsynth_net_riproduzione`: 6.5 mappa delle funzioni (estrai_features ->
  RobustScaler -> forward -> loss_batch/grad_batch/ccv_stats/hess_batch ->
  algoritmi B.1--B.4 -> acc_test); 6.6 estrai_features (clip audio -> vettore
  $x_i\in\mathbb{R}^{780}$, 6 famiglie di descrittori con formule e
  concatenazione); 6.7 forward; 6.8 loss_batch; 6.9 grad_batch; 6.10 ccv_stats;
  6.11 hess_batch. Obiettivo: un programmatore che non conosce l'audio e un
  matematico devono capire come si legano le funzioni leggendo solo gli schemi.
  Layout compatti (righe/colonne, non più catene verticali che sforavano):
  nessun `Overfull \vbox`, nessun `Float too large`. Prosa della sottosezione
  adattata. `bozza.tex` non allineata (la sezione non c'è). Documento da 92 a
  **95 pp** (`tesi.pdf`, `tesi_finale.pdf`).
- **Ultimo intervento (18/08/2026).** **Sez. 6.5.1: corrette le 5 figure TikZ
  della rete neurale (Figg. 6.5--6.9), che erano enormi e sforavano la
  pagina.** Causa: `\resizebox{0.92\textwidth}{!}` scalava a tutta larghezza
  diagrammi stretti e alti, ingrandendoli fino a ~38~cm di altezza (grad,
  ccv, hess finivano tagliati sotto il bordo pagina, sulle pp. 60/62/64).
  Fix: (1) pipeline (6.5) ristrutturata a *flusso unico* data->feat->net->
  algs->eval: eliminata la freccia `(feat) -| (net)` che si sovrapponeva al
  nodo `algs`; (2) gradiente (6.7) e varianza CCV (6.8) ridisegnate a **due
  colonne** (serpentina 3+3 e 3+2); (3) Hessiano--vettore (6.9) a griglia
  **2$\times$2**. Larghezze ridotte: pipeline 0.78, grad/ccv 0.62, hess 0.58
  di `\textwidth`. Architettura (6.6) invariata. Verificato: nessuna pagina
  ha più contenuto sotto il footer e ogni figura sta su una sola pagina.
  `bozza.tex` non allineata (la sezione non c'è). Documento da 96 a **92 pp**
  (`tesi.pdf`, `tesi_finale.pdf`).

- **Ultimo intervento (18/08/2026).** **Appendice E: riscritta la spiegazione
  del passo di Barzilai--Borwein (Sottosezione E.1).** Nuovo testo al posto
  della versione precedente: espansione di Taylor del gradiente (E.1),
  definizioni di $s_k, y_k$ (E.2), relazione fondamentale
  $y_k \approx \nabla^2 J(w_k)\, s_k$ (E.3), equazione della secante con
  l'approssimazione scalare $B = \alpha^{-1} I$ (non più numerata), le due
  formule dei minimi quadrati $\alpha_k^{(1)}$ (E.4) e $\alpha_k^{(2)}$ (E.5),
  il confronto tra le due formule ("la seconda è più aggressiva") e la
  motivazione della salvaguardia $\mathrm{clip}(\cdot, \alpha/20, 5\alpha)$;
  paragrafo finale ("Nella pratica...") spostato dopo la Tabella E.1 (ordine
  del nuovo testo). Rimossi l'enumerate delle 4 limitazioni e i riferimenti
  testuali alla Figura~`fig:bbccv` in questa sottosezione (la figura resta
  nello schema E.2). Numerazione equazioni da E.1--E.6 a E.1--E.5: verificato
  che nessun altro punto del documento cita le equazioni E.4--E.6.
  `bozza.tex` non allineata (l'appendice non c'è). Documento da 98 a
  **96 pp** (`tesi.pdf`, `tesi_finale.pdf`).

- **Ultimo intervento (18/08/2026).** **Tesi: Sez. 6.5.1 arricchita con i
  risultati dell'esperimento rete neurale.** Aggiunto in `tesi/tesi.tex`
  (sottosezione `sec:nsynth-net`): la tabella dei risultati
  (`tab:nsynth_net`: accuratezze test 96.9/99.2/97.3/96.2%, norme del
  gradiente, batch finale 2048, tempi su Apple M5, sparsità L1 58 412/234 122),
  le due figure dell'esperimento da `tesi/figure_nsynth_net/`
  (`fig:nsynth_net_acc` accuratezza vs iterazioni, `fig:nsynth_net_batch`
  dinamica del batch), i link ai notebook Colab
  (`nsynth_net_riproduzione`, `nsynth_net_test`) e al repository, e la
  descrizione del test su clip reali con etichette vere (es. clip keyboard
  classificata da tutti i metodi con probabilità > 99%). Ricompilati `tesi.pdf`
  e `tesi_finale.pdf` (96 pp invariati); `bozza.tex` non toccata (non contiene
  la sezione 6.5). Sincronizzati tesi.tex/tesi.pdf/tesi_finale.pdf nella repo
  (md5 verificati).
- **Ultimo intervento (18/08/2026).** **Notebook NSynth rete neurale: esecuzione
  locale con batch cap 2048.** Modificato `tesi/nsynth/nsynth_net_riproduzione.ipynb`:
  Cella 6 con `MAX_ITER = 200` (era 400); in **tutte le funzioni** della Cella 4
  (listati B.1–B.4) il cap della CCV passa da `N` a **2048**
  (`n = min(n_new, 2048)`, inclusa la forma inline di Newton-CG); aggiunto il
  salvataggio dei pesi finali in `pesi_net.npz` (chiavi `Dynamic_GD`,
  `Newton-CG`, `Newton-CG_L1`, `BB-CCV`). Eseguito **interamente in locale sul
  Mac** (non Colab): celle 0, 2, 3, 3b, 4, 5, 6 con le features dai `.npz`
  cache (valid 12 678, test 4 096). Risultati (seed 42, MAX_ITER=200, batch
  cap 2048): **Dynamic GD** acc 96.9% (176 s), **Newton-CG** acc 99.2% (270 s),
  **Newton-CG L1** acc 97.3% (263 s, nnz 58 412/234 122 ≈ 25%, sparsità 75%),
  **BB-CCV** acc 96.2% (247 s); batch finale 2048 per tutti; cella 6 in ~973 s
  (~16 min). `results_net.json` e `pesi_net.npz` salvati nella cartella di
  esecuzione locale (`Documents/placeholder/nsynth_net_locale/`, NON in repo
  per dimensione). Generate anche le **figure dell'esperimento** con le celle
  7–8 (`nsynth_accuracy_net.png/pdf`, `nsynth_batch_net.png/pdf` + 
  `results_net.json`) in **`tesi/figure_nsynth_net/`** (nuova cartella,
  in repo). Nuovo **`tesi/nsynth/nsynth_net_test.ipynb`**: Colab per testare i
  4 modelli addestrati su una **clip reale di uno strumento** (upload di
  `pesi_net.npz` + `scaler_net.npz`; lo scaler, RobustScaler 5–95 del training,
  è salvato in repo come `tesi/nsynth/scaler_net.npz`, i pesi restano in locale
  per dimensione). Validato localmente: accuratezze sul test set identiche
  (96.9 / 99.2 / 97.3 / 96.2%) e predizioni corrette su clip reali. Nessun
  sorgente LaTeX toccato.
- **Ultimo intervento (18/08/2026).** **Nuovo Colab: NSynth con rete neurale
  (esplorativo, oltre le ipotesi convesse).** Aggiunto
  `tesi/nsynth/nsynth_net_riproduzione.ipynb`: gli stessi algoritmi dei listati
  B.1–B.4 applicati a una rete a due strati (256–128, tanh) al posto della
  regressione logistica della Sez. 6.5. È uno **studio empirico fuori dalle
  ipotesi** della tesi (loss non convessa): da usare eventualmente come nuova
  sottosezione esplorativa, NON come sostituzione della Sez. 6.5. Rispetto al
  codice originale (che si bloccava su Colab per i loop per-esempio con
  autograd su D≈234k), le funzioni `loss_i`/`grad_i`/`hessvec_i` sono sostituite
  dalle versioni **batch matriciali** `loss_batch`/`grad_batch`/`hess_batch`,
  matematicamente equivalenti (cella 3b le verifica numericamente contro
  autograd: errori ~1e-15) ma senza loop Python. CCV fedele al listato B.1
  (varianza per-esempio esatta via `Σ‖g_i‖²` a blocchi, NON split-half);
  Newton-CG con `gamma=0` (la varianza per-esempio degli Hv è proibitiva).
  **α=1.0** (valore della tesi): con α=0.1 la line search di Wolfe del Dynamic
  GD non trovava passi (step=0). Validato su dati sintetici: 4/4 metodi
  apprendono (99–100% su classi separabili). Tempi Colab: estrazione features
  15–25 min; esecuzione ~20–60 min per metodo con MAX_ITER=400, MAXCG=30.
  Nessun sorgente LaTeX toccato (documento invariato a 91 pp).
- **Ultimo intervento (18/08/2026).** **Ripristinato il flusso a due documenti
  per `tesi_finale.pdf`.** `compila_tesi.sh` torna alla versione "merge pypdf"
  del periodo pre-`d54e1d1`: compila `frontespizio.tex` (sapthesis, 1 p) +
  `tesi.tex` (article) e unisce frontespizio + documento senza copertina ->
  `tesi_finale.pdf` (**90 pp**). Il PDF rigenerato è **byte-identico** al
  `tesi_finale.pdf` del commit `fa28586` (l'ultimo prodotto col vecchio flusso,
  md5 `29cce8c3…`), perché `tesi.tex` non è cambiato da allora.
  `tesi_sapthesis.tex`/`tesi_sapthesis.pdf` (96 pp) restano in repo come
  riferimento storico, non più usati per il PDF definitivo (README aggiornato:
  struttura, sezione PDF definitivo, nota "File in lavorazione").
- **Ultimo intervento.** **Nuovo Colab: raccomandazione musicale (Logistic
  Matrix Factorization).** Aggiunto `tesi/nsynth/music_reco_riproduzione.ipynb`:
  riproduzione di un nuovo esperimento applicativo (candidato per una futura
  Sez. 6.6) sul problema della raccomandazione musicale con feedback implicito
  (dataset **Last.fm HetRec 2011**, `user_artists.dat`: ascolti utente→artista).
  Il problema è la regressione logistica pesata su **tutte** le N = n_u × n_i
  coppie utente–artista (pesi di confidenza c_ui = 1 + α·r̂_ui, Hu et al. 2008
  / Johnson 2014), fortemente convessa grazie alla regolarizzazione L2 → siamo
  nelle ipotesi della tesi (Sez. 3.5). Il notebook esegue i **listati B.1–B.4
  verbatim** (stesso codice dei Colab NSynth), valuta **Recall@10/NDCG@10**
  con baseline popolarità, mostra le **predizioni dei 4 modelli** su un utente
  esempio e salva `music_reco_results.json`/`music_reco_results.csv` e
  `music_reco_curves.npz` per rigenerare le tabelle. Default: 250 utenti ×
  200 artisti, K=8, MAX_ITER=200 (Colab ~10–20 min); fallback automatico su
  dataset sintetico se il download fallisce. Nessun sorgente LaTeX toccato:
  documento invariato a **96 pp**.
- **Ultimo intervento.** **Figura 5.1 spostata e notebox senza sforo (pag. 14).**
  Il block diagram del Dynamic GD (schema CCV) è stato spostato alla **fine
  della Sezione 5.1, subito prima della 5.2** (come richiesto: "dopo la
  formulazione della condizione di accettazione, non in mezzo"). Conseguenza:
  la figura ora è la **Figura 5.3** (il cono è 5.1 e l'andamento di $n_k$ è
  5.2); i riferimenti `\ref` si aggiornano da soli (l'unico rimando esplicito è
  in Appendice E). **Notebox "Quando siamo lontani dal minimo":** non sfora più
  dalla pag. 14 alla 15 — ora sta interamente nella pagina stampata 14,
  grazie a `\enlargethispage{5\baselineskip}` prima del notebox "Dalla
  condizione di discesa" e a un **padding dei box leggermente compatto**
  (`\tcbset{boxsep=1mm, top=1mm, bottom=1mm}`: necessario, senza la pagina
  tornava a 97 pp con il notebox su pag. 15). Rimossi 3 commenti duplicati
  residui. Documento: **96 pp** (`tesi_finale.pdf`), nessuna pagina vuota.
  Contenuto invariato (solo impaginazione).

- **Ultimo intervento.** **Nuovo documento unico in classe sapthesis
  (`tesi_sapthesis.tex`).** L'intera tesi è stata ricostruita come documento
  unico in classe `sapthesis` (base `book`), eliminando il flusso a due
  documenti (frontespizio.tex + tesi.tex con merge pypdf). Il frontespizio
  istituzionale è ora integrato via `\maketitle` come pagina 1 (identica al
  vecchio frontespizio.pdf), poi Ringraziamenti, Abstract, Indice, i 7
  capitoli, le appendici A–E e la bibliografia. Mappatura dei livelli:
  `\section`→`\chapter`, `\subsection`→`\section`,
  `\subsubsection`→`\subsection` (i numeri mostrati restano gli stessi, es.
  5.1.2 resta 5.1.2; `secnumdepth=2` e `tocdepth=2`).
  **Impaginazione a flusso continuo (vicina alla vecchia versione article):**
  `\cleardoublepage` ridefinito a `\clearpage` (niente pagine bianche) e
  `\chapter` ridefinito senza salto di pagina forzato (si va a capo solo
  quando serve); titoli di capitolo ridimensionati a `\LARGE` compatti
  (invece del `\Huge` di default con 50pt di spazio). Corretta la deriva dei
  float: il block diagram del Dynamic GD (`\resizebox` portato a 0.8
  `\textwidth`) era più alto della pagina e bloccava la coda dei float
  (trascinava anche la Figura 5.2); rimosso un `\newpage` che orfanava il
  titolo "5.2.3 Pseudocodice". **Indice con link neri** (override locale di
  `linkcolor`; i link nel corpo restano `coverblue`). Corretti 3 riferimenti
  da "Sezione" a "Capitolo" (ora puntano a capitoli: `sec:formulazione`,
  `sec:algoritmi`, `sec:lavori`). Documento: **95 pp** (`tesi_finale.pdf`),
  nessuna pagina vuota. `tesi.tex` (article) resta come documento di lavoro
  (90 pp, con la copertina in stile sapthesis); `bozza.tex` ripristinata alla
  versione originale (la modifica della copertina NON si applica alla bozza).
  `compila_tesi.sh` aggiornato: ora compila solo `tesi_sapthesis.tex` ->
  `tesi_finale.pdf`.

- **Ultimo intervento.** **Copertina di `tesi.tex` uniformata al frontespizio
  sapthesis.** Rimossa la vecchia copertina blu custom (`\pagecolor{coverblue}`,
  testo bianco, figura TikZ del cono di discesa e filetto bianco) e sostituita
  con una copertina in stile frontespizio sapthesis: sfondo bianco, font
  sans-serif (`\sffamily`), logo Sapienza in alto (`sapienzalogo.pdf`, incluso
  nella classe sapthesis), titolo `\LARGE` e sottotitolo `\large` nel colore
  bordeaux sapthesis (nuovo `\definecolor{sapred}`, stesso valore di
  sapthesis.cls), blocco università/corso + autore e Anno Accademico in basso,
  disposti come nel frontespizio (parbox indentata, anno in fondo con
  `\vfill`). Testo della copertina invariato parola per parola (titolo
  "Selezione Dinamica della Dimensione del Campione in Metodi di Ottimizzazione
  per il Machine Learning", sottotitolo "Algoritmi adattivi a campione dinamico
  e sottocampionamento dell'Hessiana per problemi su larga scala", autore +
  `\today`, corso, Anno Accademico 2025--2026): cambiati solo font e
  impaginazione. Sillabazione disattivata nel blocco titolo (`\hyphenpenalty`
  /`\exhyphenpenalty`); sottotitolo spezzato a capo a "dinamico e"; indentazione
  delle parbox ridotta a 0.3cm perché il titolo `\LARGE` non entrava con
  l'indentazione sapthesis standard (1.72cm) nella geometria `article`
  (textwidth 15cm). `frontespizio.tex`/`frontespizio.pdf` invariati. Nota: il
  `tesi.pdf` committato era rimasto non sincronizzato (conteneva ancora le
  etichette in grassetto dell'Appendice E rimosse nel commit 23ff1e6); la
  sincronizzazione lo ha riportato coerente con il sorgente (i `\textbf`
  residui sono solo i 6 della Sez. 6.5).

- **Ultimo intervento.** **Appendice E: eliminate le etichette di paragrafo in
  grassetto.** Rimosse del tutto (non solo sgrassate) le 9 etichette
  `\textbf{...}` all'inizio dei paragrafi della sottosezione E.1
  (Barzilai--Borwein): "L'espansione di Taylor del gradiente.", "Spostamento e
  variazione del gradiente.", "La relazione tra Hessiana e dati osservati.",
  "Il collegamento con i metodi quasi Newton.", "Le due formule di
  Barzilai--Borwein.", "Un metodo quasi Newton senza memoria.", "Il caso
  quadratico è il caso ideale.", "I limiti sulle funzioni non quadratiche.",
  "Sintesi.". Ogni paragrafo ora inizia con la frase successiva (testo
  invariato); conservati i corsivi (`\emph`, es. *spostamento*, *equazione
  della secante*, i 4 punti dell'enumerate). Documento invariato a **90 pp**
  (`tesi.pdf`); nessun `\textbf` residuo in Appendice E (i 6 rimasti nel
  documento sono in Sez. 6.5). `bozza.tex` non allineata.

- **Ultimo intervento.** **Sez. 6.3 (Osservazioni comparative): paragrafo su
  una sola pagina.** Il paragrafo dopo le Tabelle 6.1–6.3 (pag. 50) sforava di
  ~4 parole ("BB-CCV – permette di progredire.") sulla pagina successiva.
  Testo ridotto di poco ("è il metodo migliore in assoluto" → "è il migliore",
  via "proprio", "e non richiede la Hessiana" → ", senza la Hessiana") e
  incisi convertiti da trattini a virgole: "l'informazione di curvatura,
  esplicita in Newton-CG o stimata dal passo di Barzilai–Borwein in BB-CCV,
  permette di progredire". Documento invariato a **90 pp** (`main.pdf`,
  `tesi.pdf`), `tesi_finale.pdf` 90 pp. `bozza.tex` non allineata.
- **Intervento precedente.** **Appendice E: spiegazione del metodo di
  Barzilai--Borwein.** Aggiunta la sottosezione E.1 "Il metodo di
  Barzilai--Borwein" (lo schema è ora nella sottosezione E.2): le tre
  equazioni fondamentali (espansione di Taylor del gradiente, definizioni di
  $s_k$/$y_k$, relazione $y_k \approx \nabla^2 J(w_k)\, s_k$), il collegamento
  con i metodi quasi Newton (equazione della secante $B_{k+1} s_k = y_k$,
  approssimazione scalare $B = \alpha^{-1} I$ e minimi quadrati), le due
  formule $\alpha^{(1)} = s^\top y / y^\top y$ e $\alpha^{(2)} = s^\top s /
  s^\top y$ (la seconda è quella usata in BB-CCV), il caso quadratico, i
  quattro limiti sulle funzioni non quadratiche (secante solo locale,
  mancanza di memoria, denominatore $s^\top y$, derivate terze), una tabella
  riassuntiva (Tab. E.1) e una sintesi che riconduce alla salvaguardia
  (clip $[\alpha/20, 5\alpha]$ + line search di Armijo) di BB-CCV. Stile
  allineato alle altre appendici: nessun riferimento a metodi non citati nel
  documento ("quasi Newton" generico, non BFGS/DFP/SR1; SVRG/SAGA citati solo
  come metodi a varianza ridotta), nessun residuo da conversazioni ("nel tuo
  codice"). Documento da 87 a **90 pp** (`main.pdf`, `tesi.pdf`),
  `tesi_finale.pdf` 90 pp. `bozza.tex` non allineata.
- **Intervento precedente.** **Sez. 6.5: due esperimenti su NSynth (famiglia +
  nota).** La sezione ora contiene: (1) riconoscimento della **famiglia
  strumentale** (10 classi, features mel 80D, acc. 57.8–59.3%, con dinamica
  del batch CCV e sparsità $L_1$); (2) riconoscimento della **nota** (pitch
  class, 12 classi, features chroma 24D, acc. 88.0–91.4%, con Newton-CG in
  linea col riferimento sklearn L-BFGS 91.5% e Newton-CG-$L_1$ convergente in
  128 iterazioni con il 51% di coefficienti nulli). Aggiunto un paragrafo che
  spiega il "modello lineare" e la semplificazione rispetto alle reti
  profonde; rimossa dalla Sez. 7.3 la voce "validazione su benchmark reali"
  (ora effettuata); nota di riproducibilità con i Colab. **Colab**:
  `tesi/nsynth/nsynth_riproduzione.ipynb` (famiglia, aggiornato con la cella
  finale "predizioni di un esempio con audio") e nuovo
  `tesi/nsynth/nsynth_nota_riproduzione.ipynb` (nota, con confronto sklearn e
  predizioni). Script: `features.py`, `run_benchmark.py`, `features_nota.py`,
  `run_nota.py`; output in `tesi/figure_nsynth/` e `tesi/figure_nsynth_nota/`.
  I dati audio/features NON sono in repo (istruzioni di riproduzione come
  prima). Documento da 85 a **87 pp**.
  `bozza.tex` non allineata.
- **Intervento precedente.** **Validazione su benchmark reale (NSynth).** Nuova
  Sez. 6.5: i quattro metodi (Dynamic GD, Newton-CG, Newton-CG~$L_1$, BB-CCV)
  sono validati sul dataset NSynth (riconoscimento famiglia strumentale, 10
  classi) con logistic regression multinomiale su features mel (80D).
  Addestramento = split validation NSynth (12 678 clip), valutazione = split
  test (4096 clip; strumenti disgiunti per costruzione → generalizzazione a
  strumenti mai visti). Risultati: accuratezza test 57.8–59.3% (Newton-CG il
  migliore e il più veloce a superare il 55%, k=31); la CCV porta il batch da
  64 a N (12678) man mano che il gradiente si riduce; la regolarizzazione $L_1$
  (ν=1e-3) azzera il 48% dei coefficienti con perdita trascurabile (collegata
  all'interpretabilità). Script in `tesi/nsynth/` (`features.py`,
  `run_benchmark.py`), output in `tesi/figure_nsynth/`, e notebook Colab
  `tesi/nsynth/nsynth_riproduzione.ipynb` che scarica NSynth, estrae le
  features, esegue i listati B.1–B.4 (verbatim dall'Appendice B) e rigenera
  figure e tabelle della Sez. 6.5 (tempo stimato in Colab: 15–25 min).
  I dati audio/features NON sono in repo: per riprodurre, scaricare NSynth
  valid/test da download.magenta.tensorflow.org, estrarre, eseguire
  `features.py` e `run_benchmark.py` (o aprire il notebook in Colab).
  Documento da 82 a **85 pp**.
  `bozza.tex` non allineata.
- **Intervento precedente.** **Sez. 6 allineata ai preset dell'app
  (`visualizzazione.html`).** Riscritto il Setup Sperimentale (6.1) sulle tre
  funzioni test dell'app: quadratica ben condizionata (κ≈1.1), molto mal
  condizionata (κ≈100) e con termine incrociato, con i parametri di default
  (N=200, w0=(2,-3), seed 42, α=0.1, θ=0.5, n0=5, 30 iterazioni, toll. 1e-6).
  Eliminati dal documento i contenuti NON riproducibili con l'app: le tabelle
  6.1/6.2 (confronti con SGD/Batch GD da `sim_exp.py`, κ≈1.4/107), le figure
  6.1/6.2 (convergenza/bar) e il problema di Rosenbrock con le relative
  discussioni. Le tabelle dell'errore sono ora le Tabb. 6.1–6.3 (4 metodi) e
  riportano i valori del pannello "Analisi" dell'app. Sez. 5.3.1: frase
  sull'interpretabilità resa esplicita. Documento da 84 a **82 pp**.
  `bozza.tex` non allineata.
- **Intervento precedente.** **RAPG rimosso dal documento.** Eliminate tutte le
  tracce di RAPG dalla tesi: listato B.5, schema E.2 (RAPG) e l'Appendice F
  (Test comparativi aggiuntivi) sono stati cancellati. Le tre tabelle
  dell'errore ($\|w_k-w_*\|_2$ a ogni iterazione, **4 metodi**: Dynamic GD,
  Newton-CG, Newton-CG~$L_1$, BB-CCV) sono state **spostate in Sez. 6.3**
  (Risultati Numerici, Tabb. 6.3–6.5) con il paragrafo "Osservazioni
  comparative" riscritto senza RAPG; l'Appendice E è rinominata "Schema
  dell'algoritmo BB-CCV" (resta un solo schema) e ora precede direttamente la
  bibliografia. Script aggiornati: `gen_tabelle_f.py` genera le tabelle a 4
  metodi (subtable 0.24), rimossi i file OCR RAPG da `tesi/ocr_f/`.
  Documento da 89 a **84 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
  Sistemata anche la Sez. 1.5 "Struttura della tesi": le appendici sono ora
  indicate come A--E (prima solo A, B e C).
  `bozza.tex` non allineata.
- **Intervento precedente.** **Appendice F: da screenshot a tabelle.** Le 15
  schermate del pannello "Analisi" (5 metodi × 3 problemi) sono sostituite da 3
  tabelle (Tabb. F.1–F.3), una per funzione obiettivo, ciascuna con i 5
  algoritmi affiancati (subtable (a)–(e)); per ogni metodo la tabella riporta
  $\|w_k-w_*\|_2$ a ogni iterazione $k$. I valori sono quelli del pannello
  "Analisi", estratti dalle 15 PNG in `tesi/figure_test/` con OCR (framework
  Vision di macOS, script `tesi/ocr_appendix_f.swift`; output OCR in
  `tesi/ocr_f/`) e inseriti tramite `tesi/gen_tabelle_f.py` (nessuna
  trascrizione manuale); i PNG restano in repo ma non sono più inclusi nel
  documento. Ritocchi: intro dell'appendice e caption riscritte ("Le Figure" →
  "Le Tabelle"), "Dalle schermate" → "Dai valori delle Tabelle~F.1--F.3",
  "in appena 10 iterazioni" corretto in "in 11 iterazioni"
  ($\|w-w_*\|=1.13\times10^{-14}$ a $k=11$). Nota: le tabelle BB-CCV hanno
  meno righe (13/12/17) perché quei run si fermano prima. Documento da 88 a
  **89 pp** (`main.pdf`, `tesi.pdf`); `tesi_finale.pdf` rigenerato.
  `bozza.tex` non allineata (l'appendice non c'è).
- **Intervento precedente.** Tre interventi collegati: (1) **Appendice F aggiornata** a
  5 metodi × 3 problemi (aggiunto **BB-CCV**): 15 screenshot del pannello
  "Analisi" (orari 22:55–23:51 del 16/08), raggruppati per funzione obiettivo
  (ben condizionata, molto mal condizionata, termine incrociato) nelle Figg.
  `fig:test_bencond/malcond/incrociato` con subfigure (a)–(e) Dynamic GD,
  Newton-CG, Newton-CG $L_1$, RAPG, BB-CCV; nuovo paragrafo "Osservazioni
  comparative" (BB-CCV il migliore in assoluto; sui mal condizionati $10^{-14}$
  in 10 iterazioni). Mappa verificata con $J(w_0)$: il 23:27:53 è su ben
  condizionata e il 23:32:21 su molto mal condizionata (nella lista erano
  scambiati); usati i 3 test mal condizionati rifatti (23:50–23:51) al posto
  di 4–6 (dati duplicati). (2) **Listati B.4 (BB-CCV) e B.5 (RAPG)** in
  Appendice B nello stile di B.1–B.3 (blocchi `pythoncode` separati da
  spiegazioni matematiche); codici salvati in `tesi/bbccv.py` e `tesi/rapg.py`.
  (3) **Appendice E rinominata** "Schemi degli algoritmi BB-CCV e RAPG":
  aggiunto lo schema RAPG (`fig:rapg`, Fig. E.2) con precondizionatore
  diagonale RMSprop limitato e Regula Falsi 1D; schema BB-CCV verificato
  contro il codice. Documento da 80 a **88 pp** (`main.pdf`, `tesi.pdf`,
  `tesi_finale.pdf`). `bozza.tex` non allineata.
- **Intervento precedente.** Nuova **Appendice F "Test comparativi aggiuntivi"**
  (`app:test`): 12 screenshot del pannello "Analisi" dell'app
  (`visualizzazione (3).html`, 16/08, orari 22:55–23:32), raggruppati per
  funzione obiettivo nelle Figg. `fig:test_bencond`, `fig:test_malcond`,
  `fig:test_incrociato` (ben condizionata $\kappa\approx1.1$, molto mal
  condizionata $\kappa\approx100$, termine incrociato), con i 4 algoritmi per
  figura in subfigure (a)–(d): Dynamic GD, Newton-CG, Newton-CG $L_1$, RAPG.
  Le immagini sono in `tesi/figure_test/` (12 PNG da screenshot iPhone, anche
  nella copia di lavoro per la compilazione di `main.tex`); scartati gli
  screenshot 23:01–23:04 (dati duplicati). Aggiunto `\usepackage{subcaption}`
  al preambolo. Documento da 78 a **80 pp** (`main.pdf`, `tesi.pdf`,
  `tesi_finale.pdf`); Appendice F a pp. 77–78 (stampate).
  `bozza.tex` non allineata (l'appendice non c'è).
- **Intervento precedente.** Nuova **Appendice E "Schema dell'algoritmo BB-CCV"**
  (`app:bbccv`, figura `fig:bbccv`): schema a blocchi del gradiente a campione
  dinamico con passo di Barzilai--Borwein (BB-CCV), in stile `fig:block_diagram`
  (catena centrale + nodo laterale "CCV violata" con aumento di $n_k$ + loop),
  con il nodo del passo di Barzilai--Borwein
  ($\alpha_k = \mathrm{clip}(\alpha_k^{BB}, \alpha/20, 5\alpha)$), la line
  search di Armijo e una caption-spiegazione che riporta i risultati del
  confronto sperimentale (15 metodi, 7 problemi, 20 seed). Inserita tra
  Appendice D e bibliografia; documento da 77 a **78 pp** (`main.pdf`,
  `tesi.pdf`, `tesi_finale.pdf`); l'Appendice E sta tutta alla pag. 77.
  `bozza.tex` non allineata (l'appendice non c'è).
- **Intervento precedente.** Appendice D compattata: i due schemi SVRG
  (`fig:svrg`) e SAGA (`fig:saga`) ora sono **affiancati sulla stessa pagina**,
  uno nella metà sinistra e l'altro nella metà destra, con le caption sotto
  ciascuno nella propria metà. Struttura: un unico `figure[H]` con due
  `minipage[b]{0.48\textwidth}`; i contenuti dei due `tikzpicture` sono
  **invariati** (unico ritocco: `\resizebox{\textwidth}{!}` →
  `\resizebox{\linewidth}{!}`). Label e numerazione immutate (i rimandi in Sez.
  4.1 non cambiano). Documento da 78 a **77 pp** (`main.pdf`, `tesi.pdf`,
  `tesi_finale.pdf`); l'Appendice D va dalla pag. 75 (introduzione) alla pag.
  76 (i due schemi). `bozza.tex` non allineata (l'appendice non c'è).
- **Intervento precedente.** Pagina Ringraziamenti (riga 219): ridotta a una sola
  frase — "Ringrazio chiunque, in qualsiasi modo, mi abbia motivato a fare di
  più" (via insegnanti, amici, SMIA, nomi citati e famiglia, come richiesto).
  Documento invariato a **78 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
- **Intervento precedente.** Nuova Appendice D "Schemi dei metodi a varianza
  ridotta" (`app:schemi`): due schemi a blocchi in stile dei 3 algoritmi per
  SVRG (`fig:svrg`, con punto di ancoraggio $\bar w$ e gradiente esatto ogni
  $m$ iterazioni) e SAGA (`fig:saga`, tavola dei gradienti $g^{(i)}$ con media
  $\bar g$), nella notazione del documento; rimandi aggiunti in Sez. 4.1
  (Lavori Correlati) dove si citano johnson2013 e defazio2014.
  Documento da 76 a **78 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
- **Intervento precedente.** Lavoro futuro (Sez. 7.3, punto 1 "Rilassamento
  dell'ipotesi di convessità forte"): riscritto per chiarire che la PL nella
  tesi è una conseguenza della convessità forte e che la proposta è assumere
  la PL ($\|\nabla J(w)\|^2 \ge 2\mu J(w)$) come ipotesi di partenza,
  eliminando la convessità forte (caso particolare con $\mu=\lambda$). Via
  l'esempio della "regressione logistica regolarizzata" (richiesto).
  Documento invariato a **76 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
- **Intervento precedente.** Bibliografia riordinata per rilevanza rispetto al
  lavoro: byrd2012, bottou2008, nocedal2006, byrd2011, bottou2018,
  robbins1951, polyak1964, karimi2016, johnson2013, defazio2014, dembo1982,
  shewchuk1994 (prima: ordine alfabetico/inserimento). Solo `thebibliography`
  riordinata; tutte le `\cite{chiave}` intatte (rinumerazione automatica).
  Documento invariato a **76 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
- **Intervento precedente.** Nota di riproducibilità: aggiunto il seme casuale
  fissato (seed $42$) in Sez. 6.1 (Setup Sperimentale) e nell'Appendice C
  (Requisiti di sistema), per dichiarare la riproducibilità esatta delle
  tabelle 6.1–6.2 a parità di codice e versioni. `sim_exp.py` usa
  `np.random.default_rng(42)`; l'app `visualizzazione.html` ha seed 42 di
  default. Listati B.1–B.3 NON modificati (restano verbatim dalla bozza).
  Documento invariato a **76 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
- **Intervento precedente.** Appendice B (Listati Python, B.1–B.3): ripristinati
  **verbatim** dalla vecchia `bozza.tex` (sezione "Visualizzazione Interattiva",
  righe 1757–2230) i blocchi `pythoncode` **e** i commenti LaTeX con formule che
  seguono ogni blocco, come richiesto dall'autore. Il documento nuovo (commit
  `a2da2a9`) aveva riscritto i listati: 18 blocchi invece di 24, commenti
  descrittivi al posto dei `#` isolati, docstring su `cg`, firma
  `newton_l1(..., eta=0.5, R=0.1)`. Ripristinati: i `#` isolati, il frammento
  duplicato di Dynamic GD (`def J_batch` e `c1, step, J_curr, g_norm2` ripetuti),
  la firma `newton_l1(..., eta=0.5)` **senza** `R` (coerente con
  `visualizzazione.html`, dove `R` è globale) e la prosa con i riferimenti
  alle equazioni (6.x) della bozza. Conservati: introduzione dell'appendice,
  sezione "Istruzioni"; titoli delle sottosezioni rinominati da
  "Listato B.x: ..." a "B.x: ..." (via la parola "Listato"). Documento da 72 a
  **76 pp**
  (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`), `bozza.pdf` invariata (56 pp).
- **Intervento precedente.** Pagina Ringraziamenti (riga 219): aggiunto
  "i miei amici e in generale" all'incipit. Testo definitivo: "Ringrazio i
  miei insegnanti, i miei amici e in generale tutte le persone con cui ho
  studiato a SMIA, un ringraziamento speciale va a Giovanni Adelfio,
  Alessandro Pisarra, Emmanuel Nsia, Michele Aliffi, Francesco Lioi e Lorenzo
  Salis. Ringrazio infine la mia famiglia, e chiunque, in qualsiasi modo, mi
  abbia motivato a fare di più". `main.tex` e `tesi/tesi.tex` allineati, PDF
  rigenerati; 72 pp invariati (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`),
  `bozza.pdf` 56 pp.
- **Intervento precedente.** Paragrafo "Regolarizzazione $L_1$ in ottimizzazione"
  (Sez. 4.4, Lavori Correlati): riscritto limitandolo all'essenziale e
  spiegando in modo intuitivo (perché la $L_1$ produce sparsità: "costa
  sempre $\nu$ per unità di coefficiente", a differenza della $L_2$);
  rimosso il frammento fuori posto sulla CCV/dimensione del batch (già
  coperto altrove) e le citazioni a riferimenti non letti (`tibshirani1996`,
  `beck2009` — voci rimosse dalla bibliografia; `byrd2011` non più usato in
  questo paragrafo). 74→73→**72 pp** (`main.pdf`, `tesi.pdf`,
  `tesi_finale.pdf`).
- **Intervento precedente.** Rimossa la Figura 6.2 ("Evoluzione di $n_k$ rispetto a
  un batch fisso", `fig:batch`, ex pag. 51): ridondante con la Figura 5.3
  (screenshot dell'app, stessa informazione su $n_k$) e unico residuo del
  confronto col batch fisso. Aggiornato il testo degli Esperimenti, che ora
  cita solo la 6.1 (convergenza) e la 6.2 (bar chart, rinumerata da 6.3).
  Documento da 74 a **73 pp** (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
  `figure/batch_size.pdf` non è più incluso nel documento (resta nella repo).
- **Intervento precedente.** Sezione Newton-CG $L_1$, Tabella 5.2 e URL GitHub:
  (1) lo schema a blocchi `fig:newton_l1` finiva in fondo al documento (pag.
  72) a causa del float `[h]`: passato a `[H]` (come già fatto per la Fig.
  5.4), ora sta nella sezione (pag. 45); (2) Tabella 5.2 ridisegnata: prima
  sforava nel margine destro (colonne p{} per 14 cm + tabcolsep > textwidth
  15 cm); ora `\footnotesize`, colonne `\raggedright` ricalibrate (totale
  ~14.4 cm), niente overfull, rimossi i break forzati `\\(...)` nelle celle;
  (3) URL GitHub corretto da `alessandrolocurcio` a `Alessandro1040`
  (coerente con il remote reale della repo). 74 pp invariati.
- **Intervento precedente.** Pagina 16 (Sez. 5.1.3, box "Teorema" della CCV): il
  testo finale del box sforava di ~2 righe nella pagina successiva. Sistemato
  compattando la sola impaginazione, contenuto invariato: unificate in
  `gathered` le due display consecutive della definizione di $\mathcal{V}$ e
  del caso "con reinserimento" (elimina lo spazio tra display adiacenti), e
  `itemsep`/`topsep`/`parskip` nulli nell'`itemize` dentro il box Teorema.
  74 pp invariati (`main.pdf`, `tesi.pdf`, `tesi_finale.pdf`).
- **Intervento precedente.** Figura 5.3 (Sez. 5.1.2, `fig:batch_qualitativa`):
  sostituito il vecchio grafico (confronto con batch fisso) con lo screenshot
  reale del pannello "n_k vs a^k" dell'app, salvato in
  `tesi/figure_sim/batch_size_app.png` (catturato da `visualizzazione.html`).
  Didascalia aggiornata: $a$ è il parametro stimato per minimi quadrati in
  scala logaritmica, $\ln a = \sum_k k\ln n_k / \sum_k k^2$ — è ciò che
  `computeBestFitA` dell'app calcola davvero (il commento nel codice dice
  "minimizza $\sum_k(n_k-a^k)^2$" ma l'implementazione minimizza
  $\sum_k(\ln n_k - k\ln a)^2$). Confronto con il batch fisso rimosso
  (inutile). Nota: la Figura 5.7 (Esperimenti, `fig:batch`) usa ancora
  `figure/batch_size.pdf` con il batch fisso. Documento invariato a 74 pp
  (`main.pdf` e `tesi.pdf`), `tesi_finale.pdf` 74 pp.
- **Intervento precedente.** Dimostrazione del cono di discesa (Sez. 5.1.2, Fig. 5.2):
  reso esplicito il passaggio finale sull'angolo — "e infine, dall'identità
  fondamentale $\sin^2\varphi+\cos^2\varphi=1$, esprimiamo il seno in funzione
  del coseno, $\sin\varphi=\sqrt{1-\cos^2\varphi}$, per concludere
  $\sin\varphi \le \theta$" (prima era solo "scriviamo il seno in funzione del
  coseno", senza citare l'identità goniometrica né la formula). Nessun altro
  contenuto toccato; `bozza.tex` non allineata (la frase non è presente lì).
  Documento invariato a 74 pp (`main.pdf` e `tesi.pdf`), `tesi_finale.pdf`
  74 pp.
- **Intervento precedente.** Pagina Ringraziamenti (subito dopo la copertina):
  rimosso il `\vspace*{3.5cm}` che spingeva titolo e testo in basso. Ora
  "Ringraziamenti" e il `placeholder` partono subito sotto l'intestazione di
  testa, in modo uniforme con la pagina Abstract successiva (che inizia in
  cima). Nessun altro contenuto toccato; `bozza.tex` invariata (i
  ringraziamenti non ci sono). Documento invariato a 74 pp (`main.pdf` e
  `tesi.pdf`), `tesi_finale.pdf` 74 pp.
- **Intervento precedente.** Rimozione globale del grassetto (`\textbf`) e del
  corsivo (`\emph`) in tutto il documento, e dei trattini nei composti in prosa
  (mini-batch → mini batch, Hessian-free → Hessian free, ecc.); conservati i
  nomi tecnici (Newton-CG, Hessiana-vettore), i `\textit` dell'autore e la
  bibliografia. Sistemata la frase interrotta in Sez. 4.2 ("Gli autori ..."
  completata); aggiunto un `\newline` prima di "Poiché il gradiente della
  popolazione totale è ignoto" (Sez. 5.1.2); compattata la Nota CCV per far
  entrare la Nota "Quando siamo lontani dal minimo..." tutta in "pagina 17";
  Figura 5.3 ora usa la simulazione reale `figure_sim/batch_size.pdf` (nuova
  cartella `tesi/figure_sim/`). Appendice B riscritta: i listati Python
  (B.1–B.3) sono ora **spezzati in blocchi** separati da commenti descrittivi e
  brevi spiegazioni (stile della bozza). Documento passato da 71 a 74 pp;
  `bozza.pdf` allineata (56 pp invariata).
  prima pagina. Nuovo `tesi/frontespizio.tex` (sapthesis, con i dati: titolo,
  sottotitolo, autore, matricola, corso, relatore, A.A. 2025/2026, tipo "Tesi
  di Laurea Triennale"; l'opzione `lam` non esiste in questa versione di
  sapthesis) e script `tesi/compila_tesi.sh` che compila frontespizio +
  documento e produce `tesi_finale.pdf` (71 pp = frontespizio + documento
  senza la vecchia copertina custom, che resta nel sorgente).
  La pagina del verso non è generata (`\SAP@composebacktitlepage` svuotata):
  niente copyright, "Tesi non ancora discussa" né email dell'autore.
  Riformulati anche i "Contributi del lavoro" (e Abstract, Sez. 1.3 e
  Conclusioni) per attribuire i metodi a byrd2012 e descrivere i contributi
  propri: analisi teorica, miglioramento di alcuni teoremi di convergenza
  (fattore di contrazione via disuguaglianza di PL nella forma forte) e
  implementazioni. Inoltre: glossario reso minimale (solo simbolo +
  significato essenziale) e rimossi dai primi due capitoli i prerequisiti di
  matematica di base (definizione di convessità, gradiente, norma di matrice):
  documento a 71 pp.
- **Intervento precedente.** Snellite le Sezioni 1–2 (Introduzione, Background e
  Notazione) per adattarle a una tesi triennale: eliminati i paragrafi
  "A chi è rivolto" e "Il problema, in parole semplici" (ridondanti con
  l'Introduzione), deduplicato il discorso sul numero di condizionamento (ora un
  unico mnbox, con tasso di convergenza, dentro la definizione di norma di
  matrice), rimosse la definizione di varianza campionaria e le righe
  $\E$/$\Var$ del glossario (nozioni date per scontate), condensato il notebox
  su $\mathcal{V}$ e il paragrafo "Collegamento...". Documento passato da 75 a
  73 pp; allineata anche `bozza.tex` (56 pp invariati).
- **Intervento precedente.** Compattato lo pseudocodice Newton-CG (Sez. 5.2.3):
  condizioni di Wolfe in un unico display `gathered`, spaziature del box ridotte
  e `itemsep`/`topsep`/`parskip` nulli nelle enumerate; Figura 5.4 passata a
  `[H]` (prima con `[h]` finiva in fondo al documento). Lo pseudocodice ora sta
  interamente su una sola pagina (75 pp invariati; allineata anche `bozza.tex`,
  56 pp invariati).
- **PDF di riferimento.** `tesi_finale.pdf` 90 pp (definitiva, con
  frontespizio), `tesi.pdf` 90 pp (documento), `bozza.pdf` 56 pp.

## Riferimento

R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu, *Sample size selection in
optimization methods for machine learning*, Mathematical Programming, 2012.
