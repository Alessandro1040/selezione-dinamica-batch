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
    ├── tesi.tex                   documento principale
    ├── tesi.pdf                   PDF compilato del documento (senza frontespizio)
    ├── frontespizio.tex           frontespizio istituzionale Sapienza (sapthesis)
    ├── compila_tesi.sh            genera tesi_finale.pdf (frontespizio + tesi.pdf)
    ├── tesi_finale.pdf            PDF DEFINITIVO: frontespizio (2 pp) + contenuto
    ├── bozza.tex                  versione bozza/draft (senza Introduzione, ecc.)
    ├── bozza.pdf                  PDF compilato della bozza
    ├── contenuti/                 frammenti LaTeX dei capitoli (intro, lavori, sezione6, ...)
    ├── figure/                    figure usate nella tesi (PDF/PNG)
    ├── conodiscesa2.jpeg          figura del cono di discesa
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

Per compilare solo il documento (senza frontespizio):

```bash
cd tesi
latexmk -pdf -shell-escape tesi.tex    # serve pygments per i listati minted
```

Per generare il **PDF definitivo** con il frontespizio istituzionale come
prima pagina:

```bash
cd tesi
./compila_tesi.sh tesi    # compila frontespizio.tex + tesi.tex -> tesi_finale.pdf
```

`tesi_finale.pdf` è composto da `frontespizio.pdf` (sola pag. 1 = frontespizio;
la pagina del verso non è generata) più `tesi.pdf` **saltando** la vecchia
copertina custom (pag. 1 di `tesi.pdf`). La copertina custom resta nel
sorgente; per ripristinarla nel PDF basta rimuovere lo slicing
`body.pages[1:]` nello script. Nella copia di lavoro usare
`./compila_tesi.sh main`.

## Note operative e stato corrente (17/08/2026)

Da tenere presente nelle sessioni di lavoro successive:

- **File in lavorazione.** La copia di lavoro "definitiva" è
  `/Users/alessandrolocurcio/Downloads/tesi/main.tex`, che è **identica** a
  `tesi/tesi.tex` della repo (verifica: `md5 -q main.tex tesi/tesi.tex`). Una
  modifica va applicata a **entrambi** i file (o a `tesi.tex` e poi copiata su
  `main.tex`), e i PDF rigenerati.
- **Documento autocontenuto.** `tesi.tex`/`main.tex` NON usano `\input`: i
  frammenti in `contenuti/` sono già incorporati nel file. Non tentare di
  ricostruire il documento a partire da `contenuti/`.
- **Figura di copertina.** La compilazione di `main.tex` richiede
  `conodiscesa2.jpeg` anche in `/Users/alessandrolocurcio/Downloads/tesi/`
  (non solo in `tesi/`). Se manca, copiarlo dalla repo.
- **Compilazione.** `cd <dir> && latexmk -pdf -shell-escape <nome>.tex`.
  Compilare solo il documento modificato per risparmiare tempo. Su macOS
  `setsid` NON esiste: per lanciare in background usare
  `(nohup latexmk -pdf -shell-escape -interaction=nonstopmode <nome>.tex > /tmp/<nome>.log 2>&1 < /dev/null &)`.
- **Bozza.** `tesi/bozza.tex` (+ `bozza.pdf`) è la versione bozza: numerazione
  ed equazioni diverse. Allinearla solo se la modifica tocca contenuti presenti
  anche lì.
- **Ultimo intervento.** **Sez. 6 allineata ai preset dell'app
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
- **PDF di riferimento.** `tesi_finale.pdf` 72 pp (definitiva, con
  frontespizio), `tesi.pdf` 72 pp (documento), `main.pdf` 72 pp (copia di
  lavoro), `bozza.pdf` 56 pp.

## Riferimento

R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu, *Sample size selection in
optimization methods for machine learning*, Mathematical Programming, 2012.
