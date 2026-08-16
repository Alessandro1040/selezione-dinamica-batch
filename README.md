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

## Note operative e stato corrente (15/08/2026)

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
- **Ultimo intervento.** Rimossa la Figura 6.2 ("Evoluzione di $n_k$ rispetto a
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
- **PDF di riferimento.** `tesi_finale.pdf` 73 pp (definitiva, con
  frontespizio), `tesi.pdf` 73 pp (documento), `main.pdf` 73 pp (copia di
  lavoro), `bozza.pdf` 56 pp.

## Riferimento

R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu, *Sample size selection in
optimization methods for machine learning*, Mathematical Programming, 2012.
