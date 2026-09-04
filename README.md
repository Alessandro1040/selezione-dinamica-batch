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
- **Ultimo intervento (04/09/2026, schemi).** **Schemi a blocchi del Cap. 5
  ripuliti: niente variabili pendenti (R, $\sigma$, $\nu$, $Y_k$, $\alpha$).**
  (1) Fig. 5.5 (Newton-CG $L_1$): rimosso $R$ da "Inizializzazione" (in
  questa variante i campioni sono fissi, $|\mathcal{H}|=|\mathcal{H}_0|$ e
  $|\mathcal{S}|=|\mathcal{S}_0|$; $R$ non è un input, cfr. action box);
  il nodo subgradiente ora mostra
  $\widetilde{\nabla}F_{\mathcal{S}_k}(w_k)=\nabla J_{\mathcal{S}_k}(w_k)+\nu z_k$
  ($\nu$ resa esplicita); il nodo CG definisce $Y_k$ ("versori delle
  coordinate libere"); il nodo "Ricerca lineare proiettata" cita Armijo con
  $\sigma$. (2) Fig. 5.2 (Newton-CG): il nodo "Ricampionamento" ora mostra
  $\mathcal{H}_{k+1}\subseteq\mathcal{S}_{k+1}$ con
  $|\mathcal{H}_{k+1}|=R|\mathcal{S}_{k+1}|$ (prima $R$ compariva solo in
  "Inizializzazione"). (3) Fig. 5.6 (BB-CCV): aggiunto $\alpha$ a
  "Inizializzazione" (usato nella formula del passo BB). Verificati anche gli
  altri schemi (Dynamic GD fig. 5.1, SVRG/SAGA in Appendice): nessun altro
  simbolo pendente; $\mu$ non compare in nessuno schema del Cap. 5. Aggiunte
  le sorgenti standalone
  `presentazione/schemi/schema_{dynamic_gd,newton_cg,newton_l1,bbccv}.tex`
  (pattern di `schema_saga.tex`/`schema_svrg.tex`) e rigenerate le immagini
  `presentazione/immagini/schema_*.pdf` (4 file). Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**135 pp** — invariate; 0 errori, 0 undefined; 16
  overfull preesistenti, nessuno sui nodi modificati) e
  `presentazione/presentazione.pdf` (**27 pp**; 0 overfull). Sincronizzati in
  repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). Nessun altro file
  toccato.


- **Ultimo intervento (04/09/2026, follow-up).** **Presentazione, slide 9
  ("La varianza del gradiente stimato: definizioni"): aggiunta la definizione
  di $\mathcal{V}$, prima assente.** La slide definiva solo la stima
  campionaria $\widehat{\mathcal{V}}$ (``v-hat''), mentre $\mathcal{V}$
  compariva senza definizione (slide 11: $\mathrm{Var}(g_k)=\mathcal{V}/n_k$,
  ``$\mathcal{V}$ (ignota) è stimata da $\widehat{\mathcal{V}}$''). Ora il
  bullet ``Definizioni'' riporta entrambe, come nella tesi (notebox Sez.~5.1):
  $\mathcal{V}:=\frac{1}{N}\sum_{i=1}^{N}(\nabla\ell(w_k;i)-\nabla
  J(w_k))^2\in\mathbb{R}^m$ è la *varianza della popolazione* del gradiente
  della loss, componente per componente (non calcolabile: $\nabla J(w_k)$ è
  ignoto), e $\widehat{\mathcal{V}}:=\frac{1}{n_k-1}\sum_{i\in\mathcal{S}_k}
  (\nabla\ell(w_k;i)-g_k)^2\in\mathbb{R}^m$ ne è la stima campionaria
  non distorta, con esplicito $\mathbb{E}[\widehat{\mathcal{V}}]=\mathcal{V}$
  (Bessel). Verificata anche la condizione di arresto di Newton-CG $L_1$
  nell'implementazione (`altro/script/gen_tabelle_riuso.py`, identica a quella
  generata dall'app e al listato in Appendice B): **non è**
  $\|\nabla f_{\mathcal{S}_k}(w_k)\|<tol$ — i criteri reali sono (i)
  $\|\widetilde{\nabla}F_{\mathcal{S}_k}(w_k)\|_2<10^{-10}$ (subgradiente di
  $F=J+\nu\|w\|_1$ sul batch, soglia fissa non parametrica, a inizio
  iterazione) e (ii) $\|\nabla J(w_{k+1})\|_2<10^{-6}$ (gradiente esatto su
  tutto il dataset, a fine iterazione); lo schema Fig.~5.5 riporta
  correttamente la versione (i) ($\|\widetilde{\nabla}F_{\mathcal{S}_k}\|<
  \mathrm{tol}$) e nessuna slide ne dà la versione ``semplice''. Ricompilato
  `presentazione/presentazione.pdf` (**27 pp** — invariate; 0 errori, 0
  undefined, 0 overfull). Nessun altro file toccato: la presentazione vive
  solo in repo, la copia Desktop/tesi non è coinvolta.

- **Ultimo intervento (04/09/2026).** **Presentazione: aggiunta la definizione
  di $\widehat{\mathcal{V}}$ (``v-hat'') e completata la spiegazione del perché
  la varianza di $g_k$ si stima con $\|\widehat{\mathcal{V}}\|_1/n_k$.**
  Prima la slide CCV usava $\widehat{\mathcal{V}}$ senza definirlo e la slide
  ``La CCV garantisce una direzione di discesa'' dava la stima
  $\|\widehat{\mathcal{V}}\|_1/n_k$ senza giustificarla. Inserita una nuova
  slide 9 (``La varianza del gradiente stimato: definizioni''):
  $g_k$ è la media campionaria non distorta ($\mathbb{E}[g_k]=\nabla J(w_k)$),
  la varianza dello stimatore è $\mathbb{E}[\|e_k\|_2^2]=\|\mathrm{Var}(g_k)\|_1$
  e
  $\widehat{\mathcal{V}}:=\frac{1}{n_k-1}\sum_{i\in\mathcal{S}_k}(\nabla\ell(w_k;i)-g_k)^2\in\mathbb{R}^m$
  (quadrato per componente; centrata sulla media campionaria $g_k$ perché
  $\nabla J(w_k)$ è ignoto; $n_k-1$ di Bessel). Slide CCV (ora 10) alleggerita:
  definizione di $g_k$ rimossa dalla formula (già data a slide 9) e bullet del
  criterio riformulato (``varianza stimata dello stimatore''). Slide 11:
  completata la frase sulla stima — $g_k$ è media di $n_k$ termini indipendenti
  $\Rightarrow\mathrm{Var}(g_k)=\mathcal{V}/n_k$, con $\mathcal{V}$ (ignota)
  stimata da $\widehat{\mathcal{V}}$ (slide 9). Commenti SLIDE rinumerati
  (da 24 a 25 slide). Ricompilato `presentazione/presentazione.pdf` (0 errori,
  0 undefined, 0 overfull). Nessun altro file toccato: la presentazione vive
  solo in repo, la copia Desktop/tesi non è coinvolta.

- **Ultimo intervento (04/09/2026).** **Aggiunto il preset non quadratico
  ``banana di Rosenbrock'' ($c=100$, rumore $\sigma=0.2$) come stress test per
  i quattro algoritmi.**
  (1) `visualizzazione.html`: nuovo preset `banana` in `LOSS_PRESETS` (`code` e
  `code_autodiff`, dataset stocastico centrato, minimo $w_*$ calcolato
  numericamente come nei preset 1D), opzione nel dropdown, chiave nelle liste
  suggerite/di default del Test batch generico; i formati speciali che
  riproducono le tabelle della tesi restano sui 4 preset quadratici.
  (2) `altro/script/riproduci_tutte_le_tabelle.py`: aggiunti
  `_make_preset_banana`, il generatore `gen_banana_test_tables` (tabella
  `tab:test_banana`, seed 42) e la verifica `--verify` della nuova tabella;
  gli output di tutte le tabelle esistenti non cambiano.
  (3) Tesi (Setup e Risultati Numerici): la banana è aggiunta all'elenco dei
  problemi test con la formula e la costruzione del dataset, e sono riportati
  i risultati (Tabella errore per iterazione, seed 42). Valori generati dallo
  script (metodologia seed→dataset→esecuzione): Dynamic GD
  $e_{30}\approx6.1\times10^{-4}$, BB-CCV $\approx1.1\times10^{-1}$ (mediana su
  5 seed $5.6\times10^{-9}$), Newton-CG $\approx2.1\times10^{-1}$ (mediana
  $6.8\times10^{-1}$), Newton-CG $L_1$ $\approx1.3\times10^{-1}$ (mediana
  $2.0\times10^{-1}$). Ricompilati `tesi.pdf` e `tesi_finale.pdf` e
  sincronizzati in repo (md5 verificati).

- **Ultimo intervento (04/09/2026, ringraziamenti).** Rimosso ``Sik World''
  dai ringraziamenti visibili della pagina Ringraziamenti di `tesi.tex` e
  spostato in commento (lista ``mantenuti come commento'', con data di
  rimozione). Ricompilati `tesi.pdf` e `tesi_finale.pdf`; sincronizzati in repo
  (md5 verificati). Nessun'altra modifica: nessuna tabella numerica coinvolta
  (nessuna riga dello script di riproduzione cambiata).
- **Ultimo intervento (03/09/2026).** **Script di riproduzione
  `riproduci_tutte_le_tabelle.py` allineato a `tesi.tex` su tutte le tabelle
  del Capitolo 6.** (1) `gen_cons_tables` ora genera anche le 4 tabelle
  per-iterazione "base vs consigliato" di Newton-CG
  (`tab:riuso_cons_{bencond,malcond,veryill,offdiag}_ncg`): la colonna
  consigliata è lo stop adattivo con validation set (P=1, f=1, p=10%, split
  fissa, seed 42), finora presente in `tesi.tex` ma non emessa dallo script.
  (2) Rimossa da `gen_descent_confronto` la riga "Media (16 casi)": la Tabella
  6.25 della tesi non la contiene (le 80 celle coincidevano già).
  (3) `--verify` esteso alle 4 tabelle Newton-CG (29 → 33 tabelle controllate;
  **3951/3951 celle OK, 0 fuori**). Validazione completa: `--tex` ora emette
  tutte le **41** tabelle numeriche del Capitolo 6 e il confronto
  cella-per-cella contro `tesi.tex` non ha differenze (resta esclusa solo la
  Tabella 5.1, analitica, non un insieme di dati). Docstring dello script
  aggiornata. Nessun PDF coinvolto (`bozza.tex` non toccata).
- **Ultimo intervento (02/09/2026, follow-up 3).** **Tesi: rimossa la sezione
  NSynth.** Eliminate da `tesi.tex` le sottosezioni 6.5 "Riconoscimento Note"
  e 6.6 "Riconoscimento Strumenti" (erano `(PLACEHOLDER)`, contenuto mai
  mergiato da `sezioni_65_66.tex`) e il `\bibitem{engel2017}` (mai citato,
  relativo solo a NSynth). Nessun riferimento pendente nel testo (verificato:
  nessun `\ref{sec:nsynth*}` né menzioni "NSynth"/"riconoscimento"). La
  sottosezione "Riuso del mini-batch" passa da 6.7 a **6.5** (rinumerazione
  automatica, riferimenti `\ref{app:riuso*}` aggiornati). Ricompilati
  `tesi.pdf` e `tesi_finale.pdf` (**132 pp**; 0 errori, 0 undefined; 13
  overfull, invariati). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). Il materiale NSynth (repo e Desktop:
  `tesi/nsynth/`, `tesi/figure_nsynth_*`, `colab_risorse/`, `sezioni_65_66.tex`)
  NON è stato cancellato: resta come materiale non più citato, in attesa di
  eventuale pulizia. `bozza.tex` non toccata.
- **Ultimo intervento (02/09/2026, follow-up 2).** **Tesi: refuso corretto in
  Sez. 5.2 (dimostrazione, caso $\alpha a > 1$).** "perche è un o piccolo
  rispetto ad $\alpha^k$" → "perché è un $o$-piccolo rispetto ad $\alpha^k$"
  (accento mancante su *perché* e notazione *o*-piccolo messa in math),
  allineato alla frase gemella del caso $\alpha a < 1$. Ricompilati
  `tesi.pdf` e `tesi_finale.pdf` (**133 pp** — invariate; 0 errori, 0
  undefined; 13 overfull, invariati). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (02/09/2026, follow-up).** **Tesi Appendice A.2
  (dimostrazione del fattore di correzione per popolazione finita): corretta
  una parentesi "chiusa ma non aperta".** Nel passaggio sull'espansione del
  quadrato della somma una parentesi testuale `(e questa servirà anche dopo,
  vera perché: …)` avvolgeva un'equazione scritta come math inline su riga
  propria: nel PDF la formula andava a capo e la `)` di chiusura restava
  visivamente orfana all'inizio della riga successiva. Riscritto il passaggio:
  l'identità `(∑a_i)² = ∑a_i² + ∑_{i≠ℓ} a_i a_ℓ` (con `a_i = X_i(I_i−E[I_i])`)
  è ora una vera equazione display, la spiegazione "vera perché" è separata in
  prosa con la propria equazione display, e la parentesi "a ponte" è stata
  eliminata. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**133 pp** — +1
  rispetto alle 132: le due equazioni in display aggiungono spazio e fanno
  slittare la paginazione dall'Appendice A in poi; 0 errori, 0 undefined; 13
  overfull, invariati). Passaggio corretto a pag. 105 stampata (106 nel
  viewer). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5
  verificati). `bozza.tex` non toccata.
- **Ultimo intervento (02/09/2026).** **Tesi: revisione stilistica — rimossi i
  residui da bozza segnalati come "troppo AI / poco professionali".** (1)
  Appendice C (fine Sez. C.2 Newton-CG): rimossa la nota in chiaro e
  sottolineata *"L'algoritmo Newton CG è sostanzialmente Dynamic GD ma la
  direzione di discesa è quella di Newton anziche l'antigradiente"* (con il
  refuso "anziche"); sostituita da un paragrafo professionale che spiega che il
  Newton-CG condivide con il Dynamic GD la regolazione dinamica del batch
  (CCV) e che la differenza è la direzione di ricerca, data dalla direzione di
  Newton approssimata (sistema \eqref{eq:5_31} risolto con il CG Hessian free).
  (2) Sez. 5.1 (nota "Regola di aggiornamento del batch"): l'enfasi
  sottolineata *"Il batch cresce automaticamente dove serve precisione"* è ora
  in corsivo (`\emph`), niente più `\underline` nel documento. (3) Rimossi i
  commenti LaTeX duplicati "Figura 5.1/5.2" (tre copie consecutive → una). (4)
  Sez. 5.1.2 (dimostrazione del cono di discesa): spezzata in passi separati la
  frase unica che descriveva i passaggi della dimostrazione, senza cambiarne il
  contenuto tecnico. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**132 pp** —
  invariate; 0 errori, 0 undefined; 13 overfull, invariati). Sincronizzati
  nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex`
  non toccata. Restano volutamente invariate (in attesa di decisione): il merge
  delle Sezioni 6.5/6.6 (ancora `(PLACEHOLDER)` in tesi.tex, contenuto in
  `sezioni_65_66.tex`), l'uso di "il paper", le ripetizioni di "In sintesi" e i
  ringraziamenti con la lista degli artisti.
- **Ultimo intervento (31/08/2026, follow-up 6).** **Tesi: resi ipertestuali
  TUTTI i riferimenti interni (cliccabili, stessa impaginazione).** (1)
  Aggiunte le `\label` mancanti alle 93 equazioni numerate manualmente
  (`\tag{...}`) che non ne avevano (mantenute le 4 già presenti
  `eq:discesa`, `eq:var_bound`, `eq:bound_gk`, `eq:cg_l1`): ora tutte le 97
  equazioni con `\tag` (Capitolo 5 e Appendice A) hanno un'etichetta
  (`eq:5_2`...`eq:A_40`). (2) Sostituiti i 99 riferimenti letterali
  `(5.x)`/`(A.x)` con `\eqref{...}` (in Sez. 5.1-5.4, tabelle, Sez. 6.7 e
  nelle dimostrazioni dell'Appendice A). (3) Sostituiti i 5 riferimenti
  letterali a sezioni/appendici: `sez.~5.1.2` → `\ref{sec:condizione-accettazione}`
  e `sez.~5.2.2` → `\ref{sec:criterio-cg}` (etichette aggiunte alle
  sottosezioni 5.1.2 e 5.2.2, che non le avevano), `Sezione~6` →
  `\ref{sec:esperimenti}`, `Appendice~B` → `\ref{app:codici}`, `Appendici~A--D`
  → `\ref{app:dimostrazioni}--\ref{app:schemi}`. Lasciati invariati i
  riferimenti esterni (eq.~4.9 e eq.~4.4--4.5 di Nocedal--Wright) e il valore
  numerico `(1.4142)`. Inclusa anche la modifica pendente del 31/08
  (rimossa in Sez. 6.2 la parentesi "(nel paper il sample size per il caso
  $L_1$ è assunto fisso, e gli autori lasciano l'estensione al caso dinamico
  come lavoro futuro)"). Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**132 pp** — invariate; 0 errori, 0 undefined; 13 overfull, invariati).
  Verificato nel PDF: link interni da 323 a 428 (0 `??`). Sincronizzati nella
  repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.

- **Ultimo intervento (31/08/2026, follow-up 5).** **Ringraziamenti: Pop Smoke,
  Millyz e Free Flow Flava spostati nei commenti LaTeX (non compaiono nel PDF);
  aggiunto Oiki tra gli artisti.** Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**132 pp** — invariate; 0 errori, 0 undefined; 13 overfull, invariati).
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.
- **Ultimo intervento (31/08/2026, follow-up 4).** **Tesi: i trattini parentetici
  (rendering ` -- `) sono stati sostituiti con virgole** in 7 passaggi:
  Introduzione ("I metodi qui analizzati, gradiente a campione dinamico, …,
  $L_1$, sono stati introdotti da Byrd…"), Sez. 2 sull'effetto della $L_1$
  ("dall'altro, ed è la proprietà più importante, la soluzione risulta
  \emph{sparsa}"), Sez. 6.2 (due passaggi: "…implementati nell'applicazione,
  Dynamic GD, …, BB-CCV (Sezione bbccv), eseguiti…" e "…per i quattro
  algoritmi, Dynamic GD, …, sui tre problemi test"), Sez. 6.7 ("…ciascuno dei
  quattro algoritmi, Dynamic GD, …, BB-CCV, si confrontano…"), Sez. 6.7.8
  ("…su tutti i problemi, il metodo converge già in poche iterazioni…, e
  quindi la base resta…") e Conclusioni ("La complessità totale, il numero di
  valutazioni di gradienti…, è…"). Lasciati invariati: nomi tecnici
  (Newton-CG, Hessiana-vettore, Barzilai--Borwein), intervalli
  (2025--2026, riferimenti tab--tab), comandi TikZ (`--`) e i trattini
  singoli negli item di Appendice C (pacchetti). Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**132 pp** — invariate; 0 errori, 0 undefined; 13
  overfull, invariati). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (31/08/2026, follow-up 3).** **Ringraziamenti: Emis
  Killa, Jitta On The Track e Denzel Curry spostati nei commenti LaTeX (non
  compaiono più nel PDF), Jason Derulo rimosso del tutto; aggiunti tra gli
  artisti Massimo Pericolo e Bassi Maestro.** Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**132 pp** — invariate; 0 errori, 0 undefined; 13
  overfull, invariati). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (31/08/2026, follow-up 2).** **Tesi Sez. 5.3.3
  (Identificazione dell'active set e della faccia ortante): aggiunto un
  elenco itemize subito dopo la (5.45)** che esplicita la regola con cui
  l'algoritmo sceglie $z_k^i$ basandosi sulla condizione di stazionarietà:
  (1) $w_i > 0$ → resta a destra; (2) $w_i < 0$ → resta a sinistra;
  (3) $w_i = 0$ e il gradiente tira a destra con forza maggiore di $\nu$ →
  la libera a destra; (4) $w_i = 0$ e il gradiente tira a sinistra con forza
  maggiore di $\nu$ → la libera a sinistra; (5) $w_i = 0$ e il gradiente è
  debole ($|\partial J/\partial w_i| \le \nu$) → la blocca a zero; chiusura
  "Quindi $z_i$ è la decisione dell'algoritmo su dove la variabile deve
  trovarsi". Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**132 pp** — l'elenco
  aggiunge una pagina; 0 errori, 0 undefined; 13 overfull, invariati).
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.
- **Ultimo intervento (31/08/2026, follow-up).** **App web: il pulsante
  "🖼️ Traiettoria 2D" ora compare SOLO quando la loss corrente è 2D.** La
  visibilità è determinata dalla dichiarazione `DIM = 1`/`DIM = 2` nel codice
  corrente di `lossCode` (la stessa regex di `runAlgorithm`/
  `renderSurfacePreview`; `DIM` assente → assunto 2D, comportamento
  preesistente). Il blocco (pulsante + hint) viene nascosto con la classe
  `hidden` per i preset 1D (`1d_quad`, `1d_quartic`, `1d_sin`, `1d_exp`) e per
  la custom con `DIM = 1`; mostrato per i preset 2D e la custom con `DIM = 2`
  (o senza `DIM`). L'aggiornamento avviene a ogni cambio preset (in
  `loadLossPreset`) e a ogni modifica del textarea del codice custom
  (`lossCode` → listener `input`). Validazione: harness Deno (stub DOM) —
  13/13 controlli (tutti i preset 1D/2D + custom con/senza `DIM`). Nessun PDF
  coinvolto; `bozza.tex` non toccata.
- **Ultimo intervento (31/08/2026).** **Test batch di `visualizzazione.html`
  esteso: ora genera TUTTE le tabelle e30 della Sez. 6.7 (Tabelle 6.20-6.29).**
  Restano fuori solo le tabelle per-iterazione (6.1-6.3, 6.4-6.19, 6.30-6.41),
  perché il batch restituisce statistiche di sintesi e non l'errore a ogni $k$.
  Ai formati esistenti (Matrice = Tabb. 6.20/6.22/6.25; Robustezza = Tabb.
  6.24/6.27) si aggiungono **5 formati dedicati** che riproducono le tabelle
  mancanti: **Robustezza riuso Migl./Pegg./Ug.** (stile Tab. 6.21, conteggio
  per-seed su 5 seed per $M{=}\infty$ e $M{=}10$), **Sensibilità iperparametri
  (validation)** (stile Tab. 6.23, griglia 3×3×3×2×2=108 hp su 16 caselle),
  **Sensibilità iperparametri (discesa)** (stile Tab. 6.26, griglia 3×3×2=18
  hp), **Confronto finale** (stile Tab. 6.28: base / $M{=}\infty$ / stop
  adattivo calibrato $P{=}1,p{=}0.1$,dyn / discesa calibrata $\tau{=}10^{-3}$,
  su 16 caselle × 5 seed) e **Sintesi consigliati** (stile Tab. 6.29: mediana
  di $e_{30}$ su 5 seed + vitt. per-seed, consigliata per-algoritmo: riuso
  $M{=}5$ per GD, \emph{base} per BB, stop adattivo $P{=}1,f{=}1,p{=}10\%$,
  split fissa per Newton-CG, riuso $M{=}3$ per Newton-L1). Implementazione: i
  runner dedicati costruiscono i cfg ed eseguono lo STESSO codice generato
  dall'app (`batchBuildFullCode`), quindi i valori coincidono con le altre run
  a meno dei soli errori numerici (runtime Pyodide/WASM nel browser vs NumPy
  di sistema); usano le dimensioni loss/algo della finestra e gli altri
  parametri ai default (seed 42, max_iter 30, alpha 0.1, θ 0.5, batch0 5,
  N 200, w0 (2,-3)). Run totali coi default: rob_riuso 240, sens_val 1728,
  sens_desc 288, confronto 400, consigliati 140. Ogni formato ha export
  LaTeX (label `tab:batch_test_*`) e JSON. **Tesi Sez. 5.2.3 (passo CG):
  aggiunto il collegamento ipertestuale alla dimostrazione in Appendice**
  ("Dimostrazione nell'appendice" → "Dimostrazione nell'Appendice~\ref{app:cg}",
  il `\ref` è un link con hyperref). Validazione: harness Deno (stub DOM) —
  sintassi JS ok, 36/36 controlli (parsing strategie, generazione codice,
  aggregazioni, renderer), export LaTeX compilati con latexmk (0 errori).
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**131 pp**, 0 errori, 0
  undefined; 13 overfull, invariati). Sincronizzati nella repo
  visualizzazione.html + tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.
- **Ultimo intervento (29/08/2026, follow-up 3).** **Tesi: Ringraziamenti estesi
  con l'elenco completo dei nomi (60 nomi, tra cui relatori, colleghi e amici).**
  Ampliata la pagina dei ringraziamenti con tutti i nomi: Giovanni Adelfio,
  Alessandro Pisarra, Lorenzo Salis, Alessandro Cattaneo, Alessio Projetti,
  Alessandra Melchionna, Mia Rodotà, Mariastella Gioia La Rocca, Azzurra
  Giordano, Rosamaria Graziosi, Natalia Pasquetto, Antonietta Todisco, Marco
  Galletti, Francesco Mondelli, Luca Montefusco, Sofia De Angelis, Cristina
  Pesci, Alberto Floris, Alessandro Centomini, Federico Carattoli, Elena
  Mannoni, Emmanuel Nsia, Ilenia Filippi, Giada Manfredi, Gaia Facioni,
  Gabriele Cirillo, Mario Prignano, Michele Aliffi, Michele Fazio, Mirko
  Bonifazi, Luca Veneri, Matteo Pinato, Mariagiusi Nicodemo, Luca Cerovaz,
  Matteo Sacripante, Aurora Di Giovanna, Giacomo Tronca, Alessandro Pirisinu,
  Francesco Terracciano, Marco dell'Oste, Luigi Capini, Luca Massimo Andrea
  Martinazzi, Claudia Malvenuto, Paolo Gaspare Bottoni, Giovanni Trappolini,
  Federico Fusco, Giulio D'Agostini, Grant Sanderson, Steve Chow, Francesco
  Zamponi, Emanuele Caglioti, Gabriella Puppo, Antonella Poggi, Federica
  Baccini, Marco Sciandrone, Fabrizio Silvestri, Marco Isopi, Elena Agliari,
  Davide Torlo. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (131 pp, 0 errori,
  0 undefined; 13 overfull, invariati). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (30/08/2026, follow-up).** **Script unico con il codice
  ESATTO di `visualizzazione.html`; Tabella 6.2 corretta.** (1)
  `altro/script/riproduci_tutte_le_tabelle.py` ora incorpora il codice Python
  generato dall'app **riga per riga**: i 18 codici (base, riuso $M$ con H
  legata/indipendente, stop adattivo con validation, riuso per discesa × 4
  algoritmi) sono estratti con il nuovo harness Deno
  `altro/script/estrae_codice_tutte.mjs` in `altro/script/codice_generato/` e
  vengono eseguiti dallo script con `exec()` in un namespace pulito, come
  l'helper `_batch_run` dell'app: ogni tabella è quindi riproducibile anche
  con l'applicazione, a meno dei soli errori numerici (build WASM Pyodide vs
  NumPy di sistema). (2) **Corretta la Tabella 6.2 della tesi**
  (`tab:test_malcond`, κ≈20): conteneva per tutti i metodi i valori del
  problema molto mal condizionato (κ≈100); rigenerata con lo script (valori
  corretti: GD 4.91e-2, BB-CCV 9.72e-5, Newton-CG 2.35e-1, Newton-CG L1
  4.88e-1) e corretta anche la relativa nota *Lettura*. (3) `--verify` esteso:
  verifica ora anche la robustezza 6.21 (conteggi interi) e i ricampionamenti
  delle Tabelle 6.22/6.25. **Validazione finale: tutte le tabelle 6.1–6.40
  riprodotte** (nessuna eccezione). Sostituisce i generatori storici
  (`riproduci_tabelle.py`, `gen_tabelle_riuso*.py`, conservati come
  riferimento). Modalità: `--tex`, `--verify tesi/tesi.tex`, `--summary`,
  `--json`. Aggiornata la Sez. 6.2.1 della tesi (riferimento allo script per
  nome). Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**131 pp**, 0 errori, 0
  undefined; 13 overfull, invariati). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf + `altro/script/riproduci_tutte_le_tabelle.py`
  + `altro/script/estrae_codice_tutte.mjs` + `altro/script/codice_generato/`
  (md5 verificati). `bozza.tex` non toccata.



- **Ultimo intervento (29/08/2026, follow-up 2).** **Tesi: Ringraziamenti
  compilati con i nomi reali.** Sostituito il placeholder `\emph{(placeholder)}`
  con l'elenco dei ringraziamenti (Giovanni Adelfio, Alessandro Pisarra,
  Lorenzo Salis, Alessandro Cattaneo, Alessio Projetti, Alessandra Melchionna,
  Mia Rodotà, Alessandro Pirisinu, Francesco Terracciano, Marco dell'Oste,
  Luigi Capini, Luca Massimo Andrea Martinazzi, Claudia Malvenuto, Paolo
  Gaspare Bottoni, Giovanni Trappolini). Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (131 pp, 0 errori, 0 undefined; 13 overfull, invariati).
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.


- **Ultimo intervento (29/08/2026, follow-up).** **Tesi Appendice C: corretta
  l'incoerenza sull'esecuzione degli esperimenti e rimosso ogni riferimento a
  `sim_exp.py`.** (1) C.3 (Esecuzione degli esperimenti numerici): riscritto il
  paragrafo -- gli esperimenti del Capitolo 6 sono stati eseguiti sul computer
  dell'autore con le implementazioni Python dell'Appendice B e seed 42;
  l'applicazione web permette di ripeterli nel browser ma il runtime Pyodide
  (build WASM) può introdurre differenze dell'ordine di qualche punto
  percentuale sui problemi mal condizionati rispetto al NumPy di sistema
  (coerente con la Sez. 6.2.1). Rimosso lo script `sim_exp.py`, non più
  presente nel repository. (2) C.4 (Esecuzione dei singoli algoritmi): rimossa
  la frase sull'esempio di regressione lineare incluso in `sim_exp.py`.
  (3) C.5 (Avvio dell'applicazione web): rimossa la frase sul server HTTP
  statico (`python3 -m http.server 8000` e `http://localhost:8000`); l'app è un
  singolo file HTML apribile direttamente nel browser. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (131 pp, 0 errori, 0 undefined; 13 overfull, invariati).
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.


- **Ultimo intervento (29/08/2026).** **Tesi: Appendice B.3 ampliata (caso m=1 di
  Newton-L1), paragrafo BB-CCV in B.4 aggiornato con riferimento alla Sez. 5.4.1,
  nuova dimostrazione in Appendice A (`app:rayleigh`), rimossa la tabella
  `tab:bb_confronto`.** (1) Appendice B.3 (`newton_l1`): aggiunta la presentazione
  del caso specifico $m=1$ con loss $\ell(x;i)=(x-a_i)^2$, $a_i=1+\varepsilon_i$, e
  dopo ogni blocco di codice un paragrafo ``Nel nostro caso:'' che concretizza
  subgradiente, vettore $z$ (faccia ortante), direzione di Newton, ricerca lineare
  proiettata e CCV in 1D. (2) Sez. 5.4.1: rimossa la tabella ``Il passo di
  Barzilai--Borwein sulle funzioni quadratiche e generali'' (`tab:bb_confronto`),
  ridondante con la prosa. (3) Appendice B.4: il paragrafo sul passo BB ora rimanda
  alla Sez. 5.4.1 (relazione fondamentale, Eq. (5.54)) e precisa che
  $s^\top H s/s^\top s$ per matrice simmetrica sta tra $\lambda_{\min}$ e
  $\lambda_{\max}$, con collegamento ipertestuale alla nuova dimostrazione.
  (4) Nuova sottosezione in Appendice A (`app:rayleigh`): dimostrazione che
  $s^\top H s/s^\top s$ è una media pesata (combinazione convessa) degli autovalori
  di $H$, con pesi $c_i^2/\sum_j c_j^2$, quindi sempre tra $\lambda_{\min}$ e
  $\lambda_{\max}$; chiusura sul passo BB. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (131 pp, 0 errori, 0 undefined; 13 overfull, nessuno nelle
  nuove sezioni). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5
  verificati). `bozza.tex` non toccata.


- **Ultimo intervento (29/08/2026).** **App web: vista dell'obiettivo
  regolarizzato $F(w)=J(w)+\nu\|w\|_1$ per il metodo Newton-CG-$L_1$.** In
  `visualizzazione.html` (file solo nella repo, nessun PDF coinvolto) aggiunto
  nel pannello del metodo Newton-CG-L1 la checkbox **"Mostra anche
  F(w) = J(w) + ν‖w‖₁ (obiettivo regolarizzato)"** (default attiva). Con la
  checkbox attiva e metodo $L_1$ selezionato: (1) nel **plot principale 2D**
  oltre alla superficie di $J$ (blu) viene disegnata la **superficie di $F$
  (oro, semi-trasparente)** con il kink della norma lungo gli assi e la
  **traiettoria proiettata su $F$** (linea oro tratteggiata, monotona perché
  l'Armijo è su $F$), titolo con `J` e `F` correnti e asse `J(w) / F(w)`;
  (2) nel **plot 1D** la curva di $F$ tratteggiata in oro accanto a $J$;
  (3) nella **vista Colab (curve di livello)** i contorni di $F$ in oro
  tratteggiati con **livelli geometrici propri** (`_levelsF`, calcolati come
  quelli di $J$ ma sul range di $F$); (4) la **caption** della galleria e la
  riga di stato riportano `mostra anche F=J+ν‖w‖₁` / `vista F=J+ν‖w‖₁
  attiva`. Con checkbox spenta o metodo diverso da $L_1$ tutto resta identico
  a prima (superficie di $J$, `surfF` vuoto, nessuna traccia $F$). Motivo:
  il metodo minimizza $F$, non $J$: senza vedere $F$ la traiettoria $L_1$
  sembra fermarsi lontano dal minimo di $J$ (in realtà converge al minimo di
  $F$, che per $\nu$ sufficiente sta sull'asse) e $J$ può anche salire da
  un'iterazione all'altra. Coerente con la Sez. 4.4/5.3 della tesi e con
  l'animazione Manim (che già disegna $F$). Validazione: harness deno con DOM
  fittizio + codice Python generato eseguito con numpy reale — scenari
  newton_l1 2D toggle on (9/9 controlli: `surfF.z` 100×100, `pts_F =
  J+ν‖w‖₁` esatto, `levelsF` 10 livelli crescenti, traccia $F$ nel plot
  principale, contorno $F$ presente, caption cita $F$), 2D toggle off (8/8:
  `surfF` vuoto, nessuna traccia $F$, caption senza $F$), 1D toggle on (9/9:
  curva $F$ nel plot principale), più GD/Newton-CG/BB toggle off (8/8);
  verificato anche il **regrid** della vista Colab (`_grid2d` ora restituisce
  6 valori: `zF=log10 F`, `JF=F`, `levelsF` corretti). Nota: i preset 1D non
  definiscono `N` nel codice (limitazione preesistente): per provare i metodi
  di Newton sul 1D serve la modalità "Autodiff numerica". `bozza.tex` non
  toccata.
- **Ultimo intervento (29/08/2026, follow-up).** **App web: vista $F$ completata
  per il plot 1D — traiettoria su $F$, punto corrente su $F$ e stella sul minimo
  di $F$.** Nel plot principale 1D (e nella vista Colab 1D) con la checkbox
  "Mostra anche F" attiva ora vengono disegnati anche: la **traiettoria su $F$**
  (linea oro, che per il metodo $L_1$ è monotona perché l'Armijo è su $F$), il
  **marker del punto corrente su $F$** (rombo oro) e una **stella sul minimo di
  $F$** (dal minimo della griglia). Motivo: in 1D la valle di $F$ può essere
  invisibile rispetto al range dell'asse y (es. preset `1d_exp` con
  $\nu=0.991$: il minimo di $F$ è a $w\approx0.251$ con $F=0.5725$, ma $F(0)
  =0.6487$ e la curva $F$ sul plot arriva a $\sim9$: la differenza di $0.08$
  non si vede), e prima la traiettoria era disegnata solo su $J$: sembrava che
  l'algoritmo si fermasse "prima" del minimo di $F$. Ora la traiettoria su $F$
  finisce esattamente sulla stella del minimo di $F$. Verificato con harness
  deno + numpy reale su `1d_exp` $\nu=0.991$, w₀=2: la traiettoria converge a
  $w=0.25125$ = minimo globale di $F$ (subgradiente di $F$ $\approx1.6\times
  10^{-10}$), che coincide con i valori della tabella di convergenza dell'app
  (errs $0.74875$, $J=0.32354$). Il minimo a $w=1$ è quello di $J$ (curva
  blu), non di $F$: $F(1)=0.991 > F(0.251)=0.5725$ perché la penale $\nu=0.991$
  sposta il minimo di $F$ verso lo zero. **Nota diagnostica:** la tabella di
  convergenza e la metrica `errs = ‖w−w*‖` usano $w^*=$ minimo di $J$; per il
  metodo $L_1$ il punto stazionario corretto è il minimo di $F$, quindi `errs`
  non va a 0 a convergenza (è la distanza dal minimo di $J$, non di $F$). La
  metrica corretta per $L_1$ è la norma del subgradiente di $F$ (qui $\sim
  10^{-10}$) o il gap $F(w)-F^*$. Nessun PDF coinvolto; `bozza.tex` non toccata.
- **Ultimo intervento (29/08/2026, follow-up 2).** **App web: guida verticale
  sul minimo di $F$ nel plot 1D.** Nella vista $F$ (checkbox "Mostra anche F")
  del plot 1D (principale e Colab) aggiunta una **linea verticale tratteggiata
  dorata** in corrispondenza del minimo della curva $F$ (dal minimo della
  griglia), oltre alla stella "min F(w)", alla traiettoria su $F$ e al marker
  del punto corrente su $F$. Verifica numerica (preset `1d_exp`, $\nu=0.991$,
  w₀=2): la linea $F$ è esattamente $F(w)=J(w)+\nu|w|$ (formula verificata sui
  dati del plot) e il suo minimo è a $w=0.25125$, che coincide con la
  convergenza di Newton-L1 (subgradiente di $F\approx1.6\times10^{-10}$).
  Nota: con le 30 iterazioni di default la traiettoria può fermarsi prima del
  minimo (es. $w\approx0.31$ vs $0.251$ sul preset `1d_exp`): non è la linea a
  essere sbagliata, è l'algoritmo non ancora convergente (aumentando le
  iterazioni raggiunge il minimo della linea). Nessun PDF coinvolto;
  `bozza.tex` non toccata.
- **Ultimo intervento (29/08/2026, follow-up 3).** **App web: metrica di
  stazionarieta' corretta per Newton-CG-$L_1$: norma del subgradiente di
  $F=J+\nu\|w\|_1$.** (1) Nuova **metric-card "‖∂F‖ (subgrad. L1)"** nella
  scheda info, visibile solo quando il metodo è $L_1$: mostra
  $\|\partial F(w_k)\|$ (subgradiente di $F$), che è la misura di
  stazionarietà corretta per il problema regolarizzato. (2) Nella **tabella di
  convergenza** aggiunta la colonna "‖∂F‖" per $L_1$ e il verdetto finale ora
  si basa su $\|\partial F\|$ (soglia $10^{-4}$ → "Convergenza raggiunta")
  invece che su $\|w-w^*\|$, che è riferito al minimo di $J$ e per $L_1$ non
  va a 0 nemmeno a convergenza (era la causa della falsa impressione di
  "blocco" sul preset `1d_exp` $\nu=0.991$: $\|w-w^*\|=0.7487$ pur essendo
  convergente, con $\|\partial F\|\approx1.6\times10^{-10}$). (3) Nel **plot
  2D** aggiunta la stella sul minimo della superficie $F$ (dalla griglia).
  Validazione: harness deno + numpy reale — `pts_sgF` popolato solo per
  newton_l1 (per GD/Newton-CG/BB vuoto e tabella senza colonna "‖∂F‖"),
  valori coerenti ($1d\_exp$ $\nu=0.991$: $\|\partial F\|$ da $2.64$ a $0.115$
  a 30 iterazioni, → ~$10^{-10}$ a convergenza). Nessun PDF coinvolto;
  `bozza.tex` non toccata.
- **Ultimo intervento (29/08/2026, follow-up 4).** **App web: preset 1D resi
  stocastici (target per campione), così la dinamica del batch (CCV) è visibile
  anche in 1D.** Prima i preset 1D (`1d_quad`, `1d_quartic`, `1d_sin`,
  `1d_exp`) avevano `loss_i(w,i) = J(w)` (identica per ogni campione):
  varianza zero → la CCV non scattava mai e `n_k` restava fisso a `batch₀`
  (corretto ma non rappresentativo degli algoritmi stocastici della tesi). Ora
  ogni preset genera un dataset con **target $a_i$ per campione** (`raw_a = 1 +
  0.2*randn(N)`, centrati così `mean(a_i)=1`), `loss_i`/`grad_i`/`hess_i`
  dipendono da `i` e la **$J$ mostrata è la media full-batch**
  (`mean_i loss_i(w,i)`); **`W_STAR` è ricalcolato a runtime** (Newton su $J$
  full-batch con differenze finite) così è coerente col minimo empirico
  (verificato: $1.0000$, $1.0055$, $0.9999$, $1.0004$ per i 4 preset).
  Effetto: varianza del gradiente $>0$ anche in 1D → **la CCV fa crescere
  `n_k`** (es. `1d_quad`/`1d_exp`: batch $5 \to 200$ in 30 iterazioni), come
  sui preset 2D. Bonus: la modalità **"Formule chiuse" (`diffMode=preset`) ora
  funziona anche sui preset 1D** (prima falliva con `NameError: N`, perché i
  preset non definivano `N`; ora lo definiscono). Validazione: harness deno +
  numpy reale — 4 preset 1D in modalità preset e autodiff (batch crescente,
  $W\_STAR$ coerente, vista $F$ e metrica $\|\partial F\|$ invariate), 2D e
  GD/Newton-CG/BB non toccati (regressione 13/13, 11/11, 10/10). Nessun PDF
  coinvolto; `bozza.tex` non toccata.
- **Ultimo intervento (28/08/2026).** **App web: nuovo strumento "Test batch"
  per esperimenti e tabelle in stile Sez. 6.7.** In `visualizzazione.html`
  (file solo nella repo, nessun PDF coinvolto) aggiunto in fondo alla sidebar,
  sotto i risultati dell'Analisi, il pulsante **"🧪 Test batch"** che apre una
  modale per eseguire il prodotto cartesiano di più dimensioni (loss,
  algoritmo, strategia di riuso, seed e tutti gli iperparametri dell'app:
  max_iter, alpha, theta, batch0, N, w0, line search, batch dinamico,
  sottocampionamento Hessiana, Hessian-free L1, R, maxcg, nu, sigma, eta) e
  generare la tabella e30 come nella Sez. 6.7. Per ogni dimensione si
  inseriscono i valori separati da virgola; spuntando **"media"** la dimensione
  viene marginalizzata (media aritmetica di e30 e ricampionamenti sui valori
  inseriti, es. più seed per la robustezza). Colonne = strategia di riuso
  (`base`, `M=inf`/`M=k`, `H ind M_H=inf`/`k`, `val P=..;p=..;tau=..;f=..;strat=..;minabs=..`,
  `desc P=..;tau=..;f=..;minabs=..`, parametri separati da `;`), righe =
  combinazioni delle dimensioni non-media (se `strategia` è su media le colonne
  sono l'ultima dimensione non-media). Ogni run esegue ESATTAMENTE il codice
  Python generato dall'app (`generateAlgoCode`) in un namespace pulito via
  Pyodide (helper `_batch_run`), con barra di avanzamento e tolleranza di
  errori per-run. Tabella con marcatori ▲/▼/≡ relativi a `base`, sfondo su
  scala logaritmica, e pulsanti **"Copia LaTeX"** (ambiente
  `table`/`tabular` come `gen_tabelle_riuso*.py --tex`), **"Scarica JSON"** e
  **storico salvato nel browser** (localStorage, ultimi 8 esperimenti, pulsanti
  👁/⬇/🗑). Validazione: sintassi JS (deno lint) e smoke test top-level con DOM
  fittizio; 11 configurazioni (GD/BB/Newton-CG/Newton-L1 × base/M=2/M=∞/val/desc/
  H ind) generate ed eseguite con Python reale: e30 identici alle tabelle della
  tesi (es. base GD quad_well 1.19e-1, GD M=2 1.18e-1, L1 M=∞ 8.17e-2,
  L1 M=5 1.00e-1), conteggio ricampionamenti coerente (base = numero di
  iterazioni, val/desc da `m_actual`, riuso manuale = `resample_pts` +
  `resize_points`). `bozza.tex` non toccata.
- **Ultimo intervento (28/08/2026, follow-up).** **Test batch: nuovo formato
  di tabella «Robustezza (sintesi)» in stile Tabella 6.27/6.24/6.21.** In
  `visualizzazione.html` aggiunto nella modale il selettore **Formato**
  (Matrice / Robustezza). In formato Robustezza (da usare con `seed` su
  `media` e una colonna `base`) la tabella ha una riga per configurazione e
  colonne `ē₃₀` (media di e30 su tutte le caselle), `vitt.` (numero di caselle
  su 16 in cui la media è minore di quella di `base`) e le medie per problema
  (prima attributo delle righe, tipicamente la loss), sia in HTML sia in
  LaTeX (`\begin{table}...\end{table}` con `\midrule` dopo la riga base).
  Validazione end-to-end su Python reale dei 400 codici della 6.27
  (4 problemi × 4 algoritmi × 5 strategie × 5 seed): `M=∞` esatto
  (3.640e-1), configurazioni del criterio di discesa identiche a 3 cifre
  significative (2.03e-1, vitt. 10/16), riga `base` 2.17e-1 vs 2.12e-1 della
  tesi (la tesi calcola la base come codice del criterio di discesa con
  ricampionamento forzato `if True:`, il tool come codice standard non-riuso;
  ~2%, percorso RNG diverso). Con line search per-algoritmo (BB=armijo,
  altri=wolfe, come nella tesi) le colonne del criterio sono allineate. Tutti
  i test JS (deno lint, smoke test DOM fittizio, funzionali, render, LaTeX)
  passano. `bozza.tex` non toccata.
- **Ultimo intervento (28/08/2026, follow-up 2).** **Test batch: line search
  per-algoritmo nel campo "Line search".** Il campo accetta ora anche la
  sintassi `gd=wolfe;bb=armijo;newton_cg=wolfe` (coppie `algoritmo=wolfe|armijo`
  separate da `;`; valore singolo senza `=` = per tutti, default `wolfe`),
  necessaria per riprodurre fedelmente le tabelle della tesi dove BB-CCV usa
  Armijo e GD/Newton-CG Wolfe. Implementata `batchResolveLineSearch` e usata
  in `batchBuildFullCode`. Validazione: sweep reale dei 400 codici della 6.27
  con la stringa composita: M=∞ esatto (3.640e-1), colonne `desc` identiche a
  3 cifre significative (2.03e-1, vitt. 10/16), `base` 2.17e-1 (nota ~2% di
  cui sopra); verificato che i codici generati contengano davvero Armijo per
  BB e Wolfe per GD. Tutti i test JS passano. README aggiornato. Nessun PDF
  coinvolto.
- **Ultimo intervento (28/08/2026, follow-up 3).** **Test batch: fix del campo
  `w0 (x,y)`.** Il valore `2.0,-3.0` è un VETTORE: la virgola interna non è un
  separatore di lista, ma veniva interpretata come tale e raddoppiava le run
  (es. la configurazione 6.27 passava da 400 a 800 run e da 16 a 32 caselle).
  Ora la dimensione `w0` usa `;` come separatore tra più vettori
  (es. `2.0,-3.0;1.0,1.0`), mentre `,` separa x e y dentro il vettore
  (`batchParseList` accetta un separatore per-dimensione; `w0` ha `sep:';'`).
  Hint della dimensione aggiornato. Verificato: `w0=2.0,-3.0` → 1 valore,
  conteggio run della 6.27 = `4 × 4 × 5 × 5` = 400, grid reale di 400 config.
  Tutti i test JS passano. README aggiornato. Nessun PDF coinvolto.
- **Ultimo intervento (28/08/2026, follow-up 4).** **Test batch: strategie
  `desc base` e `val base` per riprodurre la colonna `base` della tesi.**
  La tesi calcola la colonna `base` delle tabelle del riuso per discesa (e
  validation) come *codice del criterio con ricampionamento forzato a ogni
  iterazione* (guard `if True:`), NON come codice standard non-riuso: da qui la
  differenza `2.17e-1` vs `2.12e-1` della 6.27. Ora le strategie `desc base` /
  `val base` generano la variante del criterio e sostituiscono il guard di
  campionamento con `if True:` (come `gen_tabelle_riuso_descesa.py` /
  `gen_tabelle_riuso_validation.py`), etichettate `base (discesa)` /
  `base (validazione)`; il riconoscimento della colonna base per i marcatori
  accetta anche `desc base`/`val base`. Validazione end-to-end (sweep reale dei
  400 codici con `desc base` al posto di `base`): base ē₃₀ = **2.121e-1** con
  medie per problema identiche alla 6.27 (κ≈1.1 1.793e-1, κ≈20 1.759e-1,
  κ≈100 4.116e-1, incr. 8.142e-2), M=∞ 3.640e-1 (8/16), def. 2.034e-1 (10/16),
  τ=1e-3 2.034e-1 (10/16 vs 11/16, cella borderline), τ=1e-5 2.034e-1 (10/16).
  Tutti i test JS passano. README aggiornato. Nessun PDF coinvolto.
- **Ultimo intervento (28/08/2026, follow-up 5).** **Test batch: pulsante
  "🔍 Codice 1ª run" per il debug.** In formato Matrice e Robustezza, accanto
  a Copia LaTeX / Scarica JSON, il pulsante mostra il codice Python generato
  per la prima combinazione della griglia (guard di ricampionamento e line
  search per-algoritmo), utile per verificare che `desc base`/`val base`
  abbiano davvero il guard `if True:` e che la line search sia quella attesa.
  Diagnosi del caso 6.27 (base 2.16e-1 invece di 2.12e-1): verificato che il
  file genera il codice corretto (cella per cella, 4 algoritmi, e30 identici
  alla validazione), che il round-trip JSON `_batch_run(JSON.stringify(code))`
  dell'app è lossless, che numpy 1.26.4 (Pyodide) e 2.4.6 danno sequenze RNG
  identiche, e che il run precedente con `base` standard coincideva esattamente
  con la validazione (κ≈20 1.59e-1, κ≈100 4.33e-1): la discrepanza residua su
  κ≈20/κ≈100 con `desc base` è quindi quasi certamente una cache del browser
  o uno storico (localStorage) ricalcolato con una versione precedente del
  file. README aggiornato. Nessun PDF coinvolto.
- **Ultimo intervento (28/08/2026, follow-up 6).** **Tesi Sez. 6.2.1 (Setup
  Sperimentale): nota su riproducibilità con il Test batch e differenze WASM di
  Pyodide.** Dopo il paragrafo sulla riproducibilità del pannello "Analisi"
  aggiunto un paragrafo che spiega che gli esperimenti della Sez. 6.7 possono
  essere riprodotti anche con lo strumento *Test batch* dell'app (che genera
  le tabelle e30 in LaTeX nel browser), con la precisazione che i valori
  possono differire di qualche punto percentuale sui problemi mal condizionati
  (κ≈20 e κ≈100) per le colonne che ricampionano (`base`, `desc`), mentre le
  celle ben condizionate e `M=∞` coincidono: la causa è il build WASM di
  Pyodide (arrotondamenti in virgola mobile diversi dal NumPy di sistema), che
  si amplificano sui problemi mal condizionati; gli script di generazione del
  repository (NumPy di sistema) restano il riferimento per la riproduzione
  esatta. Ricompilati tesi.pdf e tesi_finale.pdf (131 pp, 0 errori, 0
  undefined, overfull invariati). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (28/08/2026, follow-up 7).** **Sezioni 6.5/6.6
  (estratto `tesi/sezioni_65_66.tex`): riproducibilità verificata e completata
  per la Sez. 6.6.** Verificata la riproducibilità dichiarata delle Sez. 6.5
  (nota) e 6.6 (rete): link GitHub/Colab corrispondenti ai notebook presenti in
  `tesi/nsynth/` (repo `Alessandro1040/selezione-dinamica-batch`, branch
  `main`), valori di `tab:nsynth_nota` e `tab:nsynth_net` coincidenti 1:1 con
  `colab_risorse/figure/nota/results.json` e
  `colab_risorse/figure/net/results_net.json`, parametri dichiarati nel testo
  uguali a quelli dei notebook, documento compilato senza errori né riferimenti
  indefiniti. Correzioni in `sezioni_65_66.tex`: (1) la soglia della
  validazione interna del notebook rete ora riporta i valori esatti (errori <
  1e-8 su gradiente/loss/varianza CCV e < 1e-6 sul prodotto Hessiano--vettore,
  non "errori < 10^{-8}"); (2) aggiunti in coda alla Sez. 6.6 i paragrafi
  "Riproducibilità" (eseguendo dall'inizio `nsynth_net_riproduzione` si
  riproducono estrazione features, addestramento, figure e tabella) e
  "Precisione sui valori numerici" (valori/tempi della run dell'autore, non
  Colab; riproducibilità non esatta per stocasticità e arrotondamenti; in Colab
  tempi di decine di minuti), simmetrici a quelli della Sez. 6.5. Ricompilato
  `sezioni_65_66.pdf` (15 pp, 0 errori, 0 undefined, 0 overfull nuovi).
  Sincronizzati nella repo tesi/sezioni_65_66.tex e tesi/sezioni_65_66.pdf
  (md5 verificati). `tesi.tex` non toccata (le Sez. 6.5/6.6 in tesi.tex restano
  placeholder in attesa del merge); `bozza.tex` non toccata.

- **Ultimo intervento (28/08/2026, follow-up 8).** **Tesi: 4 correzioni.** (1)
  Sez. 5.1: corretto il posizionamento delle tre figure della sezione
  (Fig. 5.1 schema a blocchi del Dynamic GD, Fig. 5.2 cono di discesa, Fig. 5.3
  andamento del batch) cambiando la spec del float da `[h]` a `[H]`: con `[h]`
  le figure non trovavano posto nella sezione e finivano in coda al documento
  (pagg. 130-131); ora compaiono nella Sez. 5.1 (pagg. 15, 17, 20), mentre gli
  schemi di Newton-CG/L1/BB-CCV erano già a posto. (2) Sez. 4.4: rimosso il
  paragrafo sul *soft thresholding*; resta solo la strategia dell'active set con
  la nuova spiegazione di come si identificano le variabili nulle all'ottimo
  (confronto di $\partial J/\partial w_i$ con $\pm\nu$ tramite il gradiente
  generalizzato, in linea con l'Eq. 5.44 della Sez. 5.3). (3) Ringraziamenti:
  inserito il segnaposto `(placeholder)` in corsivo (font `\emph`, poi
  semplificato nel follow-up 9). (4) Sez.
  7.1 (Limiti dello studio): rimossa la parentesi "(ad esempio la perdita
  logistica non lo è)". Ricompilati `tesi.pdf` e `tesi_finale.pdf` (131 pp, 0
  errori, 0 undefined). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.

- **Ultimo intervento (28/08/2026, follow-up 9).** **Ringraziamenti:
  placeholder semplificato.** Il segnaposto dei ringraziamenti è ora solo
  `\emph{(placeholder)}` (testo `(placeholder)` con il font originale in
  corsivo, come la frase che c'era prima); rimosso `[PLACEHOLDER] Inserire qui
  i ringraziamenti.` Ricompilati `tesi.pdf` e `tesi_finale.pdf` (131 pp, 0
  errori, 0 undefined). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.

- **Ultimo intervento (28/08/2026).** **Tesi Sez. 4.4: riscritto il passo
  sull'identificazione delle variabili da tenere a zero nel metodo
  Newton-CG-$L_1$.** Prima: confronto di $\partial J(w)/\partial w_i$ con
  $\pm\nu$ (se $w_i=0$ e $-\nu \le \partial J(w)/\partial w_i \le +\nu$ la
  variabile resta ancorata a zero). Ora: quando $w_i=0$, tra tutti i
  subgradienti possibili in $[-\nu,+\nu]$ viene selezionato quello che rende
  minima la componente del gradiente generalizzato (zero se possibile), cosi'
  $w_i=0$ viene riconosciuto come un vero minimo del problema ogni volta che
  lo e', e l'algoritmo non rischia di allontanarsene. Allineata alla
  spiegazione in corsivo della Sez. 5.3.2 (il gradiente generalizzato).
  Ricompilati `tesi.pdf` e `tesi_finale.pdf`; sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.

- **Ultimo intervento (27/08/2026, follow-up).** **Verifica di coerenza della
  Sez. 6.7 e dell'app dopo il fix BB-CCV.** Riesaminati `visualizzazione.html`
  (tutti i generatori: BB/validation/descesa usano il passo BB; GD, Newton-CG
  e Newton-L1 non hanno lo stesso problema, il `step = alpha` nelle Newton-L1
  è il passo iniziale legittimo della line search proiettata, presente anche
  nella versione normale) e la tesi (tabelle e prosa). Corretta la nota
  *Lettura.* sotto `tab:riuso_robustezza` (6.21), che diceva "quasi tutti i
  metodi migliorano sul termine incrociato" e "sui mal condizionati il riuso
  peggiora": con i nuovi dati BB il segno dell'effetto dipende dal metodo
  (BB-CCV migliora su κ≈20 ma peggiora su κ≈100, già alla precisione
  macchina). Verificato che `tab:riuso_cons_sintesi` (6.29, generata con il
  BB corretto) è coerente con i nuovi dati: mediana BB su κ≈20 = 4.7042e-6
  (riprodotta esattamente), su κ≈100 coerente con l'early stop. I valori
  uguali della colonna base BB su κ≈20 e κ≈100 nella tabella validation sono
  corretti (bias del training set, indipendente da κ). Ricompilati
  `tesi.pdf`/`tesi_finale.pdf` (**130 pp**, 0 errori, 0 undefined).
  Sincronizzati in repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
- **Ultimo intervento (27/08/2026).** **BB-CCV: corretto un bug che scartava
  il passo di Barzilai–Borwein negli esperimenti del riuso (Sez. 6.7), nella
  app (varianti validation/descesa) e negli harness; rigenerati tutti i dati
  BB-CCV della Sez. 6.7; corretti anche due "1.4"→"1.28" e rimossa ogni
  occorrenza di "Rayleigh".** (1) **Causa radice**: le varianti BB-CCV con
  validation/descesa dell'app (`generateBBValidation`/`generateBBDescent`
  usavano lo snippet `lsArmijoGD`/`lsWolfeGD` che contiene `step = alpha`) e
  gli harness degli esperimenti (`validation_codice_generato/bb_armijo.py`,
  `descent_codice_generato/bb_armijo.py`, e il generatore storico delle
  tabelle E) **sovrascrivevano il passo BB con il default fisso α** prima
  della line search di Armijo: BB-CCV degenerava in GD con Armijo. Corretto
  aggiungendo `lsArmijoBB`/`lsWolfeBB` (senza il reset) in
  `visualizzazione.html` e rimuovendo la riga `step = alpha` dai due file
  estratti. Il BB-CCV normale (app, tabelle test Sez. 6.5, tabelle consigliati
  6.29) era già corretto: per questo c'era la contraddizione interna (Sez.
  6.5/6.29: BB alla precisione macchina su κ≈100; Sez. 6.7: BB bloccato a
  4.67e-1). (2) **Rigenerati** con l'implementazione corretta: tabelle E-BB
  (6.7/6.11/6.15/6.19), righe BB della sintesi 6.20 e della robustezza 6.21
  (nuovi generatori in `/tmp`, non in repo), tabelle validation 6.22–6.24 e
  descesa 6.25–6.27 (rieseguiti `gen_tabelle_riuso_validation.py` e
  `gen_tabelle_riuso_descesa.py` con il BB corretto; GD/NCG/L1 invariati) e
  confronto finale 6.28. **Nuovi risultati BB-CCV**: su κ≈20 la base converge
  a 9.72e-5 e il riuso la migliora (M=∞: 9.54e-10); su κ≈100 la base è già
  alla precisione macchina (1.10e-14 a k=11) e il riuso la peggiora (7.45e-3);
  su κ≈1.1 e incrociato il riuso è neutro. *"Dynamic GD e BB-CCV si
  comportano in modo quasi identico"* rimosso (era un artefatto del bug):
  la prosa della Sez. 6.7.4/6.7.5/6.7.6/6.7.7 è stata riscritta per separare i
  due metodi. (3) **Conseguenze a valle**: nello stop adattivo con validation
  i default ora risultano peggiori della base (2.64e-1 vs 2.17e-1 su 5 seed;
  la calibrazione P=1,p=10%,dyn li porta a 2.11e-1, 10/16); nella discesa τ ha
  un effetto limitato e la config. consigliata (τ=10⁻³,P=1,f=1) dà 2.02e-1
  (11/16), ora **leggermente meglio** della validation (2.11e-1, 10/16) —
  invertita la frase "la validation è leggermente migliore". (4) Corretti
  `e_{30}≈1.4` → `1.28` (righe 3240 e 4361) e rimossi "quoziente di
  Rayleigh"/"media di Rayleigh" (righe 2304 e 6617, sostituiti con la
  definizione di curvatura media pesata). Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**130 pp**, 0 errori, 0 undefined; i 13 overfull
  preesistenti, nessun nuovo). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf + visualizzazione.html + i due bb_armijo.py
  (md5 verificati). `bozza.tex` non toccata. Nota: i nuovi dati BB delle
  tabelle E/sintesi/robustezza vengono da un generatore in `/tmp`
  (non in repo); le tabelle validation/descesa sono rigenerabili con gli
  script esistenti.
- **Ultimo intervento (27/08/2026).** **Tesi Sez. 6.7.2 (Meccanismo del
  riuso): corretta la varianza del rumore per-esempio nel modello teorico.**
  Il modello illustrativo dichiarava $\varepsilon^{(i)} \sim
  \mathcal{N}(0, 0.36\,I)$ (σ=0.6), ma i codici (`visualizzazione.html`,
  `altro/script/gen_tabelle_riuso.py`, `simulazione_batch.py`) generano i dati
  con `0.2 * np.random.randn(...)` per $a_i$ e $b_i$: in coordinate spostate
  ($\varepsilon_1 = a_i{-}1$, $\varepsilon_2 = b_i{+}2$) si ha quindi σ=0.2 e
  **Var(ε)=0.04 per componente** (misurata sul dataset seed 42:
  ≈0.035–0.039). Corretto `0.36\,I` → `0.04\,I`. Corretto anche lo
  spostamento medio del minimo campionato: `\bar\varepsilon_2 ≈ 0.04` → `≈
  0.09` per $n_k=5$ (0.2/√5 ≈ 0.089, senza correzione per popolazione
  finita; il vecchio 0.04 era σ/n, scala sbagliata per una media).
  L'archivio storico `altro/appendice_riuso.tex` (non ricompilato) conserva
  i vecchi valori. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**129 pp**,
  0 errori, 0 undefined). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (27/08/2026).** **Tesi Sez. 6.7 (Tabelle 6.23 e 6.26):
  corrette le tabelle "iperparametri" che non riportavano il nome
  dell'iperparametro nella prima colonna.** In `tab:riuso_valid_iper`
  (Tabella 6.23) e `tab:riuso_desc_iper` (Tabella 6.26) la colonna
  *Iperparametro* mostrava il nome solo sulla prima riga ("Pazienza $P$") e
  lasciava vuote le righe di $\tau$, $p$, $f$ e strategia di split. Ora il
  nome compare in **ogni riga** di ogni gruppo (Pazienza $P$, Tolleranza
  $\tau$, Percentuale $p$, Frequenza $f$, Strategia di split per la 6.23;
  Pazienza $P$, Tolleranza $\tau$, Frequenza $f$ per la 6.26). Valori
  numerici invariati. Allineati anche i generatori:
  `altro/script/gen_tabelle_riuso_validation.py` e
  `altro/script/gen_tabelle_riuso_descesa.py` (rimosso il flag `first` che
  emetteva il label solo nella prima riga del primo gruppo; etichetta
  "Strategia" resa "Strategia di split" nel generatore validation). Le righe
  di tabella generate ora iniziano con `{label} & {vs} & ...` per ogni riga.
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**129 pp**, 0 errori, 0
  undefined; i 13 overfull identici a quelli preesistenti, nessuno dentro le
  due tabelle corrette). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf e i due generatori (md5 verificati). `bozza.tex` non
  toccata.
- **Ultimo intervento (27/08/2026).** **Tesi Sez. 6.7 (Tabelle 6.5 e 6.6):
  spiegato il profilo a "scalini" e corretta la nota *Lettura.* di Newton-CG
  $L_1$ sul ben condizionato.** (1) Dopo la nota *Lettura.* della Tabella 6.5
  (`tab:riuso_bencond_ncg`, Newton-CG su $\kappa\approx1.1$) aggiunto un
  paragrafo che spiega perché l'errore procede a "scalini": la direzione di
  Newton usa la Hessiana stimata su $n_h=\min(\max(1,\lceil Rn_k\rceil),N)$
  campioni; finché $n_k=5$ si ha $n_h=1$ (curvatura di un solo campione) e la
  direzione del CG troncato spesso non soddisfa le condizioni di Wolfe sulla
  loss del batch per nessun passo nel range testato, quindi la line search
  restituisce un passo nullo e l'iterato resta fermo (tratti piatti); la CCV
  non aumenta il batch perché misura la varianza del gradiente, non la
  qualità della direzione; il calo arriva quando un nuovo campione (o un
  batch ingrandito dopo violazione della CCV) produce una direzione
  accettabile. Il fenomeno non dipende dal riuso (compare anche nella colonna
  *base*, che ha la stessa $n_k$ e la stessa $n_h$). Spiegazione verificata
  eseguendo il codice esatto di `altro/script/gen_tabelle_riuso.py`
  (diagnostica: nei tratti piatti la line search di Wolfe restituisce
  esattamente step=0.0). (2) Corretta la nota *Lettura.* della Tabella 6.6
  (`tab:riuso_bencond_l1`, Newton-CG $L_1$ su $\kappa\approx1.1$): diceva
  "comportamento intermedio, senza una tendenza netta tra le politiche di
  riuso" ma in questa tabella il riuso migliora l'errore finale con TUTTE le
  politiche (base $1.02\times10^{-1}$; $M{=}\infty$ $8.17\times10^{-2}$;
  $M{=}10$ $9.24\times10^{-2}$; $M{=}5$ $1.00\times10^{-1}$; $M{=}2$
  $9.41\times10^{-2}$; H ind. $9.78\times10^{-2}$); la nota ora lo specifica
  e precisa che il comportamento "intermedio, senza tendenza netta" vale per
  i problemi mal condizionati. Le note di `tab:riuso_malcond_l1` e
  `tab:riuso_veryill_l1` restano invariate (lì il segno dell'effetto cambia
  con la politica di riuso). Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**129 pp**, 0 errori, 0 undefined; overfull identici alle build
  precedenti). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf
  (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (27/08/2026).** **Tesi: corrette 6 note \emph{Lettura.}
  sotto le tabelle di errore del riuso (E.1--E.16) con affermazioni errate o
  non riferite al seed della tabella.** (a) Le note sotto
  `tab:riuso_bencond_bb`, `tab:riuso_malcond_gd` e `tab:riuso_veryill_gd`
  attribuivano alla tabella (che è al solo seed 42) risultati su 5 seed: le
  affermazioni su 5 seed sono ora qualificate con rimando alla Tabella
  `tab:riuso_robustezza` (per Dynamic GD su $\kappa\approx20$ il quadro su 5
  seed è misto: 2 migliora/3 peggiora, non "su tutti e 5"). (b) Le note sotto
  `tab:riuso_veryill_gd` e `tab:riuso_veryill_bb` dicevano che $M=\infty$
  peggiora su $\kappa\approx100$: le tabelle mostrano invece che migliora
  ($3.71\times10^{-1}$ contro $4.67\times10^{-1}$ della base, guadagno di
  coerenza su una base già molto lenta): corrette. (c) La nota sotto
  `tab:riuso_offdiag_ncg` diceva che $M=\infty$ fa collassare la convergenza
  sul termine incrociato: la tabella mostra che $M=\infty$ è la colonna
  migliore ($4.10\times10^{-2}$ contro $3.52\times10^{-1}$): corretta. (d)
  Corretta l'entità del collasso di Newton-CG su $\kappa\approx100$
  ($e_{30}\approx1.28$, non 1.4) nella nota di `tab:riuso_veryill_ncg`.
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**129 pp**, 0 errori, 0
  undefined; overfull identici alle build precedenti). Sincronizzati nella
  repo (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (27/08/2026).** **Tesi Sez. 6.7: aggiunte due
  spiegazioni concettuali sintetiche (richieste).** (1) **Perché il riuso
  peggiora sui problemi mal condizionati e, in particolare, per i metodi di
  Newton**: nuovo paragrafo nella sottosezione "Meccanismo: vantaggi e limiti
  del riuso" (`app:riuso-meccanismo`) che spiega la differenza geometrica
  primo vs secondo ordine ($d_k=-g_k$ è sempre allineato alla pendenza nel
  punto corrente, la discesa sul batch è garantita e il riuso provoca solo la
  deriva; in Newton $d_k=-H_k^{-1}g_k$ e un'Hessiana "vecchia" ruota/riscala
  il gradiente fresco applicando una geometria obsoleta: più $\kappa$ è
  grande più il disallineamento è dannoso, l'angolo con la vera discesa si
  avvicina a $90^\circ$ e la line search blocca il metodo). (2) **Perché
  mediana e media geometrica** nel criterio dei consigliati (Sez. 6.7.8,
  subito dopo la frase del criterio): la mediana è robusta ai seed divergenti
  (filtro di ammissibilità; la conta delle vittorie ignorerebbe l'entità del
  miglioramento e si può vincere 1/5 pur con mediana migliore) e la media
  geometrica dei rapporti è invariante alla scala assoluta degli errori
  (equivale alla media aritmetica dei logaritmi dei rapporti; dimezzare
  l'errore su un problema conta quanto raddoppiarlo su un altro). Ricompilati
  `tesi.pdf` e `tesi_finale.pdf` (**129 pp**, 0 errori, 0 undefined; i 12
  overfull identici a quelli delle build precedenti). Sincronizzati nella
  repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.
- **Ultimo intervento (27/08/2026).** **Tesi: aggiunte sotto ognuna delle 44
  tabelle una breve nota introduttiva `\noindent\emph{Lettura.}` che spiega a
  cosa serve quella tabella e sintetizza la prosa che la segue (richiesto:
  non prolissa come le spiegazioni dell'LLM).** Note aggiunte a tutte le 44
  tabelle: ``Bound di complessità'', Tab. 5.1 (confronto dei tre metodi),
  Tab. BB-CCV, Tabelle 6.1--6.3 (errore per iterazione, Sez. 5), Tabelle
  E.1--E.16 (riuso del mini-batch, errore per iterazione), E.17 (sintesi
  $e_{30}$), E.18 (robustezza su 5 seed), E.19--E.21 (stop adattivo con
  validation set), tabelle del riuso per discesa, Tabella confronto finale e
  le 13 tabelle ``consigliati'' (sintesi + confronto iterazione per
  iterazione). Ogni nota riporta la lettura chiave della tabella (es. per i
  consigliati: riuso $M=5$ per Dynamic GD, base per BB-CCV, stop adattivo
  ($P{=}1$, $f{=}1$, $p{=}10\%$, split fissa) per Newton-CG e riuso $M=3$ per
  Newton-CG $L_1$). Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**127 → 129 pp**, 0 errori, 0 undefined; i 12 overfull risultano identici
  a quelli della build precedente, verificato ricompilando la versione pre-edit
  in `/tmp`). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf
  (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 6.7: nuovo esperimento
  ``Riuso per discesa della loss sul batch'' (criterio alternativo allo stop
  adattivo con validation set), replicando esattamente la procedura della
  Sez. 6.7.6.** (1) **Fix bug in `visualizzazione.html`**: nel
  `preLoopNewton` di `descentSnippets` mancavano `patience_S = 0` e
  `patience_H = 0` (la variante validation li inizializza); in Python il
  codice generato dei metodi di Newton crashava (`UnboundLocalError`) al
  primo mancato miglioramento, in JS `patience_S += 1` su `undefined`
  produceva `NaN` e il criterio non scattava mai (varianti Newton del riuso
  per discesa rotte nell'app). Aggiunte le due righe, allineate a
  `validationSnippets`. (2) **Harness di estrazione**: nuovo
  `altro/script/estrai_codice_descesa.mjs` (Deno + DOM fittizio, come per le
  varianti *Validation) che estrae da `visualizzazione.html` i 4 codici
  Python esatti col criterio di discesa in
  `altro/script/descent_codice_generato/` (gd_wolfe, bb_armijo,
  newton_cg_tied, newton_l1_tied; line search come Sez. 6/Appendice E,
  Hessiana legata a $S_k$). (3) **Sweep**: nuovo
  `altro/script/gen_tabelle_riuso_descesa.py` (analogo a
  `gen_tabelle_riuso_validation.py`): 18 configurazioni
  ($\\tau\\in\\{10^{-5},10^{-4},10^{-3}\\}$, $P\\in\\{1,3,8\\}$,
  $f\\in\\{1,3\\}$, $\\epsilon_{\\mathrm{abs}}=0$) × 4 problemi × 4
  algoritmi, seed 42; riferimenti *base* e $M=\\infty$ sull'intero dataset
  (nessun validation set, tetto CCV $N$). Dati in `descent_sweep.json` e
  robustezza 5 seed in `descent_robustezza.json`. (4) **Tesi Sez. 6.7.7**:
  nuova sottosezione ``Riuso per discesa della loss sul batch''
  (`app:riuso-descesa`) tra la validation (6.7.6) e i consigliati (ora
  6.7.8), con meccanismo, setup, iperparametri testati, 3 tabelle
  (`tab:riuso_desc_confronto`, `tab:riuso_desc_iper`,
  `tab:riuso_desc_robustezza`) e confronto finale base vs validation vs
  discesa (`tab:riuso_confronto_finale`). Risultati: pazienza $P{=}1$
  migliore ($2.61\\times10^{-1}$ vs $3.30\\times10^{-1}$ di $P{=}8$);
  tolleranza quasi ininfluente; $f{=}1$ meglio di $f{=}3$. La monotonia
  delle ricerche lineari rende il criterio poco sensibile (per i metodi del
  primo ordine il default ≈ $M{=}\\infty$), ma per i metodi di Newton evita
  il collasso di $M{=}\\infty$. Migliore configurazione
  $\\tau{=}10^{-3},P{=}1,f{=}1$: media su 5 seed $2.31\\times10^{-1}$
  (12/16 vittorie vs *base* $2.43\\times10^{-1}$; $M{=}\\infty$
  $3.95\\times10^{-1}$). Confronto finale: validation calibrata
  $2.24\\times10^{-1}$ (11/16 vs base sul training set), discesa calibrata
  $2.31\\times10^{-1}$ (12/16 vs base sull'intero dataset): la validation è
  leggermente migliore in media ma tiene fuori il 10\\% dei dati; la discesa
  usa tutto il dataset senza split. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**127 pp**, 0 errori, 0 undefined, overfull solo
  preesistenti). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf/
  visualizzazione.html + nuovo materiale in `altro/script/` (md5 verificati).
  `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Nuovo script di riproduzione della
  Tabella 6.22 (`tab:riuso_valid_confronto`): `altro/script/
  riproduci_tabella_622.py`.** Codice pronto per Colab che riproduce
  esattamente i valori della Tabella 6.22 (4 problemi × 4 algoritmi = 16
  combinazioni; colonne base, $M=\infty$, def., $P{=}1,p{=}0.1$,dyn e
  $P{=}3,p{=}0.1$,dyn). Usa il codice ESATTO dell'app: preset =
  `LOSS_PRESETS` di `visualizzazione.html` (costruzione dataset) e algoritmi
  = codici generati dall'app (varianti *Validation) in
  `validation_codice_generato/` (Dynamic GD Wolfe, BB-CCV Armijo,
  Newton-CG e Newton-CG~$L_1$ con $H_k$ legata a $S_k$); la colonna base
  forza il ricampionamento a ogni iterazione (guard → `if True:`) come in
  `gen_tabelle_riuso_validation.py`. Validato: l'output coincide
  riga per riga con la Tabella 6.22 di `tesi.tex` (valori e ricampionamenti).
  Nessun PDF coinvolto (file solo in repo). README aggiornato.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 6.7: aggiunto
  $\kappa\approx1.67$ alla etichetta "incrociato"/"incr" in tutte le tabelle
  che elencano le quattro loss con il loro numero di condizionamento.**
  Modificate: Tab. 6.21 (già fatta), Tab. E.17 sintesi (`tab:riuso_sintesi`,
  4 righe), Tab. E.19 confronto (`tab:riuso_valid_confronto`, 4 righe), Tab.
  E.21 robustezza (`tab:riuso_valid_robustezza`, header colonna "incr." →
  "incr. ($\kappa{\approx}1.67$)") e Tab. sintesi consigliati
  (`tab:riuso_cons_sintesi`, 4 righe). Etichette: "incrociato
  ($\kappa\approx1.67$)" e "incr. ($\kappa{\approx}1.67$)", coerenti con le
  altre righe/colonne ($\kappa\approx1.1$, $\kappa\approx20$,
  $\kappa\approx100$). Aggiornati anche i generatori per la riproducibilità:
  `gen_tabelle_riuso_validation.py` (PRESET_LATEX per E.19 e header E.21) e
  `gen_tabelle_riuso.py` (label E.17 nel --data); sintassi verificata con
  `py_compile`. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**123 pp**, 0
  errori, 0 undefined). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 6.7.6: corretta la notazione
  dell'intervallo di test della tolleranza $\tau$.** La scrittura
  `($10^{-5}$--$10^{-3}$)` è stata sostituita con la notazione intervallo
  corretta `$(10^{-5},\,10^{-3})$` (parentesi tonde, non trattino).
  Corretta anche la stessa occorrenza in una nota precedente del README.
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**123 pp**, 0 errori, 0
  undefined). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf
  (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 6.7 (Tabella 6.21
  `tab:riuso_robustezza`): aggiunto il numero di condizionamento alla riga
  "incrociato" (→ "incrociato ($\kappa\approx1.67$)") e inserito dopo la
  tabella un commento breve che spiega i risultati.** Il commento: conferma
  il quadro già noto (riuso aiuta su $\kappa\approx1.1$, 5/5 per GD/BB-CCV;
  peggiora su $\kappa\approx20$ per la deriva verso il minimo campionario);
  spiega la differenza tra i metodi (Newton-CG e Newton-CG~$L_1$ riusano
  l'Hessiana, che diventa obsoleta: su $\kappa\approx20$ Newton-CG 0/5 e L1
  1--2/5, mentre GD/BB-CCV riusano solo il gradiente e mantengono qualche
  vittoria); spiega perché su $\kappa\approx100$ i metodi del primo ordine
  migliorano su 3/5 seed (4/5 per L1 con $M{=}10$): con curvatura estrema
  l'errore è dominato dal rimbalzo nella direzione piatta, attenuato dal
  riuso che fissa il campione, mentre Newton-CG resta 0/5; infine
  l'incrociato ($\kappa\approx1.67$) aiuta quasi tutti (4/5 con $M{=}\infty$),
  con l'eccezione di Newton-CG~$L_1$ (2/5), meno robusto per la combinazione
  Hessiana riusata + proiezione ortante + subgradiente $L_1$. Numeri coerenti
  con le righe della tabella. Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**123 pp**, 0 errori, 0 undefined). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 6.7.6 ("Stop adattivo con
  validation set"): riscritta in modo molto più sintetico.** (1) Il paragrafo
  "Quali iperparametri vale la pena testare" è diventato "Iperparametri
  testati" e i sei iperparametri ($P$, $\tau$, $\epsilon_{\mathrm{abs}}$, $p$,
  $f$, strategia di split) sono ora elencati con `itemize`. (2) Intro e
  meccanismo condensati in due paragrafi (criterio
  $J_{\mathrm{val}} \le J_{\mathrm{val}}^{\mathrm{best}}(1-\tau)-\epsilon_{\mathrm{abs}}$,
  CCV con tetto $\lvert\mathcal{T}\rvert$, split fisso/dinamico, Hessiana in
  modalità legata). (3) Paragrafo sui risultati di sensibilità ridotto
  (pazienza più influente: $P=1$ ~24\% meglio di $P=8$; $\tau$ ininfluente;
  $p=10\%$ migliore; $f=1$ meglio di $f=3$; strategia dinamica migliore).
  (4) "Confronto con i riferimenti" condensato (default paragonabile alla
  base con meno ricampionamenti; calibrazione $P{=}1,p{=}10\%$,dyn →
  $2.24\times10^{-1}$ vs base $2.47\times10^{-1}$; sintesi finale). Tabelle
  E.19–E.21 **intatte** (auto-generate, non modificate); numeri e riferimenti
  invariati e verificati. Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**123 pp**, 0 errori, 0 undefined). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 6.3 ("Osservazioni
  comparative"): paragrafo riscritto in modo più semplice e intuitivo.** (i)
  Problema ben condizionato ($\kappa \approx 1.1$): curvatura quasi uniforme,
  un passo scalare ben calibrato basta in ogni direzione → Dynamic GD, BB-CCV
  e Newton-CG~$L_1$ i più veloci; Newton-CG il più lento perché risolve il
  sistema di Newton senza ricavarne vantaggio. (ii) Problema molto mal
  condizionato ($\kappa \approx 100$): direzioni ripide (passo piccolo) e
  piatte (passo grande) coesistono, un passo fisso non basta → Dynamic GD,
  Newton-CG~$L_1$ e Newton-CG restano sopra $10^{-1}$; BB-CCV arriva alla
  precisione macchina in 11 iterazioni perché il passo di Barzilai--Borwein
  ricava l'informazione sulla curvatura dai soli ultimi due gradienti (senza
  Hessiana): se il gradiente è cambiato molto lungo l'ultimo passo la
  funzione è ripida e il passo si accorcia, se è cambiato poco è piatta e si
  allunga. (iii) Termine incrociato: BB-CCV di nuovo il migliore
  ($\approx 6\times10^{-4}$ in 16 iterazioni). Sintesi: BB-CCV il più
  robusto, punto di equilibrio tra costo di calcolo e adattabilità;
  Dynamic GD sufficiente solo con curvatura ben bilanciata. Rimossi il
  termine "valle" e l'analogia del sensore economico (richiesto dall'utente).
  Numeri e riferimenti invariati e verificati. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**123 pp**, 0 errori, 0 undefined). Sincronizzati nella
  repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 5.3 ("Il gradiente
  generalizzato"): aggiunta una precisazione in corsivo sulla scelta del
  subgradiente tramite il problema di proiezione.** Subito dopo la frase
  "cioè si sceglie il subgradiente che rende il gradiente generalizzato
  $d_i + g_i$ il più vicino possibile a zero" — che segue il problema di
  proiezione $g_i = \arg\min_{g\in[-\nu,\nu]} |d_i + g|$ — è stata inserita
  una nota in `\textit` che motiva la regola: quando $w_i=0$, tra tutti i
  subgradienti possibili in $[-\nu,+\nu]$ vogliamo selezionare quello che
  rende minima la componente del gradiente generalizzato (zero se possibile),
  così che $w_i=0$ venga riconosciuto come un vero minimo del problema ogni
  volta che lo è e l'algoritmo non rischi di allontanarsene. La formulazione
  è stata poi riveduta in corsivo con le virgole al posto dei trattini.
  Nessun contenuto tecnico nuovo (la regola di proiezione era già esposta),
  solo la motivazione resa esplicita. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**123 pp**, 0 errori, 0 undefined, overfull solo
  preesistenti). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf
  (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Sez. 5.1.5/5.1.6: corretto
  `\eqref{eq:discesa}` che rendeva "(5.1.5)" invece di "(5.1)".** Il `\label`
  della condizione di discesa era dentro un display `\[...\]` non numerato
  (Sez. 5.1.5, Convergenza deterministica), quindi la referenza a riga 809
  catturava il numero della sottosezione e nel PDF compariva "(5.1.5)" — un
  numero di equazione inesistente. Spostato il `\label{eq:discesa}` sul display
  che porta la numerazione (5.1) della condizione di accuratezza (Sez. 5.1.6,
  Analisi stocastica), che ora è `\tag{5.1}\label{eq:discesa}`. Verificato nel
  PDF: "(5.1.5)" sparisce come riferimento (restano solo i numeri di
  sottosezione 5.1.5 nel TOC/headings), "Dalla condizione di discesa (5.1)"
  corretto. I 42 warning pdfTeX "duplicate ignored (equation.5.1)" nel log
  sono preesistenti (mix di display `\[ \]` con `\tag` e ambienti
  `equation` nella Sez. 5) e non cambiano: cosmetici, da ripulire
  eventualmente in un refactor dedicato. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**123 pp**, 0 errori, 0 undefined, overfull solo
  preesistenti). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf
  (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (26/08/2026).** **Tesi Appendice B (spiegazione del codice
  Newton-L1): corretti 7 riferimenti a equazioni con la vecchia numerazione
  "6.x" → "5.x".** La prosa che spiega il codice $L_1$ citava equazioni del
  capitolo 6 — "(6.2)", "(6.3)", "(6.4)", "(6.5)", "(6.6)", "(6.10)", "(6.11)"
  — ma il capitolo 6 non contiene alcuna equazione: i numeri erano rimasti
  quelli di prima dello spostamento del metodo L1 nel capitolo 5. Corretti in
  (5.42) [definizione di $F_{\mathcal{S}}$], (5.43) [subgradiente
  generalizzato], (5.44) [vettore $z_k$], (5.45) [faccia ortante $\Omega_k$],
  (5.46) [active set $\mathcal{A}_k$], (5.50) [condizione di Armijo
  proiettata], (5.51) [criterio di arresto del CG] — tutti verificati contro i
  tag reali delle equazioni. Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**123 pp**, 0 errori, 0 undefined, overfull solo preesistenti).
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.
- **Ultimo intervento (25/08/2026).** **Tesi Appendice B.4: titolo semplificato,
  rimosso "Codice BB-CCV".** Il titolo della sottosezione B.4 passa da "Codice
  BB-CCV (Barzilai--Borwein con campionamento dinamico)" a "Barzilai--Borwein
  con campionamento dinamico" (resta solo la parte tra parentesi). Nessun
  riferimento da aggiornare: la sottosezione non ha `\label{app:bbccv}` (i
  riferimenti all'algoritmo usano già `sec:bbccv`) e nessun testo cita il
  vecchio titolo per esteso. Ricompilati `tesi.pdf` e `tesi_finale.pdf`
  (**123 pp**, 0 errori, 0 undefined, overfull solo preesistenti).
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).
  `bozza.tex` non toccata.
- **Ultimo intervento (25/08/2026).** **Tesi: rimossi i termini "isotropa" e
  "anisotropia" dal paragrafo "Osservazioni comparative" (Sez. 6.3), riscritti
  in parole semplici (ma da tesi).** (i) "la curvatura quasi isotropa rende un
  passo scalare ben calibrato in ogni direzione" è diventato "la curvatura
  varia pochissimo da una direzione all'altra, perciò un passo scalare ben
  calibrato è adeguato in ogni direzione"; (ii) "si adatta all'anisotropia,
  dove un passo fisso non può essere..." è diventato "si adatta ai problemi in
  cui la curvatura è molto diversa da una direzione all'altra: un passo fisso
  non può essere...". La spiegazione in parole semplici (direzione
  ripida/piatta) era già nel testo e resta invariata. Nessun contenuto tecnico
  perso. Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**123 pp**, 0 errori, 0
  undefined, overfull solo preesistenti). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata
  (nessuna occorrenza dei termini).
- **Ultimo intervento (25/08/2026).** **Tesi: corretta l'incoerenza del preset
  $\kappa\approx20$ nella Sez. 6.1 (Setup) e precisata la caption della Figura
  BB-CCV.** (1) La frase "L'applicazione offre anche il preset 'Quadratica mal
  condizionata ($\kappa\approx20$)', non usato nei test riportati in questo
  capitolo" era **falsa**: il preset $\kappa\approx20$ È usato negli esperimenti
  sul riuso del mini-batch (Sez. 6.7, tabelle `tab:riuso_malcond_gd/_ncg/_l1/_bb`
  e Risultati numerici). L'elenco delle funzioni test della Sez. 6.1 è stato
  ristrutturato in **quattro preset quadratici** (ben condizionato $\kappa\approx1.1$,
  mal condizionato $\kappa\approx20$, molto mal condizionato $\kappa\approx100$,
  termine incrociato), con l'indicazione che i primi tre sono usati nei Risultati
  Numerici (Sez. 6.3) e tutti e quattro negli esperimenti sul riuso (Sez. 6.7).
  Aggiunti gli autovalori della Hessiana per giustificare la denominazione dei
  preset: ben condizionata (Hessiana $\begin{psmallmatrix}2&0.1\\0.1&2\end{psmallmatrix}$,
  autovalori $2.1$/$1.9$, $\kappa\approx1.1$: il termine $0.1\,w_1w_2$ è un piccolo
  accoppiamento) vs termine incrociato (Hessiana $\begin{psmallmatrix}2&0.5\\0.5&2\end{psmallmatrix}$,
  autovalori $2.5$/$1.5$, $\kappa\approx1.67$: l'accoppiamento $0.5(w_1-1)(w_2+2)$ è la
  caratteristica dominante). (2) Caption Figura BB-CCV (Cap. 5): "nei tre
  esperimenti della Sezione 6" -> "nei tre problemi dei Risultati Numerici
  (Sezione 6.3)", per non confonderli con i quattro preset della parte sul
  riuso. (3) Verificata l'Appendice D: "tre problemi test, tre valori di
  $\theta$, quattro algoritmi" è corretta (descrive `sim_exp.py`:
  `probs=[build_ls(), build_ls(kappa_goal=1), build_rosen()]`, $\theta\in\{0.1,0.5,0.9\}$,
  confronto di 4 metodi) — nessuna modifica. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**123 pp**, 0 errori, 0 undefined, overfull solo
  preesistenti). Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf
  (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (25/08/2026).** **App web: nuovo criterio adattivo
  "Riuso per discesa della loss sul batch" per la durata del riuso del
  mini-batch, strutturalmente identico allo stop adattivo con validation
  set.** In `visualizzazione.html` (file solo nella repo, nessun PDF
  coinvolto) aggiunta la checkbox **"Usa la discesa della loss sul batch per
  lo stop del riuso"** (subito sotto quella della validation, mutuamente
  esclusiva: spuntare una deseleziona l'altra) con pannello iperparametri
  dello stesso stile della validation: **tolleranza relativa** `descTol`
  (default 1e-4), **soglia minima assoluta** `descMinAbs` (default 0),
  **pazienza** `descPatience` (default 1) e **frequenza valutazione**
  `descFreq` (default 1). Quando attivo, M (e M_H per i metodi di Newton)
  non si impostano più a mano: il riuso è forzato e i controlli manuali
  nascosti, come per la validation. **Criterio (versione firmata, NON in
  modulo):** si continua a riusare lo stesso mini-batch finché
  $J_{\\mathcal{S}}(w_{k+1}) \\le J_{\\mathcal{S}}(w_k) - \\text{tol}\\,|J_{\\mathcal{S}}(w_k)| - \\text{min}\\_\\text{abs}$
  (riduzione relativa tra iterati **consecutivi sullo stesso batch**: il
  confronto è sempre tra loss dello stesso $\\mathcal{S}$, $J_{\\mathcal{S}}^{\\text{prev}}$
  viene azzerata a ogni cambio batch). Un **aumento** della loss non è
  progresso → ricampiona (questo è il motivo per cui non si usa il modulo).
  Nessun validation set: i mini-batch si campionano dall'intero dataset e la
  CCV ha tetto $N$ (come nel riuso manuale); la CCV resta ortogonale
  (governa la dimensione, il nuovo criterio la durata). **Per Newton-L1 la
  funzione monitorata è $F_{\\mathcal{S}} = J_{\\mathcal{S}} + \\nu\\|w\\|_1$**
  (la stessa che la line search proiettata minimizza). Generazione codice per
  i 4 algoritmi (`generate*Descent`, dispatch in `generateGD/BB/NewtonCG/
  NewtonL1`), `preLoop`/`sample`/`evalDescent` in `descentSnippets`,
  `M_actual` e storico `desc_hist` restituiti come la validation, nuova card
  metrica **J_batch (ultima valutazione)**, nuovo grafico **"Loss sul batch
  J_batch(w_k) — riuso per discesa"**, caption della Traiettoria 2D con i 4
  iperparametri e pseudocodice dinamico (teoria) per GD, BB, Newton-CG e
  Newton-L1 (righe `J_{\\mathcal{S}}=J_{\\mathcal{S}_k}(w_{k+1})`,
  `J_{\\mathcal{S}}^{\\text{prev}}` e pazienza $p_{\\mathcal{S}}/p_{\\mathcal{H}}$).
  Validazione: sintassi JS (parse `new Function` + `deno check` sul blocco
  `<script>`), smoke test con DOM fittizio (top-level senza errori), generazione
  Python effettiva dei 4 algoritmi (varianti Wolfe/Armijo, H legata/indipendente)
  eseguita end-to-end su quadratica (seed 42): `desc_hist` popolata, `m_actual`
  con valori > 1 (il riuso avviene) e `resample_pts` non vuoti nei casi
  Armijo/Newton-CG (il criterio di discesa ricampiona davvero), più unit test
  della versione firmata (discesa → migliora; aumento → ricampiona; passo
  piatto → ricampiona). `bozza.tex` non toccata.
- **Ultimo intervento (24/08/2026).** **Sez. 6.7.6 e 6.7.7: precisato il ruolo
  di $\tau$ (tolleranza) nella frequenza di ricampionamento e corretto il
  conteggio degli iperparametri dello stop adattivo (5 → 6).** (1) Il
  paragrafo "Quali iperparametri vale la pena testare" ora spiega che
  pazienza e tolleranza controllano la frequenza attraverso il criterio
  $J_{\mathrm{val}} \le J_{\mathrm{val}}^{\mathrm{best}}(1-\tau)-\epsilon_{\mathrm{abs}}$:
  un $\tau$ più grande alza la soglia di miglioramento e ricampiona più
  spesso, uno più piccolo ricampiona meno spesso; e anticipa che
  nell'intervallo testato $(10^{-5},\,10^{-3})$ l'effetto è ininfluente
  (come mostrato dalla Tab. 6.23). (2) "L'applicazione espone cinque
  iperparametri" era errato: l'app ne espone **sei** (aggiunta la soglia
  assoluta $\epsilon_{\mathrm{abs}}$, campo `valMinAbs`, default 0, che
  compariva già nella formula del criterio ma non era elencata). Corretta
  anche l'occorrenza gemella in Sez. 6.7.7 ("Restano quindi da scegliere i
  sei iperparametri..."). (3) Sistemato il verbo mancante nella frase "La
  percentuale $p$ [bilancia] la qualità della stima di $J_{\mathrm{val}}$...".
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**123 pp**), 0 errori, 0
  undefined, overfull solo preesistenti (8). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.

- **Ultimo intervento (24/08/2026).** **Corrette 4 incoerenze fattuali nella
  Sezione 6.7 (riuso) e 1 riferimento tecnico nella Tabella 5.2.** (1) Caption
  della Tabella 6.21: l'esempio "su κ≈1.1 con M=∞ il riuso migliora su tutti
  e 5 i seed per tutti gli algoritmi" era falso (per Newton-CG e Newton-CG
  $L_1$ è 4 su 5, verificato dalla stessa tabella) → riscritto con i valori
  esatti (5/5 per GD e BB-CCV, 4/5 per i Newton). (2) Stessa caption:
  "rispetto alla colonna base della Tabella 6.20" era impreciso → "rispetto
  alla versione base con la stessa seed" (il confronto è per-seed, non contro
  i valori seed-42 della 6.20). (3) Sez. 6.7.2 (Meccanismo): "Fa eccezione il
  problema con termine incrociato" per Newton-CG era incompleto → aggiunto
  anche il caso ben condizionato (migliora su 4 seed su 5). (4) Sez. 6.7.4
  (Risultati numerici): "Newton-CG è penalizzato quasi universalmente, con
  l'eccezione del termine incrociato" contraddiceva la Tabella 6.21 (κ≈1.1:
  4/5 con M=∞) → riscritto: penalizzato sui mal condizionati (0/5 su κ≈20 e
  κ≈100), aiutato su ben condizionato (4/5) e termine incrociato con M=∞.
  (5) Tabella 5.2: "criterio di arresto (5.35)--(5.36)" → "(5.35a)--(5.36)"
  (la (5.35) esiste solo come (5.35a)/(5.35b)). Verificati invece come
  coerenti: tutte le affermazioni numeriche delle Tabelle 6.1-6.3 (12/11/16
  iterazioni, 4×10⁻¹, 1.099×10⁻¹⁴, 6.164×10⁻⁴), la sezione validation
  (medie 2.37/2.87/10.7/2.53, effetti marginali P/τ/p/f/strategia,
  vittorie 14/13/14) e i riferimenti incrociati. Ricompilati `tesi.pdf` e
  `tesi_finale.pdf` (**123 pp**), 0 errori, 0 undefined, overfull solo
  preesistenti (8). Sincronizzati nella repo tesi.tex/tesi.pdf/
  tesi_finale.pdf (md5 verificati). `bozza.tex` non toccata.

- **Ultimo intervento (24/08/2026).** **Risolte 9 incoerenze "tre algoritmi" vs
  "quattro algoritmi" nel documento.** In `tesi/tesi.tex` il documento usa
  ovunque la lista canonica di 4 algoritmi (Dynamic GD, Newton-CG,
  Newton-CG~$L_1$, BB-CCV), ma 8 punti dicevano ancora "i tre algoritmi"
  escludendo BB-CCV: Abstract, Sez. 1.4 (Contributi, punto 3),
  Sez. 6.1 (Setup: "i tre implementati nell'applicazione ... e il metodo
  BB-CCV", dove BB-CCV è invece implementato nell'app), Sez. 6.2
  (Architettura, testo e bullet), Sez. 6.4 (bullet "confrontare i tre
  algoritmi"), Conclusioni e Appendice B (che contiene 4 codici, B.1-B.4).
  Tutti aggiornati a "quattro algoritmi" (+BB-CCV dove l'elenco era esplicito).
  Inoltre la caption della Tabella 5.2 diceva "Confronto dei tre metodi
  proposti nel Capitolo 5", ambigua perché il Capitolo 5 propone anche
  BB-CCV (Sez. 5.4): riscritta in "Confronto dei tre metodi a campione
  dinamico con analisi di complessità ... il metodo BB-CCV della
  Sezione~\ref{sec:bbccv} è escluso perché non ha un bound di complessità
  totale" (verificato: la Sez. 5.4 non ha analisi di complessità). Lasciati
  invariati i "tre" corretti: "tre aspetti" (1.3), "tre problemi test"
  (esperimenti: il preset κ≈20 non è usato), "tre colonne/casi/indicazioni",
  Tab. 5.1 "Bound di complessità per tre metodi" (dinamico vs fisso vs SGD).
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**123 pp**), 0 errori, 0
  undefined, overfull solo preesistenti (8). Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.

- **Ultimo intervento (24/08/2026).** **Sez. 6.7.6: corretto il riferimento
  sbagliato alla Sezione 6.4 nel paragrafo dello stop adattivo.** In
  `tesi/tesi.tex` la frase "Il meccanismo è quello descritto nella
  Sezione~\ref{sec:visualizzazione}" (che risolveva alla Sez. 6.4
  *Visualizzazione Interattiva*, la quale NON descrive lo split
  train/validation) è diventata "Il meccanismo è il seguente": il paragrafo
  è autosufficiente e il riferimento era sia sbagliato sia ridondante
  (il meccanismo è descritto per intero nella stessa frase). La parte CCV
  con tetto $\lvert\mathcal{T}\rvert$ invece di $N$ resta verificata sul
  codice dell'app (righe 2019/2033/2049 di `visualizzazione.html`).
  Ricompilati `tesi.pdf` e `tesi_finale.pdf` (**123 pp**), 0 errori, 0
  undefined, overfull solo preesistenti. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati). `bozza.tex` non
  toccata.

- **Ultimo intervento (24/08/2026).** **App web: nella caption della
  Traiettoria 2D ora compare anche l'errore finale
  $\|w_K-w^*\|_2$ dopo `max_iter` iterazioni.** In `visualizzazione.html`
  (file solo nella repo, nessun PDF coinvolto) `buildColabCaption()` aggiunge,
  subito dopo `... iter · batch finale n=...`, il blocco
  `err finale ‖w−w*‖=<valore>` (es. `err finale ‖w−w*‖=1.010e-1`), dove il
  valore è `currentData.errs[currentData.errs.length-1]` — la metrica
  $e_k=\|w_k-w^*\|_2$ già calcolata dal codice Python generato (stessa
  notazione del riquadro di convergenza). Poiché `history` contiene
  `max_iter+1` punti (da $w_0$ a $w_{max\_iter}$), l'ultimo valore è
  l'errore dopo `max_iter` iterazioni qualunque sia il valore scelto
  dall'utente. Il blocco è difensivo (`errs` può mancare → nessun crash e
  nessun errore in caption) e vale sia per la caption mostrata sotto il
  grafico sia per le immagini salvate in galleria (`buildColabCaption` è
  usata da entrambe). Validato: sintassi del file OK (parse `new Function`
  via deno) e harness deno con DOM fittizio sulle funzioni reali
  estratte dal file — 4/4 scenari PASS (GD riuso no con `1.010e-1`; riuso
  illimitato + traiettoria quasi ferma con `2.300e-3` e warning passi reali;
  Newton-CG L1 con iperparametri R/maxcg/ν/σ/η; `currentData` senza `errs`
  → caption senza errore e senza crash). Nessun sorgente LaTeX toccato.

- **Ultimo intervento (24/08/2026).** **Nuova Figura 6.2: stessa Traiettoria 2D
  della Figura 6.1 ma sul problema quadratico molto mal condizionato
  ($\kappa\approx100$).** In `tesi/tesi.tex` aggiunta la nuova figura
  `fig:riuso_traiettorie_malcond` subito sotto la Figura~6.1
  (`fig:riuso_traiettorie`): screenshot del pannello *Traiettoria 2D*
  dell'applicazione con due esecuzioni di Dynamic GD sul preset **Quadratica
  molto mal condizionata (κ≈100)** — in alto ricampionamento a ogni
  iterazione (riuso disattivato), in basso riuso illimitato dello stesso
  mini-batch ($M=\infty$), $w_0=(2,-3)$, $\alpha=0.1$, $\theta=0.5$, $n_0=5$,
  30 iterazioni, seed 42. Caption speculare a quella della 6.1. Aggiornato
  anche il paragrafo introduttivo, che ora rimanda a entrambe le figure
  (ben condizionata vs molto mal condizionata). Nuovo file
  `tesi/fig_riuso_traiettorie_malcond.png` (da `Screenshot 2026-08-24 alle
  18.30.23.png`, identico). Ricompilati `tesi.pdf` (**123 pp** da 122) e
  `tesi_finale.pdf` (**123 pp**), 0 errori, 0 undefined, overfull solo
  preesistenti. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf/fig_riuso_traiettorie_malcond.png/README.md
  (md5 verificati). `bozza.tex` non toccata.

- **Ultimo intervento (24/08/2026).** **Tabella 6.20 (`tab:riuso_sintesi`):
  sfondo colorato per cella con scala verde → rosso (logaritmica).** In
  `tesi/tesi.tex` le 112 celle numeriche della tabella di sintesi del riuso
  (16 righe × 7 colonne: `base`, $M{=}\infty$, $M{=}10$, $M{=}5$, $M{=}3$,
  $M{=}2$, $M{=}1$) hanno ora uno sfondo `\cellcolor[RGB]{...}` che codifica
  l'errore finale $e_{30}$: scala **logaritmica** normalizzata tra il minimo
  ($5.53\times10^{-3}$, verde chiaro) e il massimo ($1.28\times10^{0}$, rosso
  scuro), gradazione continua in mezzo. Per abilitare `\cellcolor` il
  preambolo carica ora `\usepackage[table]{xcolor}`. Didascalia aggiornata con
  la spiegazione della scala. Dati, simboli $\blacktriangle$/$\blacktriangledown$/$=$
  e testo delle righe INVARIATI (verificato: 112 valori e simboli identici).
  Ricompilati `tesi.pdf` (**121 pp**) e `tesi_finale.pdf` (**121 pp**), 0
  errori, 0 undefined, overfull solo preesistenti. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf/README.md (md5 verificati). `bozza.tex`
  non toccata.

- **Ultimo intervento (24/08/2026).** **Scala logaritmica globale dei colori su
  TUTTE le tabelle della Sezione 6 + tabella 6.21 spiegata meglio + nuova
  Figura 6.1.** (1) In `tesi/tesi.tex` le 4055 celle di errore delle 36
  tabelle della Sezione 6 (6.1–6.3, dettaglio riuso 6.4–6.19, sintesi 6.20,
  validation 6.22–6.24, consigliati 6.26–6.34) ora hanno lo sfondo colorato
  con la macro `\colorcell{mantissa}{esponente}` (preambolo), che calcola il
  colore a compile-time con pgfmath su una **scala logaritmica globale**: il
  verde chiaro corrisponde all'errore minimo assoluto $1.0991\times10^{-14}$
  (Tabella 6.2, BB-CCV $\kappa\approx100$, k=11) e il rosso scuro al massimo
  $1.4142$ (punto iniziale comune). La tabella 6.20 passa dal vecchio
  `\cellcolor[RGB]` precalcolato alla stessa macro (colori ora su scala
  globale) e la didascalia è aggiornata. Aggiunta una nota introduttiva sui
  colori in "Risultati Numerici". (2) **Tabella 6.21 (`tab:riuso_robustezza`)
  ristrutturata**: le colonne $M{=}\infty$ e $M{=}10$ hanno ora tre
  sottocolonne esplicite \emph{Migl.}/\emph{Pegg.}/\emph{Uguale} (i "5 / 0 /
  0" sono divisi in celle separate) e la didascalia spiega la lettura
  (numero di seed su 5 in cui il riuso migliora/peggiora/lascia invariato
  $e_{30}$ rispetto alla base). (3) **Figura 6.1 sostituita**: il vecchio
  `download.png` (traiettorie ben/mal condizionato) è rimosso e al suo posto
  c'è `tesi/fig_riuso_traiettorie.png` (screenshot del pannello *Traiettoria
  2D* dell'app: Dynamic GD su $\kappa\approx1.1$, in alto ricampionamento, in
  basso riuso illimitato $M{=}\infty$; $w_0=(0.11,-1.66)$, seed 42, punti
  CCV violata marcati), con prosa e didascalia aggiornate. Ricompilati
  `tesi.pdf` (**121 → 122 pp**) e `tesi_finale.pdf` (**122 pp**), 0 errori, 0
  undefined, overfull solo preesistenti. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf/fig_riuso_traiettorie.png (nuovo)/
  README.md; `tesi/download.png` rimosso (non più referenziato; md5
  verificati). `bozza.tex` non toccata.

- **Ultimo intervento (24/08/2026).** **Scala colori della Sezione 6 resa
  robusta agli outlier.** Come richiesto, gli ancoraggi della scala
  logaritmica in `tesi/tesi.tex` non usano più il minimo assoluto
  ($1.0991\times10^{-14}$, outlier di precisione macchina di BB-CCV) ma il
  **1° percentile degli errori ($\approx 7.4\times10^{-3}$)** come verde
  chiaro, mantenendo il massimo $1.4142$ come rosso scuro: i valori sotto il
  1° percentile si saturano in verde e la gradazione si distribuisce su tutta
  la gamma dei valori (363 tonalità distinte nel PDF, 41 celle verde chiaro,
  127 rosso scuro). Aggiornate la nota in "Risultati Numerici" e la didascalia
  della Tabella 6.20. Ricompilati `tesi.pdf` (**122 pp**) e `tesi_finale.pdf`
  (**122 pp**), 0 errori, 0 undefined, overfull solo preesistenti.
  Sincronizzati nella repo tesi.tex/tesi.pdf/tesi_finale.pdf/README.md (md5
  verificati).

- **Ultimo intervento (24/08/2026).** **Tabelle del riuso dei metodi di Newton:
  ripristinato il formato con intestazione descrittiva e didascalie complete
  (revert delle righe in testa $M$/$M_H$/$H_k$).** In `tesi/tesi.tex` le 8
  tabelle principali (`tab:riuso_*_ncg/_l1`) e le 8 dei consigliati
  (`tab:riuso_cons_*_ncg/_nl1`) tornano al formato precedente il commit
  `f5e1ee0`: intestazione a riga unica (`base`, `$M{=}\infty$`, `$M{=}10$`,
  `$M{=}5$`, `$M{=}2$`, `H ind. $M_H{=}\infty$`; per i consigliati `k`, `base`,
  `consigliato`), rimosse le 3 righe in testa (`$M$`, `$M_H$`, `$H_k$`) e
  ripristinate le didascalie complete (spiegazione dell'ultima colonna in
  modalità *Indipendente da $S_k$*; per i consigliati la configurazione
  consigliata: stop adattivo `$P{=}1$, $f{=}1$, $p{=}10\%$, split fissa` per
  Newton-CG, riuso `$M{=}3$` per Newton-CG $L_1$). Rimossa la legenda sulle
  "righe in testa" dal Setup della Sezione 6.7. Dati delle tabelle invariati
  (verificato: 0 righe dati modificate). `altro/script/gen_tabelle_riuso.py`
  riallineato (il `--tex` scrive di nuovo il formato descrittivo). Ricompilati
  `tesi.pdf` (**122 → 121 pp**) e `tesi_finale.pdf` (**121 pp**), 0 errori, 0
  undefined, overfull solo preesistenti. Sincronizzati nella repo
  tesi.tex/tesi.pdf/tesi_finale.pdf (md5 verificati).

- **Ultimo intervento (24/08/2026).** **Appendice E eliminata: contenuto
  spostato nella Sezione 6 della tesi + righe $M$/$M_H$/modalità nelle tabelle
  dei metodi di Newton.** (1) In `tesi/tesi.tex` il segnaposto dell'Appendice E
  è stato rimosso e il contenuto integrale di `appendice_riuso.tex` è ora la
  sottosezione **6.7** "Riuso del mini-batch: iterazioni consecutive sullo
  stesso campione" (`\subsection`, label `app:riuso`), con le 7 ex-sottosezioni
  E.1–E.7 come `\subsubsection` 6.7.1–6.7.7 (label `app:riuso-*` invariate) e
  34 tabelle numerate nella Sezione 6. Testo: "Questa appendice" →
  "Questa sottosezione". "Struttura della tesi": "Appendici A--E" →
  "Appendici A--D". Tutti i `\ref` interni risolti (0 undefined).
  (2) **Tabelle dei metodi del secondo ordine**: nelle 8 tabelle Newton
  principali (`tab:riuso_*_ncg/_l1`) e nelle 8 dei consigliati
  (`tab:riuso_cons_*_ncg/_nl1`) l'intestazione è semplificata (`base`,
  `∞`, `10`, `5`, `2`, `H ind.`) e 3 righe in testa riportano il riuso del
  batch `$M$` (base 1), il riuso dell'Hessiana `$M_H$` (in modalità legata
  segue il batch: = $M$; colonna "H ind.": $M_H=\infty$) e la modalità
  (`$H_k$` legata/indipendente da $S_k$). Didascalie alleggerite (via la lunga
  spiegazione della colonna H ind.; per i consigliati il rimando è alla tabella
  di sintesi `tab:riuso_cons_sintesi`) e legenda delle righe aggiunta in E.3
  (Setup). Fix
  grammaticale nelle didascalie consigliati ("ben condizionata" → "ben
  condizionato"). (3) Ritirati i sorgenti autonomi: `tesi/appendice_riuso.tex`
  e `tesi/appendice_riuso_estratto.tex/.pdf` spostati in `altro/` (riferimento
  storico). (4) `altro/script/gen_tabelle_riuso.py` aggiornato (`--tex` scrive
  le 8 tabelle Newton col nuovo formato; default `altro/appendice_riuso.tex`).
  Ricompilati `tesi.pdf` (**83 → 122 pp**) e `tesi_finale.pdf` (**122 pp**), 0
  errori, 0 riferimenti undefined (overfull solo preesistenti). Sincronizzati
  nella repo tesi.tex/tesi.pdf/tesi_finale.pdf/README.md + spostamenti in
  altro/ (md5 verificati). `bozza.tex` non toccata.
- **Ultimo intervento (23/08/2026).** **Appendice E: nuova sottosezione E.6
  "Stop adattivo con validation set" con sweep degli iperparametri.** In
  `tesi/appendice_riuso.tex` aggiunta la sottosezione E.6
  (`app:riuso-validation`) con tre tabelle generate
  (E.19–E.21). Il meccanismo è lo stop adattivo dell'app (commit
  `e13a126`): il mini-batch viene ricampionato quando la loss sul validation
  set non migliora per `P` valutazioni consecutive. Esperimento: 4 problemi ×
  4 algoritmi (codice Python ESATTO generato dall'app — varianti
  `*Validation` estratte con harness Deno e salvate in
  `altro/script/validation_codice_generato/`), sweep su $P\in\{1,3,8\}$,
  $\tau\in\{10^{-5},10^{-4},10^{-3}\}$, $p\in\{0.1,0.2,0.3\}$,
  $f\in\{1,3\}$, strategia fisso/dinamico (108 config). Risultati: la
  pazienza bassa vince ($P{=}1$: media e30 $2.63\times10^{-1}$ vs $3.45$
  di $P{=}8$), la tolleranza è ininfluente, $p{=}10\%$ meglio di $20/30\%$,
  $f{=}1$ meglio di $3$, split dinamico meglio di fisso. La configurazione
  $P{=}1,p{=}10\%$,dinamico batte la \emph{base} (ricampionamento a ogni
  iterazione, stesso training set) in **11/16 caselle su 5 seed** (media e30
  $2.24\times10^{-1}$ vs $2.47$), ed evita il collasso di $M{=}\infty$ per
  Newton-CG (es. termine incrociato: $1.41 \to 7.88\times10^{-2}$, migliore
  persino della base). Nuovi riferimenti `base_train`/`minf_train`
  (ricampionamento a ogni iterazione / riuso illimitato sul solo training
  set) per un confronto equo con lo stop adattivo (che tiene fuori $p$ dei
  dati). Nuovo script riproducibile
  `altro/script/gen_tabelle_riuso_validation.py` (`--data`, `--analisi`,
  `--robustezza`, `--tex`) con i dati `validation_sweep.json` e
  `validation_robustezza.json`; `tesi/appendice_riuso_estratto.tex` ora
  definisce anche la label fittizia `app:codici` (riferimento all'Appendice
  B). Ricompilato `appendice_riuso_estratto.pdf`: 22 -> **25 pp**, 0 errori,
  0 undefined, 0 overfull, 1 underfull preesistente. `tesi.tex` non toccata
  (l'Appendice E resta un PLACEHOLDER): `tesi.pdf`/`tesi_finale.pdf` non
  coinvolti (latexmk "nothing to do").

- **Ultimo intervento (23/08/2026).** **App web: rimosso \"(come nel Colab)\"
  dall'etichetta del pulsante \"Traiettoria 2D\".** In
  `visualizzazione.html` (file solo nella repo, nessun PDF coinvolto) il
  pulsante `colabPlotBtn` ora si chiama semplicemente \"🖼️ Traiettoria 2D\"
  (senza il suffisso \"(come nel Colab)\"). Solo testo dell'etichetta:
  nessun cambiamento di comportamento, id, JS o CSS.
- **Ultimo intervento (24/08/2026).** **Appendice E: nuova sottosezione E.7
  "Iperparametri consigliati per ciascun metodo" con ricerca a griglia.** In
  `tesi/appendice_riuso.tex` aggiunta la sottosezione E.7
  (`app:riuso-consigliati`) con: (1) breve introduzione al validation set
  (split train/validation, criterio su $J_{\mathrm{val}}$, $P$, $f$, $p$,
  $\tau$, strategia fissa/dinamica); (2) la **configurazione consigliata per
  ciascun metodo**, scelta con ricerca a griglia sul codice esatto dell'app
  (4 preset quadratici × 5 seed, metrica mediana di $e_{30}$): **Dynamic GD →
  riuso $M=5$** (14/20 vittorie), **Newton-CG → stop adattivo $P=1$, $f=1$,
  $p=10\%$, strategia fissa** (13/20), **Newton-CG $L_1$ → riuso $M=3$**
  (14/20), **BB-CCV → base** (nessuna config testata migliora tutti i
  problemi); nessun metodo peggiora la mediana di $e_{30}$ su nessun preset;
  (3) tabella di sintesi E.22 (mediane 5 seed base vs consigliato + vittorie)
  e 12 tabelle per-iterazione E.23–E.34 (seed 42, base vs consigliato, stile
  E.1–E.16). Nota metodologica: BB-CCV è testato con line search **Armijo**
  (come nella tesi Sez. 6.1; verificato che la Tab. 6.1–6.3 BB coincide col
  codice attuale dell'app con Armijo, mentre la vecchia tabella E.10
  riportava valori incoerenti con la tesi). Generazione dati: script in `/tmp`
  (estrazione generatori dall'app via deno + numpy, 4 preset × 5 seed).
  Ricompilato `appendice_riuso_estratto.pdf`: **25 → 39 pp**, 0 errori, 0
  undefined, 0 overfull, 1 underfull preesistente. Sincronizzati nella repo
  tesi/appendice_riuso.tex e tesi/appendice_riuso_estratto.pdf (md5
  verificati). `tesi.tex` non toccata (l'Appendice E resta un PLACEHOLDER):
  `tesi.pdf`/`tesi_finale.pdf` non coinvolti.
- **Ultimo intervento (23/08/2026).** **App web: nella Traiettoria 2D una
  croce ✕ (ambra) segna i punti in cui il mini-batch è stato ricampionato
  per cause NON-CCV.** In `visualizzazione.html` (file solo nella repo,
  nessun PDF coinvolto) i generatori Python (GD, BB, Newton-CG, Newton-L1,
  sia con riuso sia con validation) ora producono anche una lista
  `resample_pts` che registra i punti in cui si ricampiona per **stop
  adattivo** (validation) o per **`max_consec`/k finito** nel riuso; la CCV
  continua a scrivere solo in `resize_points` (pallini ciano, batch
  aumentato). Nel plot 2D questi punti compaiono come **croci ✕ ambra**
  (legenda \"Ricampionamento (non-CCV)\") sopra la traiettoria, in aggiunta
  ai pallini ciano esistenti. Il codice base SENZA riuso resta
  byte-identico (`resample_pts` compare solo nei rami reuse/validation).
  Validazione: sintassi dello script (deno `new Function`) OK; harness deno
  che estrae i codici generati + esecuzione numpy su 12 configurazioni
  (base / riuso k=3 / riuso illimitato / validation per i 4 metodi):
  traiettorie **identiche** a HEAD (`history`, `batch_sizes`,
  `resize_points`) in tutte le configurazioni; `resample_pts` non vuoto con
  k=3 e con validation patience=1, vuoto con riuso illimitato, assente
  senza riuso. Nessun sorgente LaTeX toccato: nessun PDF da ricompilare.
- **Ultimo intervento (23/08/2026).** **App web: la caption dei salvataggi
  (box sotto la traiettoria e galleria) ora riporta gli iperparametri del
  validation set quando lo stop adattivo è attivo.** In
  `visualizzazione.html` (file solo nella repo, nessun PDF coinvolto) la
  funzione `buildColabCaption()` — che genera la caption degli iperparametri
  mostrata sotto la Traiettoria 2D e salvata nelle immagini della galleria
  di confronto — include, quando la checkbox \"Usa validation set per stop
  adattivo\" è attiva, tutti gli iperparametri del validation set:
  `val_pct=..%`, `pazienza=..`, `freq=..`, `tol=..`, `min_abs=..` e la
  **strategia** (`val fixed`/`val dynamic`). Con la checkbox spenta la
  caption è invariata (nessun parametro validation). I valori si leggono
  dagli stessi controlli della UI, quindi la caption riflette sempre la
  configurazione corrente (stessa logica degli altri iperparametri). La
  strategia fixed↔dynamic compariva già nel codice Python generato
  (`val_strategy = 'fixed'/'dynamic'`) e nel pseudocodice della teoria
  (nota `\mathcal{V},\mathcal{T}` fissi/ricampionati): verifica confermata
  che il cambio altera codice generato e comportamento (resampling del
  validation set a ogni cambio batch). Validazione: sintassi dello script
  principale verificata (compile `new Function`, deno); harness con DOM
  fittizio **3/3 scenari** (validation OFF → nessun parametro validation in
  caption; ON+fixed e ON+dynamic → caption con `val_pct/pazienza/freq/tol/
  min_abs` e strategia corretta). Nessun sorgente LaTeX toccato: nessun PDF
  da ricompilare. Commit e push della repo.
- **Ultimo intervento (23/08/2026).** **App web: stop adattivo con validation
  set — M e M_H decisi automaticamente dalla loss di validazione.** In
  `visualizzazione.html` (file solo nella repo, nessun PDF coinvolto) nuova
  sezione "Stop adattivo con validation set — tutti gli algoritmi" con la
  checkbox **"Usa validation set per stop adattivo"**. Quando attiva:
  (1) i controlli manuali di **M** (`maxConsecRow`) e **M_H** (`mhContainer`)
  vengono nascosti e il riuso del batch è forzato; compaiono gli iperparametri
  **Percentuale validation** (slider 5–50%, default 20%), **Tolleranza relativa**
  (default 1e-4), **Pazienza** (slider 1–10, default 3), **Frequenza
  valutazione** (1–10, default 1), **Soglia minima assoluta** (default 0.0) e
  **Strategia validation** (fixed/dynamic, default fixed). (2) I generatori
  `generateGD/BB/NewtonCG/NewtonL1` instradano alle nuove varianti
  `*Validation`: il dataset è separato in training/validation **una volta
  all'inizio (fixed)** o **a ogni cambio di batch (dynamic)**, i mini-batch si
  campionano dal SOLO training set (con tetto della CCV a `len(train_idx)`),
  e dopo ogni `w_{k+1}` (anche con step=0) si valuta `J_val`: se
  `J_val ≤ J_val^best·(1−tol) − min_abs` si resetta la pazienza, altrimenti
  dopo `patience` valutazioni senza progresso si ricampiona il batch (e
  l'Hessiana). La **CCV resta invariata** e continua a ingrandire il batch.
  Per i metodi di Newton: Hessiana legata → M_H deciso insieme a M; Hessiana
  indipendente → M_H deciso separatamente con lo **stesso criterio** (doppio
  contatore `patience_S`/`patience_H`, che col singolo slider coincidono).
  (3) Nuove metriche **M_actual** (iterazioni consecutive sullo stesso batch)
  e **J_val** e nuovo grafico "Loss di validation J_val(w_k)" (card dedicata,
  visibile solo con validation attiva). (4) Pseudocodice LaTeX della teoria
  aggiornato con le righe validation. Quando la checkbox è spenta tutto torna
  al comportamento precedente (M/M_H manuali). Validazione: `deno check` OK;
  harness con DOM fittizio (deno) per estrarre i codici Python generati;
  **11/11 combinazioni eseguite con numpy** (GD/BB/Newton-CG/Newton-L1 ×
  validation on/off, fixed/dynamic, H legata/indipendente, Hessian-free L1):
  split train/val disgiunto (fixed: 1 sola permutazione; dynamic: 1 +
  campionamenti), batch mai pescato dal validation set, `m_actual`/`val_hist`
  allineati alla history, **pazienza verifica la frequenza dei cambi batch**
  (segmenti medi 1.02 / 3.00 / 7.50 per patience 1/3/8 con CCV spenta),
  CCV che continua a far crescere `n`. Nessun sorgente LaTeX toccato: nessun
  PDF da ricompilare. Commit e push della repo.
- **Ultimo intervento (23/08/2026, follow-up).** **Fix: nella teoria dinamica
  (pseudocodice LaTeX generato) ogni riga compariva con `\n` davanti.** Le 4
  funzioni `pseudo*Validation` aggiunte sopra univano le righe con `'\n'`
  letterale (doppio backslash nel sorgente, `.join('\\n')`) invece che con il
  newline reale (`.join('\n')` come le funzioni originali): il separatore
  finiva nel testo HTML e MathJax lo mostrava come `\n` all'inizio di ogni
  riga. Corretti i 4 `.join`. Validazione: `deno check` OK; harness che
  ispeziona l'output di `pseudo*Validation`: righe ora separate da newline
  reali (prima non si spezzavano con `split('\n')`); i codici Python generati
  (11/11) sono invariati (i generatori non usano queste join). Nessun PDF
  coinvolto (solo `visualizzazione.html`). Commit `23c00e7`.
- **Ultimo intervento (23/08/2026, follow-up).** **App web: curve di livello
  stabili e MAI tagliate a metà dentro la vista.** Segnalati glitch residui:
  con lo zoom/pan le curve comparivano tagliate a metà e in alcuni momenti
  alcune parti non venivano mostrate. Cause e correzioni in
  `visualizzazione.html`: (1) i livelli erano ricalcolati a ogni regrid sulla
  vista → le curve "saltavano" o sparivano; ora sono **FISSI per run** (10
  livelli geometrici in J, uniformi in log10 J, tra un minimo e 2× il massimo
  di J di riferimento) e la superficie del contour è **z = log10(J)** — la
  trasformazione è monotona, quindi le isolinee sono le stesse di J ma i
  livelli geometrici diventano uniformi e Plotly li disegna ESATTAMENTE con
  start/end/size (hover = J originale via `customdata`); (2) durante lo
  zoom-out la griglia dati restava più piccola della vista → curve tagliate al
  bordo dati dentro lo schermo; ora una **griglia "wide"** (n=180, dominio =
  vista a 0.1× + 5%) è calcolata una volta per run (al primo render o zoom-out)
  e usata per ogni zoom ≤ 1× → lo zoom-out è istantaneo e le curve non vengono
  mai tagliate dentro la vista (solo al bordo degli assi, come deve essere);
  (3) il regrid resta solo per lo zoom-in (serve risoluzione), con **margine
  del 40%** e debounce 60 ms → anche i pan oltre il dominio rigenerano senza
  lasciare zone vuote; (4) `pickColabGrid()` sceglie al render la griglia a
  risoluzione più alta tra wide / 1× / ultimo regrid che COPRE la vista.
  Etichette dei livelli disattivate (`showlabels:false`): i valori esatti sono
  nell'hover. Nessun PDF coinvolto (l'app non è in `tesi.tex`). Validazione:
  harness con DOM fittizio **37/37 controlli** (griglia wide e suo dominio,
  zoom-out che usa la wide SENZA regrid, zoom-in con regrid al 40%, pan senza
  loop, reset, 1D no-op) + pipeline completo 2D/1D di `runAlgorithm` estratto
  dall'HTML (livelli log₁₀ uniformi, griglia log10(J)+J originale).
- **Ultimo intervento (23/08/2026).** **App web: le curve di livello della
  Traiettoria 2D ora compaiono SEMPRE, anche zoommando su un punto, e lo zoom
  può uscire fino a 0.1× per vedere tutte le ellissi.** Prima la griglia delle
  isolinee era calcolata UNA volta (55×55) sul range della traiettoria: con lo
  zoom le linee si spezzavano/coarseggiavano, e per i problemi molto mal
  condizionati (κ≈100, ellissi con rapporto assi √κ=10) nemmeno la vista 1×
  mostrava le curve complete (si estendono molto oltre i bounds del preset).
  Ora in `visualizzazione.html`: (1) la griglia viene **rigenerata a ogni
  zoom/pan** sul dominio VISIBILE (con margine 15%) via una funzione Python
  `_grid2d()` riusabile, con **risoluzione adattiva** (n=90–160 punti per lato
  in base allo zoom, griglia iniziale 55→100) → isolinee lisce e sempre
  presenti; (2) i **livelli sono ricalcolati sulla vista** (8 livelli a
  spaziatura radiale in √J, così le ellissi di una quadratica risultano
  equispaziate; tetto a 2× il massimo di J della traiettoria quando la vista
  si spinge oltre, per non far dominare gli angoli lontani) → zoommando vicino
  al minimo o su un punto in salita compaiono comunque curve di livello;
  (3) lo **slider Zoom va da 0.1× a 10×** (prima 1×–10×): a 0.1× si vedono
  TUTTE le curve di livello del problema κ≈100 in un colpo solo; (4) pan
  (trascina) rigenera, "⟲ Vista completa" ripristina la griglia 1× iniziale, e
  i re-render interni (stessi range) NON innescano cicli di regrid (confronto
  dei range con tolleranza nel handler `plotly_relayout`). La traiettoria, i
  marker e l'avviso "passi reali" restano invariati; l'1D è invariato. Nessun
  PDF coinvolto (l'app non è in `tesi.tex`). Validazione: parse JS (deno
  `new Function`) OK; harness con DOM fittizio **27/27 controlli** (zoom-out
  0.1× amplia la vista 10×, regrid con margine e n corretto, zoom-in, pan
  senza loop, reset, 1D no-op); simulazione Python di `_grid2d` su 5 viste
  (iniziale, zoom-out κ≈100, zoom-in al minimo, vista in salita, vista
  remota) + pipeline completo 2D/1D di `runAlgorithm` estratto dall'HTML.
- **Ultimo intervento (23/08/2026).** **App web: la Traiettoria 2D ora rileva e
  segnala le traiettorie "quasi ferme".** In `visualizzazione.html` la funzione
  `colabTrajectoryStats()` deduplica i punti IDENTICI consecutivi della history
  (line search fallita → `step=0.0` → `w` invariato) e considera "quasi ferma"
  una traiettoria con ≤1 passo reale oppure <30% di iterazioni con movimento
  (su run ≥3 iterazioni). In quel caso: (1) la traiettoria viene disegnata sui
  soli punti DISTINTI (niente più cumulo di 30 marker sovrapposti nello stesso
  punto); (2) appare un avviso ambrato con la spiegazione e i suggerimenti
  (line search **Armijo**, **batch₀ più grande**, **k finito** con riuso);
  (3) la caption della immagine salvata in galleria riporta `⚠️ nM/n passi
  reali`. Motivo della segnalazione (diagnosi, NON regressione): con riuso
  batch illimitato + line search Wolfe + batch₀ piccolo, la Wolfe non trova
  passi accettabili (per una quadratica la condizione di curvatura
  `g_new·d ≥ c2·gd` è soddisfatta solo per t≈α, caso borderline perché
  `d'Hd=-gd`), l'algoritmo si blocca, il gradiente resta grande e la CCV sul
  batch riusato non scatta mai (`V/n≈560 ≪ θ²‖g‖²≈6700-8300`) → batch fisso
  per sempre. Riproduzione numerica della config segnalata (Newton-CG, κ≈100,
  riuso illimitato, Wolfe, batch₀=5): 2 punti distinti su 31 (0 movimenti con
  batch₀=20/50); con Armijo 31 punti distinti e ‖w-w*‖=0.079. Il riuso batch è
  stato introdotto il 21/08 (commit `ba21146`): prima di allora la modalità
  senza riuso generava lo stesso identico codice (diff verificato), quindi il
  comportamento vecchio è invariato. Validazione: `deno check` OK + harness
  jsc **14/14 controlli**. Nessun PDF coinvolto (l'app non è in `tesi.tex`).
- **Ultimo intervento (23/08/2026).** **App web: traiettoria 2D con pan
  (trascina), slider di zoom e "Salva immagine" che conserva la vista
  corrente.** In `visualizzazione.html` (file solo nella repo) il plot della
  traiettoria 2D ora: (1) **trascinando sull'immagine si sposta la vista**
  (`dragmode: 'pan'`); (2) ha uno **slider Zoom 1×–10×** con etichetta e
  pulsante **"⟲ Vista completa"** che ripristina la vista al 100%; (3) ha la
  **modebar Plotly disattivata** (`displayModeBar: false`) quindi NON ci sono
  più i tool lasso/select/pan/zoom-to-image; (4) **"Salva immagine per
  confronto" salva ESATTAMENTE la vista mostrata**: se l'utente ha zoommato e
  trascinato (es. solo la parte in alto a destra), l'immagine salvata nella
  galleria (e la sua caption) riflette quella vista — lo snapshot si aggiorna
  via `plotly_relayout` (range x/y correnti) e la vista si conserva tra i
  render, azzerandosi solo con una nuova run (nuovo riferimento
  `currentData`). Validazione: harness con DOM fittizio **42/42 controlli**
  (aggiunti: dragmode pan, modebar nascosta, zoom 2× dimezza l'ampiezza, pan
  aggiorna vista+snapshot, pin salva i range [5,6]/[-4,-3], pin non
  ri-renderizza il plot principale, reset vista completa, nuova run azzera la
  vista). Nessun PDF coinvolto (solo file web in repo).
- **Ultimo intervento (23/08/2026).** **App web: pulsante "Traiettoria 2D (come
  nel Colab)" sotto il plot principale + galleria di confronto.** In
  `visualizzazione.html` (file solo nella repo) ora, sotto "Superficie J(w) e
  percorso J(w_k)", un pulsante mostra UN solo grafico 2D per la configurazione
  corrente — algoritmo, loss, **w₀ = quello scelto dall'utente** (mai
  casuale), riuso batch on/off, iperparametri — con le **curve di livello di
  J(w)** (ellissi), il percorso, i **puntini ciano dove la CCV è violata e il
  batch aumenta** e i marker w₀/w_finale/w* (stile Colab). Un pulsante "Salva
  immagine per confronto" fissa il grafico in una galleria sotto con una
  **caption degli iperparametri** (algoritmo, loss, w₀, α, θ, batch₀, max_iter,
  N, seed, batch dinamico/fisso, riuso batch, line search, R/maxcg/ν/σ/η,
  riuso Hessiana), così si può cambiare configurazione e generare altri plot da
  confrontare (quante volte si vuole); ogni immagine si può rimuovere
  singolarmente o tutte insieme. Implementazione: gli algoritmi generati
  (`generateGD/generateNewtonCG/generateNewtonL1/generateBB`) ora **restituiscono
  anche `resize_points`** (terzo output) registrato dentro il ramo CCV, così i
  puntini compaiono a OGNI violazione della CCV (come nel Colab, anche quando
  il batch è saturo a N); livelli delle curve calcolati dinamicamente dal range
  di J della traiettoria (`_levels` in evalCode); rendering Plotly coerente col
  tema (2D con `scaleanchor` = aspect uguale, curva 1D per le loss 1D).
  Validazione: sintassi JS con JavaScriptCore + harness con DOM fittizio
  **22/22 controlli** (render 2D/1D, caption, pin/rimozione singola/tutte,
  senza-dati) e **10 codici Python generati eseguiti** con numpy sulla loss
  quad_ill (ricampionamento/riuso/fisso per i 4 algoritmi: resize_points
  coerenti, livelli crescenti, JSON ok). Nessun PDF coinvolto (solo file web in
  repo).
- **Ultimo intervento (23/08/2026).** **App web: la teoria dinamica ora segue
  TUTTI gli iperparametri (batch dinamico, sottocampionamento Hessiana,
  Hessian-free L1, riuso Hessiana legato/indipendente).** In
  `visualizzazione.html` (file solo nella repo) gli pseudocodici erano blocchi
  statici che reagivano solo a riuso (default/reuse) e line search
  (Wolfe/Armijo): "Batch dinamico" disattivato, "H_k ⊆ S_k" vs "H_k
  indipendente", "Hessian-free" vs "Hessiana esplicita" e "Riuso Hessiana"
  legato/indipendente (M_H) cambiavano il codice Python generato ma non la
  teoria. Ora lo pseudocodice è **generato dinamicamente** da
  `pseudoGD/pseudoNewtonCG/pseudoNewtonL1/pseudoBB(o)` e iniettato nei contenitori
  `#pseudoTarget-<algo>` da `renderPseudo()` (ri-typeset con MathJax) in base a
  tutte le opzioni: batch fisso senza CCV, H_k ⊆ S_k o campionamento indipendente,
  Hessiana esplicita/hessian-free in L1, ramo legato/indipendente con M_H (valore
  numerico se personalizzato). Aggiunti anche i wrapper `.dyn-ccv/.dyn-fixed`
  attorno alle formule CCV (GD, Newton-CG, BB), le note
  `.subset-note/.indep-note` sul campionamento dell'Hessiana e le prose
  `.hf-free/.hf-explicit`; le reuse-note ora commutano `.rn-ccv/.rn-fixed` e
  `.rn-tied/.rn-independent`. Bonus: BB ora mostra uno pseudocodice anche a riuso
  disattivato (prima solo formule). Validazione: sintassi JS (JavaScriptCore,
  parse OK); harness sulle funzioni reali — 479/479 controlli su 80 combinazioni
  (GD 8, Newton-CG 32, Newton-L1 32, BB 8) più renderPseudo con DOM fittizio; e
  **tutti gli 80 blocchi LaTeX generati compilano con pdflatex (0 errori)**.
  Nessun PDF coinvolto (solo file web in repo).

- **Ultimo intervento (23/08/2026).** **App web: i controlli algoritmo-specifici
  (line search, sottocampionamento Hessiana, Hessian-free L1) compaiono solo per
  i metodi che li usano.** In `visualizzazione.html` (file solo nella repo) i tre
  selettori del blocco "Codice Python" erano sempre visibili per qualunque
  algoritmo: "Line search — GD, Newton-CG, BB" compariva anche con Newton-L1
  (che usa il suo σ-Armijo) e con Custom, "Sottocampionamento Hessiana —
  Newton-CG, Newton-L1" anche con GD/BB/Custom (metodi senza Hessiana), e
  "Hessian-free in Newton L1 — Newton-L1" anche con GD/BB/Newton-CG. Ora ogni
  riga è racchiusa in un proprio contenitore (`lineSearchRow`, `hkSubsetRow`,
  `hessianFreeL1Row`) e `updateParamsVisibility()` la mostra solo per i metodi
  pertinenti: line search per GD/Newton-CG/BB, sottocampionamento Hessiana per i
  due Newton, Hessian-free solo per Newton-L1. I pannelli numerici
  (θ/batch₀, R/maxCG, ν/σ/η) erano già condizionati all'algoritmo selezionato.
  Validazione: sintassi JS con JavaScriptCore (parse OK) e harness con DOM
  fittizio sulle funzioni reali: 70/70 controlli di visibilità superati
  (5 algoritmi × 2 stati di riuso). Nessun PDF coinvolto (solo file web in repo).

- **Ultimo intervento (23/08/2026).** **App web: la teoria dinamica ora segue
  la line search scelta (Wolfe/Armijo).** In `visualizzazione.html` gli
  pseudocodici e le formule di line search di GD, Newton-CG e BB erano fissi
  (GD/BB mostravano sempre Armijo, Newton-CG sempre Wolfe) e non reagivano al
  selettore "Line search". Ora ogni blocco esiste in due varianti (Wolfe e
  Armijo), mostrate/nascoste da `updatePseudoVariants()` in base a
  `lineSearchMode`, combinate con la dimensione riuso (default/reuse) in
  un'unica passata; i blocchi senza variante line search (es. Newton-L1)
  seguono solo il riuso. Rese dinamiche anche le formule "Iterazione" (GD),
  "Line search e aggiornamento" (Newton-CG), "Line search (Armijo)" (BB) e le
  sottosezioni Formule "9. Line search" (GD) e "6. Line search" (Newton-CG).
  Validazione: 65/65 blocchi math compilano con pdflatex (il 66° è la stringa
  JS della config MathJax `\[','\]`, non un blocco math), `\[`/`\]` bilanciati
  66/66, e simulazione Python della visibilità: per ogni combinazione
  reuse×wolfe esattamente 1 pseudocodice visibile per GD/Newton-CG/L1 (BB: 1
  solo a riuso attivo, com'era prima) con il metodo giusto. Nessun PDF
  coinvolto.

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
  definizione di $\widehat{\mathcal V}_k$ (usata ma mai definita). Refinement
  (stessa giornata): etichetta del parametro rinominata "Riuso Hessiana" (era
  "modo Hessiana") e variabili scritte con `\mathtt{da\_ricampionare}` (in
  `\texttt` MathJax mostrava `\_` letterale; in `\mathtt`, alfabeto matematico,
  il trattino basso è renderizzato correttamente). Validazione:
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
  condizionato ($\kappa\approx1.1$) la curvatura varia pochissimo da una
  direzione all'altra, perciò un passo scalare adeguato funziona in ogni
  direzione: Dynamic GD/Newton-CG-$L_1$/BB-CCV
  sono i più veloci e Newton-CG il più lento (paga la soluzione del sistema di
  Newton senza guadagnarne); (ii)~sul molto mal condizionato ($\kappa\approx
  100$) il numero di condizionamento entra nel tasso dei metodi del primo
  ordine (Sez. 4), quindi Dynamic GD/Newton-CG-$L_1$/Newton-CG restano sopra
  $10^{-1}$, mentre BB-CCV raggiunge la precisione macchina in 11 iterazioni
  perché il passo di Barzilai--Borwein stima la curvatura dai soli gradienti
  (App. E) e si adatta ai problemi in cui la curvatura è molto diversa da una
  direzione all'altra; (iii)~con il termine incrociato BB-CCV
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
