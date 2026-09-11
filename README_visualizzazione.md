# 🎛️ Visualizzazione Interattiva — Selezione Dinamica della Dimensione del Campione

**Guida utente dell'applicazione `visualizzazione.html`**

> **In una frase.** È un laboratorio che gira interamente dentro il browser: scegli una
> funzione obiettivo, un algoritmo a campione dinamico e le sue varianti, premi
> **Ricalcola** e vedi *in diretta* come la dimensione del mini-batch $n_k$ cresce seguendo
> la **Condizione di Controllo della Varianza (CCV)** mentre $w_k$ scende verso il minimo.

Sapienza Università di Roma · Scienze Matematiche per l'Intelligenza Artificiale ·
Alessandro Lo Curcio · A.A. 2025–2026
Riferimento: Byrd, Chin, Nocedal & Wu, *Sample size selection in optimization methods for
machine learning*, Mathematical Programming, 2012.

---

## Sommario

1. [Che cosa fa (e perché è utile)](#1-che-cosa-fa-e-perché-è-utile)
2. [Come aprirla](#2-come-aprirla)
3. [La mappa dell'applicazione](#3-la-mappa-dellapplicazione)
4. [Percorso guidato in 60 secondi](#4-percorso-guidato-in-60-secondi)
5. [Pannello «Funzione di Loss»](#5-pannello-funzione-di-loss)
6. [Pannello «Algoritmo»](#6-pannello-algoritmo)
7. [Pannello «Parametri» (e le otto leve sperimentali)](#7-pannello-parametri-e-le-otto-leve-sperimentali)
8. [Pannello «Codice Python dell'algoritmo»](#8-pannello-codice-python-dellalgoritmo)
9. [Esecuzione: Ricalcola, Default, animazione](#9-esecuzione-ricalcola-default-animazione)
10. [Le metriche](#10-le-metriche)
11. [I grafici, uno per uno](#11-i-grafici-uno-per-uno)
12. [🧪 Analisi (tabella di convergenza)](#12--analisi-tabella-di-convergenza)
13. [🧪 Test batch (le tabelle della tesi)](#13--test-batch-le-tabelle-della-tesi)
14. [📘 Teoria dell'algoritmo selezionato](#14--teoria-dellalgoritmo-selezionato)
15. [Algoritmo personalizzato](#15-algoritmo-personalizzato)
16. [Esperimenti guidati](#16-esperimenti-guidati)
17. [Come leggere i risultati](#17-come-leggere-i-risultati)
18. [Dove sta cosa nella tesi](#18-dove-sta-cosa-nella-tesi)
19. [Note, limiti e piccoli trucchi](#19-note-limiti-e-piccoli-trucchi)

---

## 1. Che cosa fa (e perché è utile)

I metodi di ottimizzazione per il machine learning calcolano il gradiente su un
**mini-batch** invece che sull'intero dataset: ogni iterazione costa poco, ma la stima è
rumorosa. La domanda centrale della tesi è: **quanto grande deve essere il campione a ogni
iterazione?** La risposta dei metodi *dinamici* è: *tanto quanto basta* perché il gradiente
campionario soddisfi la condizione di accuratezza

$$\|g_k - \nabla J(w_k)\| \le \theta\,\|g_k\|, \qquad 0<\theta<1,$$

cioè la **CCV**. Quando la CCV è violata, il batch viene **aumentato**; quando è
soddisfatta, si procede con il campione già estratto.

L'app rende tutto questo **visibile e manipolabile**. In particolare puoi:

- **vedere** il percorso $w_k$ sulla superficie di $J(w)$ e su un piano con le curve di livello;
- **vedere la crescita del batch** $n_k$ e confrontarla con l'andamento geometrico $a^k$,
  con l'esponente $a$ stimato ai minimi quadrati dai dati della tua esecuzione;
- **confrontare** i quattro algoritmi della tesi — Dynamic GD, Newton-CG, Newton-CG $L_1$,
  BB-CCV — a parità di ogni altro parametro;
- **accendere e spegnere** ogni ingrediente: batch dinamico, riuso del mini-batch, stop
  adattivo su validation set, riuso dell'Hessiana, line search, sottocampionamento
  dell'Hessiana, Hessian-free per $L_1$;
- **scrivere la tua loss** (o il tuo algoritmo) in Python e vederlo eseguire *davvero*, nel
  browser, senza installare nulla;
- **esportare** tabelle di esperimenti in stile Sezione 6 della tesi (LaTeX, JSON, copia);
- **leggere la teoria** dell'algoritmo selezionato accanto ai grafici, con pseudocodice e
  formule numerate.

Tutto ciò che vedi è **deterministico**: a parità di parametri e di `seed`, i risultati si
riproducono identici (il dataset sintetico è generato con quel seme).

## 2. Come aprirla

1. Fai **doppio clic** su `visualizzazione.html` (o trascinalo in una finestra del browser).
2. Usa un browser moderno — Chrome, Edge, Firefox o Safari.
3. Aspetta qualche secondo al primo avvio: in basso nel pannello «Codice Python
   dell'algoritmo» il badge **`Python`** diventa `pronto` quando l'interprete è caricato.

**Non serve installare niente**: niente Python, niente pip, niente server. Il browser scarica
una volta sola le librerie da internet e poi lavora in locale.

| Cosa viene scaricato | A cosa serve | Nota |
|---|---|---|
| **Pyodide** (Python in WebAssembly) | esegue il codice della loss e degli algoritmi | il pezzo più pesante: ~10 MB, serve qualche secondo |
| **Plotly.js** | tutti i grafici interattivi (zoom, hover, legende cliccabili) | |
| **MathJax** | rende le formule della teoria e i tooltip matematici | |
| **Google Fonts** (Inter, JetBrains Mono, Playfair Display) | tipografia dell'interfaccia | se manca la rete, l'app funziona lo stesso con i font di sistema |

> 💡 **Suggerimento.** Dopo il primo caricamento il browser tiene tutto in cache: le aperture
> successive sono quasi istantanee. Se resti offline, l'app continua a funzionare finché la
> cache non viene svuotata.

Tutti i calcoli avvengono **nel tuo browser**: i parametri che inserisci non vengono inviati
da nessuna parte.

## 3. La mappa dell'applicazione

L'interfaccia è una **griglia a due colonne** (massimo 1400 px): a sinistra i **controlli**,
a destra i **risultati**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TESTATA  Sapienza · SMIA · Alessandro Lo Curcio · A.A. 2025–2026        │
│  Selezione Dinamica della Dimensione del Campione in Metodi di           │
│  Ottimizzazione per il Machine Learning                                  │
│  badge: [Visualizzazione Interattiva] [Python]   legenda ● percorso ★ w* │
├───────────────────────────────┬──────────────────────────────────────────┤
│  CONTROLLI (colonna sinistra) │  RISULTATI (colonna destra)              │
│                               │                                          │
│  1. Funzione di Loss          │  • Schede delle metriche                 │
│     preset · modalità · code  │    w · J(w) · ‖∇J‖ · ‖w−w*‖ · n_k        │
│                               │  • Superficie J(w) e percorso            │
│  2. Algoritmo                 │  • Traiettoria 2D (curve di livello)     │
│     GD · Newton-CG · L1 · BB  │  • Confronto immagini salvate            │
│                               │  • 📘 Teoria dell'algoritmo selezionato  │
│  3. Parametri                 │  • Analisi stocastica: n_k vs a^k        │
│     w0 · max_iter · α · N ·   │  • J_val(w_k) — stop adattivo            │
│     seed · θ · batch0 · …     │  • J_batch(w_k) — riuso per discesa      │
│     + 8 leve sperimentali     │                                          │
│                               │                                          │
│  4. Codice Python dell'algo   │                                          │
│     (generato, eseguibile)    │                                          │
│                               │                                          │
│  5. Animazione                │                                          │
│     ▶ ⏸ ⟲ · slider · loop     │                                          │
│                               │                                          │
│  6. Metriche                  │                                          │
│     ⟳ Ricalcola ⟲ Default     │                                          │
│     🧪 Analisi 🧪 Test batch  │                                          │
│     🖼️ Traiettoria 2D         │                                          │
└───────────────────────────────┴──────────────────────────────────────────┘
```

I pannelli che compaiono **solo quando servono**: la card *Traiettoria 2D* si apre col
pulsante dedicato, la card *Confronto immagini salvate* quando salvi almeno un'immagine, le
card *J_val(w_k)* e *J_batch(w_k)* quando accendi rispettivamente lo **stop adattivo con
validation set** e il **riuso per discesa della loss**.

## 4. Percorso guidato in 60 secondi

1. Apri `visualizzazione.html` e aspetta il badge **`Python: pronto`**.
2. Lascia tutto ai valori di default (**Quadratica ben condizionata**, **Dynamic GD**).
3. Premi **⟳ Ricalcola** (pannello *Metriche*): la superficie si popola e il badge diventa
   `pronto`.
4. Premi **▶** nel pannello *Animazione*: guarda $w_k$ camminare sulla superficie.
5. Apri **🖼️ Traiettoria 2D**: vedi le curve di livello, i **punti in cui la CCV ha
   aumentato il batch** e il ricampionamento. Trascina per spostare la vista, usa lo slider
   per lo zoom, poi **📌 Salva immagine per confronto**.
6. Guarda **Analisi stocastica: $n_k$ vs $a^k$** e l'esponente $a$ stimato: è il
   *fit* geometrico della crescita del batch.
7. Cambia algoritmo in **Newton-CG**: il codice Python, la teoria, lo pseudocodice e i
   parametri **si aggiornano da soli**. Premi di nuovo **Ricalcola** e confronta.
8. Attiva **Iterazioni consecutive sullo stesso mini-batch** e premi **Ricalcola**: si
   aprono le card *J_val(w_k)*/*J_batch(w_k)* e il batch smette di crescere a ogni
   iterazione.
9. Apri **🧪 Test batch** e premi **▶ Avvia** per generare le tabelle in stile Sezione 6
   della tesi.

## 5. Pannello «Funzione di Loss»

Qui decidi **cosa** stai minimizzando. Puoi partire da un preset e poi modificarlo, oppure
scrivere la tua loss da zero.

### 5.1 La lista dei preset

| Preset | Funzione | Dim. | Note |
|---|---|---|---|
| Quadratica ben condizionata ($\kappa\approx1.1$) | $J=(w_1-1)^2+(w_2+2)^2+0.1\,w_1w_2$ | 2 | default dell'app |
| Quadratica mal condizionata ($\kappa\approx20$) | $J=20(w_1-1)^2+(w_2+2)^2$ | 2 | il batch cresce molto di più |
| Quadratica molto mal condizionata ($\kappa\approx100$) | $J=100(w_1-1)^2+(w_2+2)^2$ | 2 | caso severo, ottimo per vedere la CCV scattare |
| Quadratica con termine incrociato | $J=(w_1-1)^2+(w_2+2)^2+0.5(w_1-1)(w_2+2)$ | 2 | Hessiana non diagonale |
| Funzione di Rosenbrock ($c=100$) | $J=\frac1N\sum_i\big[(w_1-a_i)^2+100\big((w_2-b_i)-(w_1-a_i)^2\big)^2\big]$ | 2 | **non** quadratica |
| 1D Quadratica | $J=\frac1N\sum_i(w-a_i)^2$ | 1 | valle parabolica |
| 1D Quartica + quadratica | forma quartica con minimo interno | 1 | |
| 1D Sinusoidale + quadratica | oscillazioni + termine quadratico | 1 | |
| 1D Esponenziale + quadratica | $J=\frac1N\sum_i\big[e^{0.5(w-a_i)^2}-1\big]$ | 1 | |
| ✏️ **Custom** | quello che scrivi tu | 1 o 2 | vedi §5.3 |

I preset **1D** mostrano una sezione nel piano $(w, J)$ invece della superficie: sono il modo
più veloce per capire *dove* il batch decide di crescere lungo una valle.
Le **formule esplicite** della loss di ogni preset e del rumore aggiunto a ogni esempio sono in
**§5.5**.

### 5.2 Modalità gradiente / Hessiana

| Modalità | Cosa fa | Quando usarla |
|---|---|---|
| **Predefinite (formule chiuse)** | usa le derivate esatte scritte nel preset, per **singolo esempio** (`loss_i`, `grad_i`, `hess_i`, `hessvec_i`) | quando vuoi i risultati «puliti», senza errore di derivazione numerica |
| **Autodiff numerica (automatica)** | fornisce solo `J(w)` e le derivate vengono calcolate automaticamente | quando scrivi una loss nuova e non vuoi derivarla a mano |

> ℹ️ Con **Custom** la modalità è sempre autodiff.

### 5.3 L'editor del codice

Sotto i menu c'è l'editor con il codice Python della loss. Il messaggio dell'app è
esplicito: *«Modifica la loss e premi Ricalcola. Il plot usa esattamente questo codice.»*
Non è una simulazione: è **il codice che verrà eseguito**.

Il codice di un preset contiene tutto ciò che serve:

```python
import numpy as np

def J(w):                 # obiettivo (media sui dati)
    x, y = w[0], w[1]
    return (x - 1)**2 + (y + 2)**2 + 0.1*x*y

def gradJ(w): ...         # gradiente esatto
def hessJ(w): ...         # Hessiana esatta

DIM = 2                   # dimensione del problema
W_STAR = np.array([1.0, -2.0])      # minimo, per l'errore ‖w−w*‖
XMIN, XMAX = -1.5, 3.5              # riquadro dei grafici
YMIN, YMAX = -4.0, 0.0
```

Il pulsante **↻ Ripristina** rimette il preset selezionato al suo codice originale.

### 5.4 Il dataset sintetico «centrato»

Nei preset il dataset non è una tabella di dati, ma **$N$ esempi generati** con
`np.random.seed(seed)` e poi **centrati**: da ogni coefficiente viene sottratta la propria media
campionaria, così che le medie campionarie siano **esattamente** i valori nominali
($\bar a = 1$, $\bar b = -2$, $\bar c = 0.1$ oppure $0.5$). È il trucco che rende confrontabile
l'errore $\|w_k-w_\*\|$ con il minimo vero: per i preset quadratici la $J$ *campionaria*
coincide con quella *nominale* a meno di una costante (e con gradiente identico), mentre per i
preset non quadratici la coincidenza vale solo nel limite $N\to\infty$ — in quel caso l'app
calcola $w_\*$ numericamente. Tutte le formule sono in **§5.5**.

### 5.5 Le formule esplicite: loss e rumore, preset per preset

Qui c'è tutto quello che serve per **riprodurre a mano** (o citare in tesi) ciò che l'app fa.

#### 5.5.1 Il rumore: lo schema comune

Il rumore è **solo** quello dei dati, e viene generato una volta sola all'avvio di ogni
esecuzione. Con $z_{a,i}, z_{b,i}, z_{c,i} \sim \mathcal{N}(0,1)$ indipendenti
(`np.random.randn`), $i = 1,\dots,N$:

$$
\tilde a_i = \mu_a + \sigma_a\,z_{a,i},\qquad
\tilde b_i = \mu_b + \sigma_b\,z_{b,i},\qquad
\tilde c_i = \mu_c + \sigma_c\,z_{c,i}
$$

e poi la **centratura** che fissa esattamente le medie:

$$
a_i = \tilde a_i - \frac1N\sum_{j=1}^N \tilde a_j + \mu_a
\qquad\Longrightarrow\qquad \frac1N\sum_{i=1}^N a_i = \mu_a
$$

(analogamente per $b_i$ e $c_i$). I valori nominali sono $\mu_a = 1$ e $\mu_b = -2$ per tutti i
preset, con $\mu_c = 0.1$ per *quad_well* e $\mu_c = 0.5$ per *quad_offdiag*. Le deviazioni
standard **sono il rumore aggiunto a ogni esempio**:

| Preset | $\sigma_a$ | $\sigma_b$ | $\sigma_c$ | dimensione |
|---|---|---|---|---|
| Quadratica ben condizionata | **0.2** | **0.2** | **0.05** | 2 |
| Quadratica mal condizionata | **0.2** | **0.2** | — | 2 |
| Quadratica molto mal condizionata | **0.2** | **0.2** | — | 2 |
| Quadratica con termine incrociato | **0.2** | **0.2** | **0.05** | 2 |
| Rosenbrock | **0.2** | **0.2** | — | 2 |
| 1D Quadratica / Quartica / Sinusoidale / Esponenziale | **0.2** | — | — | 1 |
| ✏️ Custom | nessun rumore | — | — | 1 o 2 |

Il `seed` del pannello *Parametri* è esattamente il seme di `np.random.seed(...)`: cambiarlo
cambia **la realizzazione del rumore** (cioè i numeri $a_i, b_i, c_i$), non gli iperparametri.

#### 5.5.2 La loss per esempio, preset per preset

Ogni preset definisce la sua **loss di un singolo esempio** $\ell_i(w)$; la funzione obiettivo
che l'app calcola e disegna è sempre la media

$$J(w) \;=\; \hat J_N(w) \;=\; \frac1N\sum_{i=1}^N \ell_i(w).$$

| Preset | loss per esempio $\ell_i(w)$ | $J$ nominale (con $a_i\to\mu_a$, $b_i\to\mu_b$, $c_i\to\mu_c$) |
|---|---|---|
| ben condizionata | $(w_1-a_i)^2+(w_2-b_i)^2+c_i\,w_1w_2$ | $(w_1-1)^2+(w_2+2)^2+0.1\,w_1w_2$ |
| mal condizionata | $20\,(w_1-a_i)^2+(w_2-b_i)^2$ | $20\,(w_1-1)^2+(w_2+2)^2$ |
| molto mal condizionata | $100\,(w_1-a_i)^2+(w_2-b_i)^2$ | $100\,(w_1-1)^2+(w_2+2)^2$ |
| termine incrociato | $(w_1-a_i)^2+(w_2-b_i)^2+c_i(w_1-a_i)(w_2-b_i)$ | $(w_1-1)^2+(w_2+2)^2+0.5(w_1-1)(w_2+2)$ |
| Rosenbrock | $x_i^2+C\,z_i^2$ con $x_i=w_1-a_i$, $z_i=(w_2-b_i)-x_i^2$, $C=100$ | $(w_1-1)^2+100\big[(w_2+2)-(w_1-1)^2\big]^2$ |
| 1D Quadratica | $(w-a_i)^2$ | $(w-1)^2$ |
| 1D Quartica | $(w-a_i)^4+0.1\,(w-a_i)^2$ | $(w-1)^4+0.1\,(w-1)^2$ |
| 1D Sinusoidale | $1-\cos(w-a_i)+0.1\,(w-a_i)^2$ | $1-\cos(w-1)+0.1\,(w-1)^2$ |
| 1D Esponenziale | $e^{\frac12(w-a_i)^2}-1$ | $e^{\frac12(w-1)^2}-1$ |
| Custom | quella che scrivi tu | la tua $J(w)$ |

> 🧩 Nei preset 1D la variabile è una sola, ma il codice lavora comunque in $\mathbb{R}^2$: la
> seconda componente è fittizia (le funzioni `grad_i`/`hess_i` restituiscono `[·, 0.0]`).

#### 5.5.3 Che cosa comporta la centratura

La centratura fissa le medie ma **non** le varianze campionarie: $\hat\sigma_a^2 \approx 0.2^2$,
$\hat\sigma_b^2 \approx 0.2^2$, $\hat\sigma_c^2 \approx 0.05^2$. Ne segue, per i preset
quadratici in cui il rumore entra solo nei termini del tipo $(w_j-a_i)^2$,

$$
\hat J_N(w) \;=\; J_{\text{nom}}(w) + \sum_j c_j\,\hat\sigma_j^2
\qquad\text{(scarto costante, indipendente da } w\text{)}
$$

| Preset | $\hat J_N(w)-J_{\text{nom}}(w)$ con seed 42 e $N=200$ |
|---|---|
| ben condizionata | $\hat\sigma_a^2+\hat\sigma_b^2 = 0.073270$ |
| mal condizionata | $20\,\hat\sigma_a^2+\hat\sigma_b^2 = 0.728720$ |
| molto mal condizionata | $100\,\hat\sigma_a^2+\hat\sigma_b^2 = 3.488510$ |
| 1D Quadratica | $\hat\sigma_a^2 = 0.034497$ |

Poiché la costante non dipende da $w$, **gradiente, Hessiana e minimizzatore coincidono
esattamente** con quelli nominali (verificato numericamente: scarto $\sim 10^{-16}$).

Due eccezioni da conoscere:

- **termine incrociato**: qui $c_i$ moltiplica $(w_1-a_i)(w_2-b_i)$, quindi la coincidenza è
  affine, non solo costante:
  $\hat J_N(w) = J_{\text{nom}}(w) + c_0 + \delta^{\mathsf T}w$ con, al seed 42,
  $\delta \simeq (3.2\cdot10^{-4},\;1.24\cdot10^{-3})$. Il gradiente è quindi sfasato di una
  costante e il minimo è spostato di una quantità $O(1/\sqrt N)$;
- **preset non quadratici** (Rosenbrock e i tre 1D non quadratici): l'identità non vale, perché
  la media di una funzione non lineare dei coefficienti non è la funzione delle medie. Lo scarto
  è $O(1/\sqrt N)$ e dipende da $w$ — è esattamente il motivo per cui in quei preset il codice
  calcola $w_\*$ **numericamente** (`_wstar()`: Newton con differenze finite su $\hat J_N$).

#### 5.5.4 Il rumore che l'algoritmo vede davvero (mini-batch e CCV)

Per completezza, le formule esatte dell'implementazione (sono le stesse dello pseudocodice
generato nell'editor):

$$
\mathcal S_k = \{i_1,\dots,i_{n_k}\} \ \text{estratto} \ \textbf{senza reinserimento} \
\text{da } \{1,\dots,N\}
\qquad(\texttt{np.random.choice(N, size=n, replace=False)})
$$

$$
g_k = \frac{1}{n_k}\sum_{i\in\mathcal S_k}\nabla\ell_i(w_k),
\qquad
\hat s_j^2 = \frac{1}{n_k-1}\sum_{i\in\mathcal S_k}\big(g_{ij}-\bar g_j\big)^2,
\qquad
\hat V_k = \sum_{j=1}^{d}\hat s_j^2
$$

dove $\hat s_j^2$ è la varianza campionaria **non distorta** (`ddof=1`) della $j$-esima
coordinata dei gradienti del mini-batch e $\hat V_k$ è quindi la **traccia della covarianza
campionaria**. Il test di CCV implementato è la versione *plug-in* della condizione
$\operatorname{tr}(\Sigma)/n \le \theta^2\|g\|^2$:

$$
\text{se}\quad \frac{\hat V_k}{n_k} \;>\; \theta^2\,\|g_k\|^2
\qquad\Longrightarrow\qquad
n_{k+1} = \min\!\left(N,\ \left\lceil \frac{\hat V_k}{\theta^2\,\|g_k\|^2}\right\rceil + 1\right)
$$

e l'iterazione corrente si completa comunque con il campione già estratto (il nuovo $n_{k+1}$
vale per l'estrazione successiva). Per i metodi di Newton il secondo campione — l'Hessiana — ha
dimensione $n_h = R\,|\mathcal S_k|$ ed è estratto anch'esso senza reinserimento, da
$\mathcal S_k$ (default) oppure da tutto il dataset.

> 📐 Queste due formule sono il motivo per cui il grafico $n_k$ vs $a^k$ ha la forma che ha: la
> regola $n_{k+1}\propto \hat V_k/\|g_k\|^2$ fa crescere il batch **solo quando serve**, e il
> fattore $\theta^2$ al denominatore regola quanto.

#### 5.5.5 E con «✏️ Custom»?

Il preset *Custom* **non genera alcun dataset**: niente rumore, niente media su esempi. Lo
scheletro che l'app mette nell'editor è un metodo deterministico:

```python
def my_algorithm(w0, alpha, max_iter):
    w = np.array(w0, dtype=float)
    history = [w.copy().tolist()]
    batch_sizes = [1]
    for k in range(max_iter):
        g = gradJ(w)                     # gradiente esatto: nessun campionamento
        if np.linalg.norm(g) < 1e-10:
            break
        w = w - alpha * g
        history.append(w.copy().tolist())
        batch_sizes.append(1)
    return history, batch_sizes
```

Con Custom, quindi, la CCV **non entra in gioco** (la varianza campionaria è nulla) e il grafico
$n_k$ resta piatto a 1. Se vuoi studiarci una variante dinamica, definisci tu `loss_i`/`grad_i`
e campiona — oppure parti da uno dei quattro metodi generati (che hanno già la CCV) e
modificalo.



## 6. Pannello «Algoritmo»

| Metodo | In due parole | Usa |
|---|---|---|
| **Gradiente a Campione Dinamico (GD)** | discesa più ripida con passo scelto da line search (Wolfe di default) | $\theta$, `batch0`, line search |
| **Newton-CG con Campionamento Dinamico** | direzione di Newton risolta dal CG su un'Hessiana *sottocampionata* ($H_k\subseteq S_k$, $|H_k|=R\,|S_k|$) e prodotta Hessiana-vettore | $R$, `max CG iter`, sottocampionamento, riuso Hessiana |
| **Newton-CG con Regolarizzazione $L_1$** | come sopra ma su $F(w)=J(w)+\nu\|w\|_1$, con faccia ortante, active set e ricerca lineare **proiettata** | $\nu$, $\sigma$, $\eta$, Hessian-free |
| **Barzilai–Borwein con Campionamento Dinamico (BB-CCV)** | passo BB *clippato* + Armijo, senza Hessiana | come GD |
| ✏️ **Algoritmo personalizzato** | il tuo algoritmo in Python | vedi §15 |

Ogni volta che cambi metodo **cambiano automaticamente**: le righe di parametri visibili, il
codice Python generato, lo pseudocodice e i blocchi di teoria/formule a destra. Il testo sotto
il selettore (**Algoritmo**) è un promemoria del comportamento atteso (per esempio: *«GD con
campione dinamico e controllo della varianza»*).

## 7. Pannello «Parametri» (e le otto leve sperimentali)

### 7.1 I parametri di base

| Controllo | Default | Intervallo | Significato |
|---|---|---|---|
| **w₀ (x, y)** | `2.0, -3.0` | — | punto di partenza ($w_0$ può anche essere immesso come coppia, mai casuale) |
| **max_iter** | 30 | 5–200 (passo 5) | numero massimo di iterazioni |
| **α** | 0.1 | 0.001–2 | passo base della line search |
| **N (dataset)** | 200 | 10–2000 | quanti esempi contiene il dataset sintetico |
| **seed** | 42 | 0–9999 | seme del generatore: cambia la realizzazione del rumore (§5.5.1), non gli iperparametri |
| **θ (toll. CCV)** | 0.5 | 0.01–0.99 | soglia della condizione di controllo della varianza: **piccolo ⇒ batch grandi** |
| **batch0** | 5 | 1–50 | dimensione del primo mini-batch |
| **R ($\|H\|/\|S\|$)** | 0.2 | 0.05–0.9 | Newton: quanta parte del batch serve per l'Hessiana ($|H_k|=R\,|S_k|$) |
| **max CG iter** | 10 | 1–50 | Newton: iterazioni massime del gradiente coniugato |
| **ν (penalità $L_1$)** | 0.1 | 0.001–1 | $L_1$: peso della regolarizzazione |
| **σ (Armijo)** | 0.1 | 0.001–0.5 | $L_1$: parametro del backtracking |
| **η (toll. CG)** | 0.5 | 0.01–0.99 | $L_1$: tolleranza interna del CG |
| ☑ **Mostra anche F(w) = J(w) + ν‖w‖₁** | attivo | — | disegna in più l'obiettivo regolarizzato (utile con $L_1$) |

I parametri **che non riguardano l'algoritmo scelto vengono nascosti**: `θ`/`batch0` per GD e
BB, `R`/`max CG` per Newton-CG, `ν`/`σ`/`η` per $L_1$.

### 7.2 Le otto leve sperimentali

Sono il cuore «da ricerca» dell'app: ogni leva aggiunge una variante concreta del metodo e
cambia *anche* il codice Python generato e lo pseudocodice mostrato a destra.

**① Batch dinamico (CCV) — tutti gli algoritmi**
`Attivo (default)` · `Disattivato (batch fisso)`
*Default: la dimensione del batch cresce automaticamente quando la varianza supera la soglia.*
Spegnendolo ottieni il metodo a **batch fisso**: è il termine di paragone per vedere quanti
esempi risparmia (o spreca) la regola dinamica.

**② Iterazioni consecutive sullo stesso mini-batch — tutti gli algoritmi**
`Disattivato (default)` · `Attivo`
*Default: a ogni iterazione si ricampiona il mini-batch. Attivo: lo stesso mini-batch viene
riusato per più iterazioni consecutive.*
Con **Max iterazioni consecutive per mini-batch**: `Illimitato` (si riusa finché la CCV resta
soddisfatta, con limite naturale `max_iter`) oppure `Personalizzato` (valore $k$). Con $k=1$
si ricampiona a ogni iterazione; valgono i criteri di stop adattivo descritti sotto.
Quando il riuso è attivo compaiono le card **$J_{val}(w_k)$** e **$J_{batch}(w_k)$**.

**③ Stop adattivo con validation set — tutti gli algoritmi**
`☐ Usa validation set per stop adattivo`
*Quando attivo, M (e $M_H$ per i metodi di Newton) non si impostano più a mano: il mini-batch
si ricampiona quando la loss sul validation set non migliora.*

| Sotto-controllo | Default | Significato |
|---|---|---|
| % validation | 20 | quota del dataset riservata alla validazione |
| pazienza | 3 | quante valutazioni consecutive senza progresso prima di ricampionare |
| frequenza | 1 | ogni quante iterazioni si valuta |
| strategia | `Fixed (val set fisso)` / `Dynamic (ricampionato a ogni cambio batch)` | se il validation set cambia quando cambia il batch |
| tolleranza | `1e-4` | soglia relativa di progresso |
| min_abs | `0.0` | soglia assoluta di progresso |

Criterio esatto mostrato dall'app:
$$J_{val}(w_{k+1}) \le J_{val}^{best}\cdot(1-\text{tol}) - \text{min\_abs}.$$

**④ Riuso per discesa della loss sul batch — tutti gli algoritmi**
`☐ Usa la discesa della loss sul batch per lo stop del riuso`
Criterio (la loss è calcolata **sullo stesso mini-batch in uso**, non su un insieme esterno):
$$J_{batch}(w_{k+1}) \le J_{batch}(w_k) - \text{tol}\cdot|J_{batch}(w_k)| - \text{min\_abs}.$$
Sotto-controlli: tolleranza (`1e-4`), min_abs (`0.0`), pazienza (`1`), frequenza (`1`).

**⑤ Riuso dell'Hessiana (Newton)**
`Legato a S_k (default teoria)` · `Indipendente da S_k`
*Legato: $H_k$ viene riusato insieme a $S_k$ finché la CCV è soddisfatta. Indipendente: $H_k$
viene riusato per $M_H$ iterazioni.*
Con **Max riusi consecutivi Hessiana** `Illimitato`/`Personalizzato` (default 10).

**⑥ Line search — GD, Newton-CG, BB**
`Condizioni di Wolfe` · `Solo Armijo`
*Default per algoritmo, come da pseudocodice della tesi: Wolfe per Dynamic GD e Newton-CG,
Armijo per BB-CCV. (Newton-L1 usa sempre la sua ricerca proiettata.)*

**⑦ Sottocampionamento dell'Hessiana — Newton-CG, Newton-$L_1$**
`H_k ⊆ S_k (default teoria)` · `H_k indipendente da S_k`
*Default: $H_k$ è estratto come sottoinsieme di $S_k$, coerente con il paper.*

**⑧ Hessian-free in Newton $L_1$**
`Attivo (default teoria)` · `Hessiana esplicita`
*Default: CG Hessian-free anche per $L_1$, senza costruire $H$ esplicita.*

Il pulsante **↻ Ripristina** in fondo al pannello rimette tutti i parametri e le leve ai
valori di default.

## 8. Pannello «Codice Python dell'algoritmo»

L'editor con intestazione **«Algoritmo (modificabile)»** contiene l'implementazione Python
dell'algoritmo selezionato **così come verrà eseguita** da Pyodide. Non è una vetrina: è il
codice che produce i numeri dei grafici.

La cosa notevole è che **il codice si riscrive da solo**: cambiando algoritmo o anche solo una
delle otto leve, l'app rigenera la versione corrispondente. Le varianti generate coprono
tutte le combinazioni previste:

- versione **base** (ricampionamento a ogni iterazione),
- versione con **riuso** del mini-batch,
- versione con **stop adattivo su validation set**,

per ciascuno dei metodi: GD, Newton-CG, Newton-$L_1$, BB-CCV. Il pulsante **↻ Ripristina**
rigenera il testo dai parametri correnti se lo hai modificato; accanto c'è il badge di stato
**`Python`**, che segnala l'inizializzazione dell'interprete.

Questo pannello è anche il modo più rapido per **leggere la differenza tra le varianti**: due
esecuzioni con leve diverse si spiegano leggendo il codice che l'app ti mette davanti.

## 9. Esecuzione: Ricalcola, Default, animazione

### 9.1 I pulsanti di esecuzione (pannello «Metriche»)

| Pulsante | Cosa fa |
|---|---|
| **⟳ Ricalcola** | esegue l'algoritmo con i parametri correnti e ridisegna tutti i grafici |
| **⟲ Default** | riporta *tutti* i parametri ai valori di default |
| **🧪 Analisi** | apre la tabella di convergenza (§12) |
| **🖼️ Traiettoria 2D** | apre il grafico con le curve di livello (§11.2) |
| **🧪 Test batch** | apre gli esperimenti a griglia con le tabelle della tesi (§13) |

Il badge di stato (`pronto` / `▶ play`) e la riga di diagnostica sotto il grafico principale
ti dicono se il calcolo è concluso e riassumono l'esecuzione.

> ⏱️ **Quanto ci vuole?** Con i default (N=200, 30 iterazioni) è questione di istanti. Con
> N=2000, 200 iterazioni e Newton-CG il calcolo può richiedere qualche secondo: è Python in
> WebAssembly, quindi va più lento di Python nativo. Se il problema è 1D o l'animazione è
> fluida, tutto è istantaneo.

### 9.2 Il pannello «Animazione»

| Controllo | Funzione |
|---|---|
| **▶** | avvia la riproduzione: $w_k$ si muove di iterazione in iterazione |
| **⏸** | mette in pausa |
| **⟲** | riporta l'animazione all'iterazione 0 |
| **numero** (`iter`) | indice dell'iterazione corrente, scrivibile a mano |
| **slider** | scorre liberamente avanti/indietro tra le iterazioni (perfetto per «fermarsi sul punto in cui il batch è cresciuto») |
| ☑ **Loop automatico** | alla fine riparte da capo |
| ☑ **Mostra percorso** | disegna la scia $w_0\to w_k$ sulla superficie |

## 10. Le metriche

Nel pannello *Metriche* compaiono le schede numeriche. Quelle «condizionali» appaiono solo
quando hanno senso:

| Scheda | Simbolo | Significato |
|---|---|---|
| **w** | $w_k$ | punto corrente |
| **J(w)** | $J(w_k)$ | valore dell'obiettivo (verde/blu/ambra a seconda della scheda) |
| **‖∇J‖** | $\|\nabla J(w_k)\|$ | norma del gradiente esatto: quanto sei vicino alla stazionarietà |
| **‖w − w\*‖** | $\|w_k-w_\*\|$ | distanza dal minimo vero (il miglior indicatore di convergenza) |
| **‖∂F‖ (subgrad. $L_1$)** | $\|\partial F(w_k)\|$ | norma del subgradiente, **solo** con Newton-$L_1$ |
| **Batch size $n_k$** | $n_k$ | dimensione dell'ultimo mini-batch estratto |
| **M_actual** | $M$ effettivo | quante iterazioni consecutive hai davvero fatto sullo stesso mini-batch |
| **J_val (ultima valutazione)** | $J_{val}$ | ultima loss di validazione (riuso con validation set) |
| **J_batch (ultima valutazione)** | $J_{batch}$ | ultima loss sul batch in uso (riuso per discesa) |

Le stesse grandezze compaiono nei **tooltip del grafico**: passando il mouse sui punti vedi
`iter k · w = … · J(w) = …`, così non devi indovinare quale punto stai guardando.

## 11. I grafici, uno per uno

Tutti i grafici sono Plotly: puoi zoomare (trascinando o con la rotella), passare il mouse per
i valori esatti, **cliccare le voci di legenda** per accendere/spegnere curve, ed esportare
l'immagine (icona della fotocamera nella barra in alto a destra del grafico).

### 11.1 «Superficie J(w) e percorso J(w_k)»

Il grafico principale. Mostra:

- la **superficie** (o la sezione, per i problemi 1D) della funzione obiettivo;
- il **percorso** $w_0 \to w_1 \to \dots \to w_k$ con i singoli punti $w_k$;
- il punto di partenza $w_0$ e il minimo $w_\*$ (★);
- se hai spuntato **Mostra anche F(w)**, la superficie/curva dell'obiettivo regolarizzato
  $F(w)=J(w)+\nu\|w\|_1$ con il percorso su $F$ (utile con Newton-$L_1$, dove i due obiettivi
  hanno minimi diversi).

Nella **legenda della testata** hai il codice colori: il punto ★ segna i pesi ottimali.

### 11.2 «Traiettoria 2D — J(w) e percorso J(w_k)»

Si apre con il pulsante **🖼️ Traiettoria 2D**. È la vista più *narrativa* dell'app: le
**curve di livello** di $J(w)$ nel piano $(w_1,w_2)$ con sopra la traiettoria, i punti
$\{w_k\}$ e — soprattutto — due marcatori distinti:

| Marcatore | Significato |
|---|---|
| **CCV violata (batch aumentato)** | qui il batch è stato **ingrandito** perché la condizione di varianza non era soddisfatta |
| **Ricampionamento (non-CCV)** | qui il campione è stato **ricambiato** per altre ragioni (riuso, fine pazienza, cambio di Hessiana…) |

Con Newton-$L_1$ compaiono anche le curve di livello di $F$ e il minimo di $F$.

Come si usa:

- **trascina** sull'immagine per spostare la vista, **slider** per lo zoom, **⟲ Vista
  completa** per tornare all'inquadratura iniziale;
- **📌 Salva immagine per confronto** congela l'immagine *con la vista corrente* (zoom e
  spostamento inclusi) nella galleria di confronto;
- la caption ricorda che viene mostrata **una sola immagine**: quella dell'ultima esecuzione
  con i parametri correnti, e che $w_0$ è sempre quello scelto da te (mai casuale);
- ⚠️ se la traiettoria **esce dal riquadro** calcolato compare un avviso: in quel caso allarga
  i limiti (i preset 2D hanno `XMIN/XMAX`, `YMIN/YMAX` nel codice della loss).

### 11.3 «Confronto immagini salvate»

La **galleria** con tutte le immagini che hai congelato con 📌. Le card si dispongono a griglia
(più colonne quando lo schermo lo permette). Serve per il confronto visivo che non si può fare
a numeri: *dove* il batch è cresciuto, *quanto* è diverso il percorso tra GD e Newton-CG,
*come cambia* la traiettoria aumentando $\theta$. Il pulsante **🗑️ Rimuovi tutte** svuota la
galleria.

### 11.4 «Analisi stocastica: n_k vs a^k»

Il grafico della tesi: sull'asse $x$ l'iterazione $k$, sull'asse $y$ la **dimensione del
batch** $n_k$.

- la traccia **`n_k (batch effettivo)`** è la dimensione realmente usata a ogni iterazione —
  una scaletta, perché il batch cresce e poi resta costante;
- la traccia **`aᵏ`** è il fit geometrico: l'app stima l'esponente $a$ che meglio approssima
  i dati risolvendo un problema ai minimi quadrati **in scala logaritmica**
  ($\log a = \frac{\sum_k k\log n_k}{\sum_k k^2}$) e lo mostra in legenda (`a = …`);
- l'annotazione riassume **`aᵏ · a = … · errore rel. medio = …%`**: se l'errore relativo è
  piccolo, la crescita del tuo esperimento è davvero geometrica, come previsto dalla teoria;
- un marcatore verticale indica l'**iterazione corrente** dell'animazione.

### 11.5 «$J_{val}(w_k)$ — stop adattivo»

Compare quando attivi **Usa validation set per stop adattivo**. Ogni punto è una valutazione
di $J_{val}$: quando la loss non migliora per **pazienza** valutazioni consecutive il
mini-batch viene **ricampionato** (e l'Hessiana, se legata). Dal grafico si vede il «dente di
sega» classico: discesa, plateau, ricampionamento, nuova discesa. Il marcatore `val corrente`
lega il grafico all'iterazione dell'animazione.

### 11.6 «$J_{batch}(w_k)$ — riuso per discesa»

Compare quando attivi **Riuso per discesa della loss sul batch**. Ogni punto è una valutazione
di $J_{batch}$: quando la riduzione relativa scende sotto la soglia per **pazienza**
valutazioni consecutive si ricampiona. È il grafico che risponde alla domanda: *quanto posso
spremere lo stesso mini-batch prima che convenga cambiarlo?*

## 12. 🧪 Analisi (tabella di convergenza)

Premi **🧪 Analisi** per avere i numeri dietro l'animazione. Se non hai ancora eseguito
l'algoritmo, l'app risponde *«⚠ Esegui prima Ricalcola.»*

La finestra mostra una tabella con una riga per iterazione:

| iter | ‖w−w\*‖ | J(w) | (con Newton-$L_1$: anche ‖∂F‖) |
|---|---|---|---|
| 0 | … | … | … |
| 1 | … | … | … |

…e una sintesi finale che classifica l'esito:

- **✓ Convergenza raggiunta · ‖w−w\*‖ = …** se la metrica di arresto è sotto $10^{-4}$;
- **⚠ Convergenza parziale · …** altrimenti.

Per Newton-$L_1$ la metrica usata è **‖∂F‖** (la norma del subgradiente, che va a zero nel
minimo di $F$), non ‖w−w\*‖, che è riferita al minimo di $J$ — un dettaglio che l'app spiega
essa stessa nel pannello.

È il modo più comodo per **confrontare due configurazioni** senza guardare i grafici: esegui,
leggi la colonna `iter`, cambia una leva, riesegui, confronta la riga finale.

## 13. 🧪 Test batch (le tabelle della tesi)

Il pannello *Esperimenti batch (tabelle tesi)* apre l'esperimento a **prodotto cartesiano**:
definisci un elenco di valori per ogni dimensione e l'app esegue **tutte** le combinazioni,
riportando l'errore $e_{30}$ (errore all'ultima iterazione) come nella **Sezione 6.5** della
tesi.

### 13.1 Le dimensioni disponibili

Ogni dimensione è una casella con elenco di valori separati da virgola, un'etichetta
**`media`** e un suggerimento. Le dimensioni sono:

| Dimensione | Default | Note |
|---|---|---|
| **Loss (preset)** | `quad_well,quad_ill,quad_very_ill,quad_offdiag,rosenbrock` | chiavi dei preset |
| **Algoritmo** | `gd,bb,newton_cg,newton_l1` | Dynamic GD, BB-CCV, Newton-CG, Newton-$L_1$ |
| **Strategia di riuso (colonne)** | `base,M=inf,M=10,M=5,M=2,H ind M_H=inf` | vedi §13.2: diventano le **colonne** della tabella |
| **Seed** | `42` | più seed con `media` ⇒ **robustezza** (es. i 5 seed della tesi) |
| **max_iter** | `30` | $e_{30}$ = errore all'ultima iterazione |
| **alpha (passo)** | `0.1` | |
| **θ (toll. CCV)** | `0.5` | |
| **batch0** | `5` | |
| **N (dataset)** | `200` | |
| **w0 (x,y)** | `2.0,-3.0` | coppie separate da `;`, es. `2.0,-3.0;1.0,1.0` |
| **Line search** | `wolfe` | `wolfe`, `armijo`, oppure per algoritmo `gd=wolfe;bb=armijo;newton_cg=wolfe` |
| **Batch dinamico CCV** | `dynamic` | `dynamic` / `fixed` |
| **Sottocamp. Hessiana** | `subset` | `subset` (da $S_k$) / `none` (da tutto $N$) |
| **Hessian-free L1** | `free` | `free` / `explicit` |
| **R (\|H\|/\|S\|)** | `0.2` | |
| **max CG iter** | `10` | |
| **ν (penalità L1)** | `0.1` | |
| **σ (Armijo)** | `0.1` | |
| **η (toll. CG)** | `0.5` | |

In alto vedi il conteggio in tempo reale: per esempio **`Run: 5 × 4 × 6 = 120`**. Spuntando
**`media`** su una dimensione, i risultati vengono **mediati** su quei valori e la dimensione
*scompare* dalle righe della tabella (è così che si costruisce la «robustezza» su più seed).

### 13.2 Il vocabolario delle strategie (le colonne)

| Valore | Significato |
|---|---|
| `base` | ricampionamento a ogni iterazione |
| `desc base` / `val base` | base «alla tesi»: il criterio con ricampionamento forzato |
| `M=inf`, `M=k` | riuso del mini-batch: illimitato o per $k$ iterazioni |
| `H ind M_H=inf`, `H ind M_H=k` | Hessiana **indipendente** da $S_k$, riusata per $M_H$ iterazioni |
| `val` | stop adattivo su validation set, con parametri `P=…;p=…;tau=…;f=…;strat=…;minabs=…` |
| `desc` | riuso per discesa della loss, con parametri `P=…;tau=…;f=…;minabs=…` |

Esempi pronti nei suggerimenti: `val P=1;p=0.1;tau=1e-4;f=1;strat=dynamic` e
`desc P=1;tau=1e-3;f=1`.

### 13.3 I formati di tabella

Il menu **Formato:**

| Formato | Cosa produce |
|---|---|
| **Matrice (confronto)** | la tabella completa: righe = combinazioni, colonne = strategie |
| **Robustezza (sintesi)** | sintesi su più seed |
| **Robustezza riuso Migl./Pegg./Ug. (Tab. 6.2)** | il formato della Tabella 6.2 della tesi |
| **Confronto finale (Tab. 6.7)** | il formato della Tabella 6.7 |
| **Sintesi consigliati (Tab. 6.8)** | il formato della Tabella 6.8 |

I tre formati «da tesi» usano le dimensioni *loss*/*algo* con i loro default e gli altri
parametri ai valori della finestra.

### 13.4 Esecuzione ed esportazione

- **▶ Avvia** esegue tutti i run con **barra di avanzamento** (`0/0` → `n/n`) e la possibilità
  di seguire l'avanzamento run per run;
- **⟲ Ripristina default** rimette le dimensioni ai valori iniziali;
- i risultati restano in uno **storico** salvato nel browser (`localStorage`), quindi
  ritrovi gli esperimenti fatti anche dopo aver chiuso la pagina;
- per ogni tabella hai tre strumenti: **📋 Copia LaTeX**, **💾 Scarica JSON** e
  **🔍 Codice 1ª run** (mostra il codice Python della prima esecuzione della tabella, utile per
  documentare esattamente cosa è stato calcolato).

> 📌 Il formato **Robustezza riuso (Tab. 6.2)** usa i cinque seed della tesi —
> `42, 7, 123, 2024, 999` — e confronta $M=\infty$ e $M=10$ contando in quanti seed il riuso
> è **migliore**, **peggiore** o **uguale** al ricampionamento a ogni iterazione.

## 14. 📘 Teoria dell'algoritmo selezionato

Il pannello di teoria è **cucito su misura sull'algoritmo scelto**: cambi metodo e il contenuto
cambia. È organizzato in card affiancate (griglia automatica) e contiene **descrizione,
pseudocodice e formule numerate**, tutte rese in LaTeX.

### 14.1 Gradiente a Campione Dinamico

Descrizione — *Condizione di Controllo della Varianza (CCV)*, *Aggiornamento batch*, *Batch
fisso*, *Iterazione*, *Pseudocodice*.
Formule:

1. Controllo della varianza (CCV)
2. Regola di aggiornamento del batch
3. Condizione di discesa
4. Convergenza deterministica
5. Iterazioni per precisione $\varepsilon$
6. Ipotesi di limitatezza della varianza
7. Convergenza stocastica
8. Complessità totale
9. Line search (**Armijo** oppure **Wolfe**, in funzione della modalità selezionata)

### 14.2 Newton-CG con Campionamento Dinamico

Descrizione — *Residuo del CG vs errore di Hessiana*, *Stima dell'errore di Hessiana*,
*Coefficiente $\gamma$ e soglia adattiva*, *Test di arresto del CG*, *Line search e
aggiornamento*, *Aggiornamento batch (CCV sul gradiente)*, *Batch fisso*, *Pseudocodice*.
Formule:

1. Sistema lineare (Hessian-free)
2. Residuo del CG vs errore di approssimazione dell'Hessiana
3. Stima dell'errore di Hessiana
4. Coefficiente $\gamma$ e soglia adattiva
5. Test di arresto del CG
6. Line search (Wolfe / Armijo)
7. Aggiornamento batch (CCV sul gradiente)

### 14.3 Newton-CG con Regolarizzazione $L_1$

Descrizione — *Gradiente generalizzato (subgradiente)*, *Faccia ortante e active set*,
*Minimizzazione nel sottospazio libero*, *Proiezione ortante e aggiornamento*, *Pseudocodice*.
Formule:

1. Funzione obiettivo ($F(w)=J(w)+\nu\|w\|_1$)
2. Gradiente generalizzato (subgradiente)
3. Identificazione dell'active set e della faccia ortante
4. Minimizzazione nel sottospazio
5. Ricerca lineare proiettata
6. Verifica CCV e re-campionamento

### 14.4 Barzilai–Borwein con Campionamento Dinamico

Card **Barzilai–Borwein con Campionamento Dinamico — Formule** + *Pseudocodice*: il passo BB
clippato, la line search di Armijo e il meccanismo di ricampionamento.

### 14.5 Algoritmo personalizzato

Card *Algoritmo Personalizzato* con *Interfaccia richiesta* e *Variabili disponibili* (§15).

> 🔎 **Il dettaglio che fa la differenza:** lo *pseudocodice* si adatta alle leve che accendi.
> Con il **riuso** attivo compare la versione con il ciclo interno sullo stesso mini-batch; con
> lo **stop adattivo** compare la versione con validation set. È il modo più comodo per
> verificare che l'implementazione che stai guardando (e che stai eseguendo) corrisponda
> davvero a ciò che descrivi nel testo.

## 15. Algoritmo personalizzato

Scegliendo **✏️ Algoritmo personalizzato** puoi implementare il tuo metodo e vederlo eseguire
insieme agli altri. L'interfaccia richiesta (testuale dall'app) è:

```python
def mio_algoritmo(w0, alpha, max_iter, theta, batch0, ...):
    ...
    return history, batch_sizes

history, batch_sizes = mio_algoritmo(w0, alpha, max_iter, theta, batch0, ...)
```

- **Input:** `w0`, `alpha`, `max_iter`, `theta`, `batch0` (più `N` dal pannello parametri).
- **Variabili disponibili:** `w0`, `alpha`, `max_iter`, `theta`, `batch0`, `N`, `loss_i`,
  `grad_i`, `hess_i`, `hessvec_i`, `grad_full`, `J`.
- **Output:** `history` = lista dei punti $w_k$, `batch_sizes` = lista delle dimensioni $n_k$.

Restituendo `batch_sizes` hai gratis tutto il resto: il grafico $n_k$ vs $a^k$, i marcatori
dei ricampionamenti, l'animazione, la tabella di convergenza. Questo rende l'app un banco di
prova anche per varianti **non** presenti nella tesi.

## 16. Esperimenti guidati

Otto ricette pronte: parametri concreti e cosa guardare. Tutte partono dai default dell'app.

**① Quanto conta la soglia $\theta$?**
Preset *Quadratica mal condizionata* · algoritmo *Dynamic GD* · $\theta = 0.1,\,0.3,\,0.5,\,0.7$
(uno per volta, premendo **Ricalcola** e salvando l'immagine con 📌).
👉 Guarda: il valore finale di $n_k$, l'esponente $a$ e i punti «CCV violata».

**② Dinamico vs batch fisso.**
Stesso preset · **Batch dinamico (CCV): `Disattivato (batch fisso)`** con `batch0 = 5`.
👉 Guarda: quante iterazioni servono per raggiungere lo stesso $\|w_k-w_\*\|$; poi riaccendi il
dinamico e confronta quanti esempi totali ha «speso» ciascuno dei due.

**③ Quanto posso riusare lo stesso mini-batch?**
Preset mal condizionata · **riuso attivo** con *Max iterazioni consecutive* `Personalizzato` a
$M = 1, 2, 3, 5$ e poi `Illimitato`.
👉 Guarda: la scheda **M_actual**, il grafico $J_{batch}(w_k)$ e il numero di ricampionamenti.

**④ Stop adattivo: validation o discesa?**
Preset *molto mal condizionata* · attiva prima **Usa validation set**, poi **Riuso per discesa**.
👉 Guarda: i «denti di sega» di $J_{val}(w_k)$ e $J_{batch}(w_k)$; cambia *pazienza* (1 → 10) e
vedi come cambia la frequenza dei ricampionamenti.

**⑤ GD contro Newton sulla mal condizionata.**
Preset $\kappa\approx100$ · esegui *Dynamic GD*, salva l'immagine; passa a *Newton-CG*, esegui,
salva. Poi sperimenta `R = 0.05, 0.2, 0.5`.
👉 Guarda: il percorso, il numero di iterazioni, e quanto piccola può restare l'Hessiana
($|H_k| = R\,|S_k|$) senza perdere convergenza.

**⑥ $L_1$, faccia ortante e Hessian-free.**
Algoritmo *Newton-CG con Regolarizzazione $L_1$* con $\nu = 0.1$ e poi $\nu = 0.5$,
**Mostra anche F(w)** spuntato. Poi confronta **Hessian-free** vs **Hessiana esplicita**.
👉 Guarda: la scheda **‖∂F‖**, la traiettoria sulla faccia ortante e la differenza tra minimo di
$J$ e minimo di $F$.

**⑦ Fuori dal caso quadratico.**
Preset *Rosenbrock*: la teoria della CCV non richiede che $J$ sia quadratica.
👉 Guarda: la forma della traiettoria e la crescita di $n_k$ lungo la valle curva.

**⑧ La tua variante.**
*Algoritmo personalizzato*: parti dal codice di Dynamic GD, poi cambia la regola (per esempio
aumenta $n_k$ solo ogni due violazioni consecutive) e confronta con la CCV «pura».
👉 Guarda: `batch_sizes` (anche stampandolo), il grafico $n_k$ vs $a^k$ e la tabella di
convergenza.

## 17. Come leggere i risultati

**$n_k$ vs $a^k$.** Il fit geometrico è il cuore della teoria: la dimensione del campione deve
crescere come $a^k$. Se l'**errore relativo medio** mostrato nell'annotazione è piccolo, il tuo
esperimento conferma l'andamento previsto; se è grande, spesso significa che il batch ha
**saturato** (vedi sotto) o che le iterazioni utili sono troppo poche.

**Saturazione a $N$.** Quando $n_k$ raggiunge la dimensione del dataset, la CCV non può più
essere soddisfatta aumentando il campione: il metodo è di fatto diventato **full-batch**. Non è
un errore, è la risposta del metodo a un obiettivo troppo mal condizionato per $\theta$ scelto.

**$\theta$ piccolo ⇒ batch grandi.** $\theta$ è la tolleranza *relativa* nella CCV: chiedendo
un gradiente campionario più preciso spendi più esempi per iterazione, ma fai passi più
informati. Il confronto interessante è sempre *a parità di iterazioni* e poi *a parità di
esempi consumati* ($\sum_k n_k$).

**Quale metrica guardare.**

| Situazione | Metrica | Perché |
|---|---|---|
| Convergenza generale | $\|w_k-w_\*\|$ | distanza dal minimo vero: la più onesta |
| Vicinanza alla stazionarietà | $\|\nabla J(w_k)\|$ | indipendente dal minimo di riferimento |
| Newton-$L_1$ | $\|\partial F(w_k)\|$ | è il criterio corretto: $w_\*$ è il minimo di $F$, non di $J$ |
| Qualità della soluzione | $J(w_k)$ | ma è «schiacciata» vicino al minimo: usala in scala logaritmica |

**I due tipi di ricampionamento.** Nei marcatori distingui **CCV violata (batch aumentato)**
— il meccanismo della tesi — da **Ricampionamento (non-CCV)**, che è l'effetto delle leve
sperimentali (riuso, pazienza esaurita, cambio di Hessiana). Quando studi il metodo «puro»,
questi ultimi non devono esserci.

**`M_actual`.** È la prova immediata che il riuso sta funzionando: se vale 1, stai
ricampionando a ogni iterazione; se cresce, stai spremendo il mini-batch.

**«Convergenza parziale» è spesso normale.** Con un $L_1$ con $\nu$ grande, o con $N$ piccolo,
o con `max_iter` basso, la soglia $10^{-4}$ può non essere raggiunta pur essendo il metodo
perfettamente sensato: guarda la colonna `iter` e l'andamento di $J(w_k)$.

## 18. Dove sta cosa nella tesi

| Cosa vedi nell'app | Dove è descritto nella tesi |
|---|---|
| Il problema, la dimensione del mini-batch, il campionamento dinamico | Cap. 1 (§1.2–§1.3) |
| Preset e funzione obiettivo $J$, ipotesi su $J$ | Cap. 3 (§3.1–§3.2) |
| Perché i metodi stocastici, SVRG/SAGA, campionamento dinamico, $L_1$ | Cap. 4 (§4.1–§4.4) |
| **Dynamic GD**: CCV, regola di aggiornamento del batch, pseudocodice, convergenza | §5.1 (incl. §5.1.2 condizione di accettazione, §5.1.3 regola di aggiornamento, §5.1.4 pseudocodice, §5.1.5–§5.1.6 analisi e complessità) |
| **$n_k$ vs $a^k$**, stima dell'esponente $a$, complessità | §5.1.5 (analisi stocastica e complessità) e Figura 5.3 |
| **Newton-CG**: sistema Hessian-free, criterio di arresto del CG, $\gamma$, $R$, pseudocodice | §5.2 (incl. §5.2.1 struttura, §5.2.2 criterio di terminazione, §5.2.3 pseudocodice) |
| **Newton-CG $L_1$**: subgradiente, active set, faccia ortante, proiezioni, ricerca proiettata | §5.3 (tutte le sottosezioni) |
| **BB-CCV** (passo Barzilai–Borwein + CCV) | §5.4 (incl. §5.4.1 il metodo BB, §5.4.2 schema dell'algoritmo) |
| Setup sperimentale ($N$, seed, $w_0$, iperparametri) | §6.1 |
| Architettura software (algoritmi e dati) | §6.2 |
| Risultati numerici ($e_{30}$ e confronti) | §6.3 |
| **L'applicazione web stessa** (questa app) | §6.4 *Visualizzazione Interattiva* |
| **Leve di riuso**: $M$, $M_H$, riuso dell'Hessiana, stop adattivo con validation set, riuso per discesa della loss, iperparametri consigliati | §6.5 (Riuso del mini-batch) — sottosezioni: descrizione, meccanismo, setup, risultati, sintesi, stop adattivo, discesa, consigliati |
| Formati delle tabelle del **Test batch** | Tabelle 6.2, 6.7, 6.8 di §6.5 |
| Codice Python degli algoritmi (l'editor modificabile) | Appendice B (B.1 Dynamic GD, B.2 Newton-CG, B.3 Newton-CG $L_1$, B.4 BB-CCV) |
| Requisiti, installazione, avvio dell'app | Appendice C (in particolare C.5 «Avvio dell'applicazione web») |
| Riassunto dei risultati, limiti e direzioni future | §7.1–§7.3 |

## 19. Note, limiti e piccoli trucchi

**🔌 Serve internet al primo avvio.** L'app è un singolo file `.html` *autonomo* — non c'è
nessuna cartella da copiare — ma carica le librerie da CDN (Pyodide, Plotly, MathJax, font).
Dopo il primo caricamento tutto è in cache e l'app funziona anche offline.

**🐢 Pyodide non è Python nativo.** È Python compilato in WebAssembly: comodo per
sperimentare, ma con `N` grande, `max_iter` alto e Newton-CG su problemi 2D i calcoli
richiedono qualche secondo. Se ti serve una scansione pesante, il **Test batch** è la strada
giusta: prepara la griglia e lascia lavorare la barra di avanzamento.

**📐 I problemi 1D sono un caso a parte.** Nei preset 1D il minimo $w_\*$ non è noto in forma
chiusa: l'app lo calcola **numericamente** con un Newton su differenze finite, come si vede nel
codice del preset. È il motivo per cui nello script Python 1D compare la funzione `_wstar()`.

**🧪 Il dataset è sintetico e «centrato».** Da ogni coefficiente viene sottratta la sua media
campionaria, così le medie sono *esattamente* quelle nominali e, per i preset quadratici, la $J$
campionaria coincide con quella nominale a meno di una costante: le formule esatte (loss e
rumore, preset per preset) sono in **§5.5**. È una comodità sperimentale — nel machine learning
reale $J$ è una *stima* del rischio vero, e questo è uno dei motivi per cui il campione deve
essere scelto con cura.

**💾 Lo storico del Test batch** vive nel `localStorage` del browser: se lui è pieno, l'app
avvisa e lo storico non viene salvato (l'esperimento resta comunque valido e scaricabile in
JSON).

**Trucchi veloci.**

- La **legenda è cliccabile**: spegni la superficie o il percorso per vedere meglio l'altro
  (leggibilità enorme sui grafici 3D).
- L'icona **fotocamera** nella barra del grafico salva un PNG di quella vista: comodissimo per
  la tesi, senza passare dalla galleria.
- Con la **slider dell'iterazione** ti fermi esattamente sull'iterazione in cui il batch è
  cresciuto, poi apri la *Traiettoria 2D* e salvi l'immagine.
- Cambiando il **seed** vedi quanto i risultati dipendono dai dati: è il modo più rapido per
  capire se una differenza tra due configurazioni è reale o è rumore.
- L'editor **«Algoritmo (modificabile)»** accetta modifiche: cambia un dettaglio, premi
  **Ricalcola** e guarda subito l'effetto. Per tornare all'originale c'è **↻ Ripristina**.

---

**Guida riferita a `visualizzazione.html`** — applicazione web interattiva della tesi
*«Selezione Dinamica della Dimensione del Campione in Metodi di Ottimizzazione per il Machine
Learning»* (Sapienza Università di Roma, SMIA, A.A. 2025–2026).
Tutte le etichette, i valori di default e i messaggi citati in questa guida sono quelli
effettivamente presenti nell'applicazione.
