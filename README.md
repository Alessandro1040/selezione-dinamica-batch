# Selezione Dinamica della Dimensione del Campione

Simulazione autonoma del **metodo del gradiente a campione dinamico** basato
sulla *Condizione di Controllo della Varianza* (CCV).

Questo repository accompagna la tesi *"Selezione Dinamica della Dimensione
del Campione in Metodi di Ottimizzazione per il Machine Learning"* e
riproduce la **Figura 5.3**: l'andamento della dimensione del batch $n_k$
nel corso delle iterazioni, confrontato con un batch fisso.

## Contenuto

| File | Descrizione |
|---|---|
| `simulazione_batch.py` | Script Python autocontenuto (unico file, dipende solo da `numpy` e `matplotlib`) |
| `figure_sim/` | Figure generate: `batch_size.pdf/png` (Fig. 5.3) e `convergenza.pdf/png` (Fig. 6.1, compagnona) |

## Come eseguire

```bash
python3 simulazione_batch.py
```

Le figure vengono salvate nella cartella `figure_sim/`.

## Parametri della simulazione (identici a `sim_exp.py` della tesi)

| Parametro | Valore | Significato |
|---|---|---|
| `N` | 1000 | numero di esempi |
| `m` | 10 | dimensionalità dei parametri |
| `THETA` | 0.5 | tolleranza nella CCV |
| `BATCH0` | 16 | batch iniziale |
| `MAX_ITER` | 250 | budget di iterazioni |
| `TOL` | 1e-6 | tolleranza del criterio di arresto |
| `SEED` | 42 | seme per la riproducibilità |

Problema: regressione lineare (loss quadratica), $X \sim \mathcal{N}(0, I)$,
$\kappa \approx 1.4$, $w_0 = (2,\dots,2)$.

## Regola CCV

Se $\dfrac{\|\widehat{\mathcal V}\|_1}{n} > \theta^2 \|g\|^2$, allora

$$n \leftarrow \min\left(\left\lceil \dfrac{\|\widehat{\mathcal V}\|_1}{\theta^2 \|g\|^2}\right\rceil + 1,\ N\right).$$

## Riferimento

R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu, *Sample size selection in
optimization methods for machine learning*, Mathematical Programming, 2012.
