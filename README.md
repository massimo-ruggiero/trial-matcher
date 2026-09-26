# trial-matcher

Data la nota di ammissione di un paziente, trova i trial clinici per cui è
eleggibile, fra i 375.580 del dataset TREC Clinical Trials.

Il problema non è trovare trial che parlano della sua malattia: quello lo fa
qualsiasi motore di ricerca. Il problema è che molti di quei trial **il paziente
non può farli**, perché viola un criterio di esclusione — ha già fatto una certa
terapia, ha un'età fuori range, ha una patologia concomitante. Nei giudizi dei
medici del TREC questi trial hanno una categoria propria, *excluded*, e nel 2021
sono **più numerosi di quelli eleggibili**.

Un sistema di ricerca non può distinguerli: un trial escluso somiglia alla nota
del paziente quanto uno adatto, a volte di più. Da qui l'architettura in due fasi:

1. **Ricerca** — la nota diventa un vettore, si recuperano i candidati
2. **Verifica** — per ogni trial candidato un LLM locale controlla **criterio per
   criterio** se il paziente li soddisfa, citando la frase della nota che lo
   dimostra; i verdetti riordinano la lista

## Interfaccia

```bash
uv run streamlit run gui.py
```

Si scrive la nota del paziente (o si carica uno dei topic del TREC) e si cercano
i trial. Ogni risultato si può aprire subito: i criteri sono leggibili prima di
spendere tempo di modello, divisi in inclusione ed esclusione. Il pulsante
*Verify* manda il modello a giudicarli uno per uno, circa 35 secondi; *Verify
all* mette in coda tutta la lista e sblocca ogni trial appena è pronto, mentre
gli altri restano consultabili.

Verificato un trial, accanto a ogni criterio compare **la frase della nota che
lo decide**, evidenziata anche nella nota stessa. Ogni citazione viene cercata
nella nota: se non si trova alla lettera è marcata `(not in the note)`, perché
un modello che riscrive la cartella mentre dice di citarla è la cosa che va
vista per prima. Dove non c'è niente da citare — o dove la citazione da sola non
basta — il modello scrive una motivazione, marcata `WHY` per distinguerla dal
testo reale.

I colori delle righe seguono **la conseguenza per il paziente, non il verdetto**:
un criterio di esclusione risultato vero è rosso anche se la risposta è "sì".

Il primo avvio richiede una ventina di secondi, di cui la metà per aprire
l'indice: Qdrant in locale carica in memoria i vettori di ogni collection
presente in `data/qdrant`, quelle delle ablation comprese. Poi encoder e indice
restano in cache e ogni interazione è immediata. Serve Ollama acceso per la
verifica, e l'app va chiusa prima di lanciare `build` o `eval`, perché Qdrant in
locale accetta un processo alla volta.

## Prerequisiti

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com) con `ollama pull gemma4:12b`, per la seconda fase
- git

Qdrant gira in modalità locale dentro Python: l'indice è una cartella in
`data/qdrant`, nessun server da avviare.

```bash
uv sync
cp .env.example .env
uv run pytest
```

## Come si usa

**Cercare** fra i trial indicizzati:

```bash
uv run python -m src.retrieve.search "75 yo M with metastatic papillary thyroid cancer" --mode dense --limit 10
```

**Giudicare** i criteri dei trial in cima alla lista di un paziente. Legge la
shortlist da un run file già prodotto, quindi non tocca l'indice:

```bash
uv run python -m src.assess.judge --topics 12-31 --depth 50 --run dense_medembed-small
```

Circa 35 secondi per trial. Per ogni criterio il modello restituisce le
citazioni, una motivazione quando serve e il verdetto, **in quest'ordine**: la
generazione segue l'ordine dei campi, quindi deve cercare una prova prima di
poter decidere, non giustificare a posteriori una decisione già presa.

I verdetti finiscono in `data/processed/verdicts2021.jsonl` con una chiave che
comprende modello e versione del prompt: interrompere e rilanciare riprende da
dove era arrivato, e cambiare prompt o schema non riusa mai i verdetti vecchi.

**Riordinare** e misurare il risultato:

```bash
uv run python -m src.assess.rerank
```

Non chiama nessun modello, rilegge i verdetti salvati. Per questo le varianti di
aggregazione costano secondi: `--unclear`, `--scoring`, `--grounded-only`.

## Come abbiamo scelto i modelli

Ogni scelta viene da una misura sul development set (2021), non da un'intuizione.

**Encoder: `MedEmbed-small`.** MedEmbed è un fine-tune medico di BGE — stessa
architettura, stesse dimensioni — quindi confrontarli isola il dominio dalla
taglia:

| encoder | parametri | nDCG@10\* | recall@50 |
|---|---|---|---|
| bge-small | 33M | 0.2511 | 0.180 |
| **medembed-small** | 33M | **0.2991** | **0.205** |
| bge-base | 109M | 0.2973 | 0.194 |
| medembed-base | 109M | 0.3065 | 0.218 |

Il modello medico da 33M batte quello generale da 109M. Fra small e base medici
la differenza non è distinguibile dal rumore (confronto appaiato su 75 topic:
17 vittorie contro 18), quindi si sceglie il più economico.

\* solo i trial eleggibili contano come rilevanti: la metrica standard assegna
punti anche ai trial *excluded*, cioè proprio all'errore che vogliamo correggere.

**Solo ricerca densa.** La fusione RRF con BM25 conviene finché i due segnali
sono vicini: con bge-small l'ibrido era il migliore, con MedEmbed il denso da
solo vince su ogni metrica stretta.

**Fusione calcolata da noi, con k=60.** Quella integrata in Qdrant usa
`1/(1+rank)`, che annulla il premio all'accordo fra le due liste e costa 8 punti
percentuali di nDCG@10.

**Giudice: `gemma4:12b`.** Fra i modelli locali provati è l'unico che restituisce
un verdetto per ogni criterio senza saltarne e senza inventare citazioni.

## Riprodurre il benchmark

```bash
uv run python -m src.ingest.download --corpus     # topics, qrels, 375k XML (~10 GB)
uv run python -m src.ingest.parse_topics
uv run python -m src.ingest.parse_trials          # -> trials_judged.jsonl (48.714)
uv run python -m src.index.build                  # ~4 min
uv run python -m src.eval.run --year 2021         # run file TREC + metriche
```

`--only-judged` (default in `parse_trials`) tiene solo i trial che compaiono nei
giudizi: è il subset di sviluppo. **Le metriche su questo subset sono gonfiate**,
perché mancano i 327.000 trial che nella realtà farebbero concorrenza. Vanno
usate per confrontare i sistemi fra loro, non come risultato.

Per confrontare encoder diversi, ognuno nella sua collection:

```bash
uv run python -m src.index.build --encoder bge-base
uv run python -m src.eval.run --year 2021 --encoder bge-base
```

Gli encoder disponibili sono in `config.ENCODERS`. Lo splitter dei criteri
(`src.ingest.criteria`) non serve più alla pipeline — il giudice riceve il blocco
intero — ma resta per le statistiche sul corpus.

**Protocollo:** i topic 1-11 del 2021 sono serviti per tarare prompt e parametri
e non producono numeri riportabili; i topic 12-75 sono la misura di sviluppo; il
2022 è il test set, toccato una volta sola alla fine.

## Confrontare i giudici su Kaggle

Il giudizio dei criteri non tocca né l'indice né la GPU per gli embedding: legge
una shortlist già prodotta e parla con Ollama. Si sposta quindi su una macchina
con GPU gratuita senza portarci il corpus.

Si costruisce il pacchetto di input — topic, shortlist e i soli blocchi dei
trial in lista, circa 2 MB invece di 239:

```bash
uv run python -m src.analysis.pack --topics 12-31 --depth 20
```

Il pacchetto contiene anche `src/` — 74 KB di Python — perché questo repository
è privato e un notebook non ha credenziali per clonarlo: un `git clone` via
https resterebbe fermo ad aspettare una password che nessuno digiterà. Se il
repository diventa pubblico, si può clonare e saltare questa parte.

Produce `kaggle/pack.zip` (0,6 MB), che si trascina in *New Dataset* su Kaggle:
lo estrae mantenendo le cartelle. Il nome del dataset è indifferente, il
notebook cerca i file per nome.

Il notebook si carica da *Import Notebook*, prendendo `kaggle/judge_ablation.ipynb`
da qui. Installa Ollama e giudica la stessa shortlist con ogni modello. Servono
**GPU T4 x2** e **internet attivo** nelle impostazioni.

Il codice viaggia con i dati, quindi **una modifica locale arriva su Kaggle solo
ricostruendo il pacchetto**. Il riepilogo stampato dal giudice riporta la
versione del prompt: serve anche a riconoscere una copia vecchia.

La prima esecuzione ha `SMOKE = True`: un topic, un trial, pochi minuti, per
verificare che i modelli si carichino e che i verdetti finiscano dove devono.
Poi `SMOKE = False` e *Save Version*, che gira in batch a browser chiuso.

I percorsi sono variabili d'ambiente, perché su Kaggle l'input è in sola
lettura:

| variabile | dove |
|---|---|
| `DATA_PROCESSED` | topic e blocchi dei criteri, in lettura |
| `RUNS` | i run file |
| `DATA_OUT` | i verdetti, in scrittura |

Il file dei verdetti si scarica e si accoda a `data/processed/verdicts2021.jsonl`;
la chiave `(topic, trial, modello, prompt)` rende l'unione idempotente. Ricostruire
il pacchetto a quel punto **ci mette dentro anche i verdetti**, quindi una nuova
versione del dataset contiene lo stato di tutti e ogni sessione riparte da lì:

```
1000 shortlisted, 855 already judged, 145 to go, from topic 29
```

Il pacchetto è lo stato completo dell'esperimento, non i soli dati di partenza.

**Protocollo dell'ablation.** Trentadue topic (12-43) per profondità 50 fanno
1.600 trial e circa 29.000 criteri per modello, con un modello per persona in
parallelo. La profondità viene da dove la curva dell'oracolo si appiattisce: fino
a 50 ogni trial in più allarga lo spazio in cui un giudice può migliorare la
classifica, oltre no. Su 32 topic il confronto appaiato distingue differenze di
nDCG@10\* da 0.045 in su, su uno spazio di 0.53.

È anche la configurazione della run finale, quindi il confronto produce già il
risultato di sviluppo, per tutti i modelli invece che per il solo vincitore. I
topic si giudicano in ordine crescente: chi si ferma prima resta confrontabile
sul prefisso comune.

## Struttura

- `src/ingest/` — download, parsing degli XML, splitting dei criteri
- `src/index/` — embedding e indicizzazione su Qdrant
- `src/retrieve/` — ricerca densa, BM25 e fusione RRF
- `src/eval/` — run file in formato TREC e metriche
- `src/assess/` — verifica dei criteri con l'LLM e riordino
- `src/analysis/` — statistiche su qrels e corpus, pacchetto di input per Kaggle
- `data/`, `runs/` — dati, indice e risultati, **non versionati**

## Sviluppo

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```
