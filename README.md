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

> Un'interfaccia grafica è prevista: scrivere la nota, vedere i trial, e per
> ciascuno il dettaglio dei criteri con l'evidenza evidenziata nel testo.
> Finché non esiste, tutto passa dalla riga di comando.

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

Circa 35 secondi per trial. I verdetti finiscono in `data/processed/verdicts2021.jsonl`
con una chiave che comprende modello e versione del prompt: interrompere e
rilanciare riprende da dove era arrivato, e cambiare il prompt non riusa mai i
verdetti vecchi.

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

## Struttura

- `src/ingest/` — download, parsing degli XML, splitting dei criteri
- `src/index/` — embedding e indicizzazione su Qdrant
- `src/retrieve/` — ricerca densa, BM25 e fusione RRF
- `src/eval/` — run file in formato TREC e metriche
- `src/assess/` — verifica dei criteri con l'LLM e riordino
- `src/analysis/` — statistiche su qrels e corpus
- `data/`, `runs/` — dati, indice e risultati, **non versionati**

## Sviluppo

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```
