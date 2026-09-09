# trial-matcher

Clinical trial matching sul dataset TREC Clinical Trials: data la nota di
ammissione di un paziente, trova i trial per cui è eleggibile.

Architettura in due fasi:

1. **Coarse retrieval** — ricerca ibrida (dense + BM25, fusi con RRF) sul corpus
   dei trial, produce una shortlist.
2. **Fine-grained eligibility assessment** — per ogni trial candidato, un LLM
   valuta ogni criterio atomico contro la nota del paziente e restituisce
   `Met` / `Violated` / `Undetermined` con l'evidenza citata. I verdetti
   riordinano la shortlist.

Il punto è la classe `1` dei qrels (*excluded*): trial semanticamente pertinenti
al paziente ma clinicamente sbagliati per lui. Un ranker basato sulla similarità
non li distingue; l'assessment criterio per criterio dovrebbe.

## Prerequisiti

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (Docker Desktop su macOS/Windows)
- git

## Setup

```bash
uv sync                              # ricrea l'ambiente da uv.lock
cp .env.example .env                 # configurazione locale
docker compose up -d                 # avvia Qdrant
uv run pytest                        # verifica che tutto risponda
```

Dashboard Qdrant: http://localhost:6333/dashboard

## Dati

```bash
uv run python -m src.ingest.download              # topics + qrels
uv run python -m src.ingest.download --corpus     # + 375k trial XML (~10 GB)
uv run python -m src.ingest.parse_topics          # -> topics{2021,2022}.jsonl
uv run python -m src.ingest.parse_trials          # -> trials_judged.jsonl
uv run python -m src.ingest.parse_trials --no-only-judged   # corpus completo
```

`--only-judged` (default) tiene solo i 48.714 trial che compaiono nei qrels:
è il subset di sviluppo. Le metriche su questo subset sono gonfiate — si cerca
solo fra documenti già giudicati — e vanno usate per il debug, non per il report.

## Analisi

```bash
uv run python -m src.analysis.qrels     # distribuzione dei giudizi + figure
uv run python -m src.analysis.corpus    # struttura degli XML
uv run python -m src.analysis.corpus --sample 5000   # su un campione
```

Protocollo: **2021 come development set**, **2022 come test set**, toccato una
volta sola alla fine.

## Struttura

- `src/ingest/` — download, parsing XML, splitting dei criteri
- `src/index/` — embedding e indicizzazione su Qdrant
- `src/retrieve/` — hybrid search e fusione RRF
- `src/analysis/` — statistiche su qrels e corpus
- `data/` — corpus e indice, **non versionati**
- `reports/figures/` — figure per la relazione

## Comandi utili

```bash
docker compose up -d      # avvia Qdrant
docker compose stop       # ferma, mantiene i dati
docker compose logs -f    # log del servizio
uv run ruff check .       # lint
uv run ruff format .      # formattazione
uv run pytest             # test
uv run pytest -m "not integration"   # test senza Qdrant
```
