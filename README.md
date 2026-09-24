# trial-matcher

Clinical trial matching sul dataset TREC Clinical Trials: data la nota di
ammissione di un paziente, trova i trial per cui è eleggibile.

Architettura in due fasi:

1. **Coarse retrieval** — ricerca ibrida (dense + BM25, fusi con RRF) sul corpus
   dei trial, produce una shortlist.
2. **Fine-grained eligibility assessment** — per ogni trial candidato, un LLM
   locale verifica ogni criterio atomico sulla nota del paziente
   (`yes` / `no` / `unclear`) citando la frase che lo dimostra. I verdetti
   riordinano la shortlist.

Il punto è la classe `1` dei qrels (*excluded*): trial semanticamente pertinenti
al paziente ma clinicamente sbagliati per lui. Un ranker basato sulla similarità
non li distingue; l'assessment criterio per criterio dovrebbe.

## Prerequisiti

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com) per la seconda fase
- git

Qdrant gira in modalità locale dentro Python: l'indice è una cartella in
`data/qdrant`, nessun server da avviare.

## Setup

```bash
uv sync                              # ricrea l'ambiente da uv.lock
cp .env.example .env                 # configurazione locale
uv run pytest                        # verifica che tutto risponda
```

## Dati

```bash
uv run python -m src.ingest.download              # topics + qrels
uv run python -m src.ingest.download --corpus     # + 375k trial XML (~10 GB)
uv run python -m src.ingest.parse_topics          # -> topics{2021,2022}.jsonl
uv run python -m src.ingest.parse_trials          # -> trials_judged.jsonl
uv run python -m src.ingest.parse_trials --no-only-judged   # corpus completo
uv run python -m src.ingest.criteria              # -> criteria.jsonl
```

`--only-judged` (default) tiene solo i 48.714 trial che compaiono nei qrels:
è il subset di sviluppo. Le metriche su questo subset sono gonfiate — si cerca
solo fra documenti già giudicati — e vanno usate per il debug, non per il report.

## Indice e valutazione

```bash
uv run python -m src.index.build                  # dense + BM25 -> data/qdrant
uv run python -m src.eval.run --year 2021         # run file TREC + metriche
uv run python -m src.retrieve.search "testo della nota" --mode hybrid --limit 10
```

Protocollo: **2021 come development set**, **2022 come test set**, toccato una
volta sola alla fine.

## Analisi

```bash
uv run python -m src.analysis.qrels     # distribuzione dei giudizi + figure
uv run python -m src.analysis.corpus    # struttura degli XML
uv run python -m src.analysis.corpus --sample 5000   # su un campione
```

## Struttura

- `src/ingest/` — download, parsing XML, splitting dei criteri
- `src/index/` — embedding e indicizzazione su Qdrant
- `src/retrieve/` — hybrid search e fusione RRF
- `src/eval/` — run file TREC e metriche
- `src/assess/` — verifica dei criteri con LLM (in corso)
- `src/analysis/` — statistiche su qrels e corpus
- `data/` — corpus e indice, **non versionati**
- `runs/` — run file generati, **non versionati**
- `reports/figures/` — figure per la relazione

## Comandi utili

```bash
uv run ruff check .       # lint
uv run ruff format .      # formattazione
uv run pytest             # test
```
