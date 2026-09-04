# legal-rag

Sistema di Retrieval-Augmented Generation su normativa italiana con
architettura two-pass e verifica claim-by-claim delle citazioni.

## Prerequisiti

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (Docker Desktop su macOS/Windows)
- git

## Setup

```bash
git clone https://github.com/<username>/legal-rag.git
cd legal-rag

uv sync                              # ricrea l'ambiente da uv.lock
cp .env.example .env                 # configurazione locale
mkdir -p data/raw data/processed     # non versionate

docker compose up -d                 # avvia Qdrant
uv run pytest                        # verifica che tutto risponda
```

Dashboard Qdrant: http://localhost:6333/dashboard

## Struttura

- `src/ingest/` — download, parsing, normalizzazione URN, chunking
- `src/index/` — embedding e indicizzazione su Qdrant
- `src/retrieve/` — hybrid search e fusione RRF
- `data/` — corpus e indice, **non versionati**

## Comandi utili

```bash
docker compose up -d      # avvia Qdrant
docker compose stop       # ferma, mantiene i dati
docker compose logs -f    # log del servizio
uv run ruff check .       # lint
uv run ruff format .      # formattazione
uv run pytest             # test
```