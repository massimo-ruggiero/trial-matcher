import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Qdrant in local mode: an on-disk collection opened in-process, no server.
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = os.getenv("QDRANT_COLLECTION", "trials")

# Named vectors inside a point. Indexing and search must agree on these, so
# they live here rather than being repeated as literals in both modules.
DENSE = "dense"
SPARSE = "bm25"


def get_device() -> str:
    """Best available torch device, so the code runs on a peer machine too."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
