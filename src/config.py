import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _dir(name: str, default: Path) -> Path:
    """Overridable from the environment, so the same commands run where the
    inputs are mounted read-only and the outputs have to go somewhere else."""
    return Path(os.getenv(name, default)).expanduser()


DATA_RAW = _dir("DATA_RAW", ROOT / "data" / "raw")
DATA_PROCESSED = _dir("DATA_PROCESSED", ROOT / "data" / "processed")
# Verdicts and other written artefacts: the same place, unless the inputs are
# read-only.
DATA_OUT = _dir("DATA_OUT", DATA_PROCESSED)
RUNS = _dir("RUNS", ROOT / "runs")

# Qdrant in local mode: an on-disk collection opened in-process, no server.
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = os.getenv("QDRANT_COLLECTION", "trials")

# Named vectors inside a point. Indexing and search must agree on these, so
# they live here rather than being repeated as literals in both modules.
DENSE = "dense"
SPARSE = "bm25"

# bge-*-v1.5 are asymmetric: the instruction goes on the query, never on the
# document. MedEmbed is a bge fine-tune, so it inherits the convention.
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True, slots=True)
class Encoder:
    model: str
    prefix: str = ""


ENCODERS = {
    "bge-small": Encoder("BAAI/bge-small-en-v1.5", BGE_PREFIX),
    "bge-base": Encoder("BAAI/bge-base-en-v1.5", BGE_PREFIX),
    "medembed-small": Encoder("abhinand/MedEmbed-small-v0.1", BGE_PREFIX),
    "medembed-base": Encoder("abhinand/MedEmbed-base-v0.1", BGE_PREFIX),
}
# Chosen on the 2021 ablation: the medical fine-tune beats the general model of
# the same size, and growing inside the medical family adds nothing measurable.
DEFAULT_ENCODER = "medembed-small"


def collection_for(encoder: str) -> str:
    """One collection per encoder. The name always carries the encoder, so a
    query can never land in an index built by a different model: two encoders
    of equal width would not even raise an error."""
    return f"{COLLECTION}_{encoder.replace('-', '_')}"


def get_device() -> str:
    """Best available torch device, so the code runs on a peer machine too.

    torch is imported here and not at the top: judging criteria needs neither
    an encoder nor a GPU, and that machine should not have to install it."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
