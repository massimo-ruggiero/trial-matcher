import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum

# Fixed namespace for deterministic point IDs. Never change it: doing so
# remaps every point in the collection and turns a re-index into a duplicate.
NAMESPACE = uuid.UUID("6f1e2a3c-4b5d-4e6f-8a9b-0c1d2e3f4a5b")


class Kind(StrEnum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class Source(StrEnum):
    """How a criterion's kind was decided. Recorded so the cost of guessing can
    be measured instead of assumed."""

    HEADER = "header"  # an explicit "Inclusion/Exclusion Criteria" line
    INFERRED = "inferred"  # guessed from the wording, e.g. "must not have"
    DEFAULT = "default"  # no signal anywhere: assumed inclusion


@dataclass(frozen=True, slots=True)
class Topic:
    topic_id: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Trial:
    nct_id: str
    title: str
    summary: str = ""
    detailed: str = ""
    conditions: tuple[str, ...] = ()
    criteria_text: str = ""
    min_age: str = ""
    max_age: str = ""
    gender: str = ""

    @property
    def point_id(self) -> str:
        """
        Deterministic UUID derived from `nct_id`.
        Qdrant only accepts unsigned integers or UUIDs as point IDs, so the
        NCT id cannot be used directly: uuid5 hashes it. Re-indexing the same
        trial overwrites its point instead of adding a second copy.
        """
        return str(uuid.uuid5(NAMESPACE, self.nct_id))

    def to_document(self, detailed: bool = False) -> str:
        """
        The text that gets embedded. `detailed` is off by default: only 62% of
        the corpus has one, and its length skews BM25 length normalization while
        pushing the title out of the encoder's 512-token window. It is an
        ablation knob, not a default.
        """
        parts = [self.title, ", ".join(self.conditions), self.summary]
        if detailed:
            parts.append(self.detailed)
        return "\n\n".join(p for p in parts if p)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conditions"] = list(self.conditions)
        return d


@dataclass(frozen=True, slots=True)
class Criterion:
    nct_id: str
    kind: Kind
    text: str
    index: int
    source: Source = Source.DEFAULT

    @property
    def ref(self) -> str:
        return f"{self.nct_id}|{self.kind.value[:4]}{self.index}"

    def to_dict(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "kind": self.kind.value,
            "text": self.text,
            "index": self.index,
            "source": self.source.value,
            "ref": self.ref,
        }
