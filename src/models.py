import uuid
from dataclasses import asdict, dataclass

# Fixed namespace for deterministic UUID generation.
NAMESPACE = uuid.UUID("6f1e2a3c-4b5d-4e6f-8a9b-0c1d2e3f4a5b")


@dataclass(frozen=True, slots=True)
class LegalChunk:
    """
    Smallest immutable and indexable unit of a legal document, usually a single comma.
    """
    text: str
    urn: str                        # Uniform Resource Name
    article: str | None = None
    comma: str | None = None 
    heading: str | None = None
    date: str = ""                  # # ISO 8601, e.g. "2001-06-08"
    source_url: str = ""

    @property
    def ref(self) -> str:
        """
        Human-readable reference.
        """
        parts = [self.urn]
        if self.article:
            parts.append(f"art{self.article}")
        if self.comma:
            parts.append(f"co{self.comma}")
        return "|".join(parts)

    @property
    def point_id(self) -> str:
        """
        Deterministic UUID derived from `ref`.
        Qdrant only accepts unsigned integers or UUIDs as point IDs, so the
        reference string cannot be used directly. uuid5 hashes it instead.
        """
        return str(uuid.uuid5(NAMESPACE, self.ref))

    def to_payload(self) -> dict:
        """
        Flatten into the JSON metadata.
        """
        return {**asdict(self), "ref": self.ref}