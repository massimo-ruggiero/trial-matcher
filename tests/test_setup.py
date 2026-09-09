import pytest
from qdrant_client import QdrantClient

from src.config import QDRANT_URL


@pytest.mark.integration
def test_qdrant_reachable():
    client = QdrantClient(url=QDRANT_URL)
    assert client.get_collections() is not None
