from pathlib import Path

from src.ingest.parse_topics import parse

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_all_topics():
    assert len(parse(FIXTURES / "topics2021.xml")) == 75


def test_topic_ids_are_ints():
    topics = parse(FIXTURES / "topics2021.xml")
    assert topics[0].topic_id == 1
    assert all(isinstance(t.topic_id, int) for t in topics)


def test_whitespace_is_collapsed():
    for t in parse(FIXTURES / "topics2021.xml"):
        assert "\n" not in t.text
        assert "  " not in t.text
        assert t.text == t.text.strip()
