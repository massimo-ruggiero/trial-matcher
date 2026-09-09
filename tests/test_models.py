from src.models import Criterion, Kind, Trial


def test_point_id_depends_only_on_nct_id():
    a = Trial(nct_id="NCT00000102", title="one", summary="foo")
    b = Trial(nct_id="NCT00000102", title="another", summary="bar")
    # Re-indexing an updated trial must overwrite its point, not duplicate it.
    assert a.point_id == b.point_id


def test_point_id_is_stable_across_runs():
    # Pinned on purpose: if this value changes, the namespace changed and every
    # point already in Qdrant is orphaned.
    trial = Trial(nct_id="NCT00000102", title="one")
    assert trial.point_id == "a018741b-43d2-5c4c-92ac-4975bca4ba07"


def test_distinct_trials_get_distinct_ids():
    a = Trial(nct_id="NCT00000102", title="one")
    b = Trial(nct_id="NCT00000104", title="one")
    assert a.point_id != b.point_id


def test_document_excludes_detailed_description():
    trial = Trial(
        nct_id="NCT1",
        title="title",
        summary="summary",
        detailed="detailed",
        conditions=("a", "b"),
    )
    doc = trial.to_document()
    assert "detailed" not in doc
    assert doc == "title\n\na, b\n\nsummary"


def test_document_includes_detailed_when_asked():
    trial = Trial(nct_id="NCT1", title="title", summary="summary", detailed="detailed")
    assert "detailed" in trial.to_document(detailed=True)


def test_criterion_ref():
    c = Criterion(nct_id="NCT1", kind=Kind.EXCLUSION, text="...", index=3)
    assert c.ref == "NCT1|excl3"
