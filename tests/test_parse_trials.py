from pathlib import Path

from src.ingest.parse_trials import parse_trial

FIXTURES = Path(__file__).parent / "fixtures"


def test_extracts_core_fields():
    trial = parse_trial(FIXTURES / "dash.xml")
    assert trial is not None
    assert trial.nct_id == "NCT00000102"
    assert trial.title
    assert trial.conditions


def test_criteria_keeps_line_structure():
    trial = parse_trial(FIXTURES / "numbered.xml")
    assert "\n" in trial.criteria_text
    assert "Inclusion Criteria" in trial.criteria_text
    assert any(line.startswith("    ") for line in trial.criteria_text.splitlines())


def test_other_fields_are_collapsed():
    trial = parse_trial(FIXTURES / "dash.xml")
    assert "\n" not in trial.title
    assert "\n" not in trial.summary


def test_missing_fields_are_empty_strings():
    trial = parse_trial(FIXTURES / "dash.xml")
    for value in (trial.summary, trial.detailed, trial.min_age, trial.gender):
        assert isinstance(value, str)


def test_trial_without_criteria_is_still_parsed():
    trial = parse_trial(FIXTURES / "no_criteria.xml")
    assert trial is not None
    assert trial.nct_id == "NCT00005736"
    assert trial.criteria_text == ""
    assert trial.title


def test_returns_none_on_malformed_xml(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<clinical_study><unclosed>")
    assert parse_trial(bad) is None
