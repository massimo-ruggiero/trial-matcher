from src.assess.judge import expand, is_grounded, judge, load_shortlist

NOTE = "Patient is a 45-year-old man. Therapy included 11 cycles of temozolomide."


def test_quote_must_appear_in_the_note():
    assert is_grounded(["11 cycles of temozolomide"], NOTE)
    assert not is_grounded(["12 cycles of temozolomide"], NOTE)


def test_whitespace_and_case_do_not_matter():
    assert is_grounded(["patient is a   45-YEAR-OLD man"], NOTE)


def test_two_passages_are_checked_one_by_one():
    # One criterion can need two separate sentences; each has to be real.
    assert is_grounded(["45-year-old man", "11 cycles of temozolomide"], NOTE)
    assert not is_grounded(["45-year-old man", "20 cycles of temozolomide"], NOTE)


def test_no_quote_is_not_grounded():
    assert not is_grounded([], NOTE)
    assert not is_grounded(["   "], NOTE)


def test_shortlist_keeps_rank_order_and_cuts_at_depth(tmp_path):
    run = tmp_path / "run.txt"
    run.write_text("1 Q0 NCTa 1 3 x\n1 Q0 NCTb 2 2 x\n1 Q0 NCTc 3 1 x\n2 Q0 NCTd 1 3 x\n")
    assert load_shortlist(run, depth=2) == {"1": ["NCTa", "NCTb"], "2": ["NCTd"]}


def test_topic_ranges_are_expanded():
    assert expand("1-3,11") == {"1", "2", "3", "11"}
    assert expand("11") == {"11"}


def test_trial_without_criteria_is_recorded_not_sent():
    record = judge(NOTE, "   ", "NCT1", model="unused")
    assert record["error"] == "no criteria"
    assert record["criteria"] == []
