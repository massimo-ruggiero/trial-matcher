import re

from src.assess.prompt import PROMPT_VERSION, SYSTEM, build_messages

NOTE = "45-year-old man with type 2 diabetes on metformin. No history of cancer."
BLOCK = """
        Inclusion Criteria:
          -  Age 18 or older
        Exclusion Criteria:
          -  History of cancer
"""


def test_the_block_is_passed_as_it_is():
    user = build_messages(NOTE, BLOCK)[1]["content"]
    assert BLOCK in user
    assert NOTE in user


def test_instructions_and_data_are_separate_messages():
    messages = build_messages(NOTE, BLOCK)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM


def test_the_worked_example_is_invented():
    # It teaches the behaviour without carrying wording from a real topic,
    # which would tune the prompt on the data we then measure on.
    assert "drug X" in SYSTEM
    assert "temozolomide" not in SYSTEM


def test_version_is_a_short_hash():
    assert re.fullmatch(r"[0-9a-f]{8}", PROMPT_VERSION)
