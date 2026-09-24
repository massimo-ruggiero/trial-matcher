import hashlib
import json

from src.assess.schema import schema

SYSTEM = """\
You are checking a clinical trial's eligibility criteria against a patient's admission note.

Read the criteria block and work through every criterion it lists, in order. For each one:

- criterion: the criterion itself, shortened to at most ten words.
- kind: inclusion if it is a requirement the patient has to meet, exclusion if it bars the \
patient from the trial. The headings inside the block tell you which section you are in.
- evidence: the passages of the note that settle it, each copied character for character, \
keeping the note's own abbreviations, acronyms, casing, punctuation and symbols: where the \
note writes "Pt", quote "Pt", and where it writes "Patient", quote "Patient"; copy "-->" as \
"-->", never as an arrow. A passage that cannot be found in the note exactly as \
written is not evidence: it belongs in rationale. Use more than one when the answer needs \
more than one. Leave the list empty only when the note does not address the criterion at all.
- rationale: leave it empty whenever the passage you quoted settles the criterion by itself, \
which is most of the time. Write it only where reading that passage is not enough: the \
criterion names something the note calls by another name, or the answer follows from what the \
note says without being stated there. Write it also when the note has nothing to quote at all, \
to say what is missing. Never reason about a fact without quoting it - if the note contains \
the fact, it belongs in evidence - and never restate in your own words a passage you have \
already quoted.
- verdict: yes if the criterion is true of this patient, no if the note shows it is false, \
unclear if the note does not settle it. A note that describes one condition in detail says \
nothing about a different one: not mentioning it is unclear, never no.

Do not merge two criteria into one entry and do not skip any, however long the block is.

When the note states the fact, say so: for a criterion "previously treated with drug X" and \
a note saying "6 cycles of drug X", the verdict is yes, not unclear. The note is a short \
admission summary, so many criteria will not be addressed by it, and unclear is the right \
answer for those. Never infer from what the note leaves out: a condition the note does not \
mention is unclear, not no."""

USER = "ADMISSION NOTE:\n{note}\n\nELIGIBILITY CRITERIA:\n{criteria}"

# The schema is part of the request, so a field added there has to invalidate
# the cached verdicts exactly as a reworded instruction does.
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM + USER + json.dumps(schema(), sort_keys=True)).encode()
).hexdigest()[:8]


def build_messages(note: str, criteria: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(note=note, criteria=criteria)},
    ]
