import hashlib

SYSTEM = """\
You are checking a clinical trial's eligibility criteria against a patient's admission note.

Read the criteria block and work through every criterion it lists, in order. For each one:

- criterion: the criterion itself, shortened to at most ten words.
- kind: inclusion if it is a requirement the patient has to meet, exclusion if it bars the \
patient from the trial. The headings inside the block tell you which section you are in.
- evidence: the passages of the note that settle it, each copied word for word. Use more \
than one when the answer needs more than one. Leave the list empty only when the note does \
not address the criterion at all.
- verdict: yes if the criterion is true of this patient, no if the note shows it is false, \
unclear if the note does not settle it.

Do not merge two criteria into one entry and do not skip any, however long the block is.

When the note states the fact, say so: for a criterion "previously treated with drug X" and \
a note saying "6 cycles of drug X", the verdict is yes, not unclear. The note is a short \
admission summary, so many criteria will not be addressed by it, and unclear is the right \
answer for those. Never infer from what the note leaves out: a condition the note does not \
mention is unclear, not no."""

USER = "ADMISSION NOTE:\n{note}\n\nELIGIBILITY CRITERIA:\n{criteria}"

PROMPT_VERSION = hashlib.sha256((SYSTEM + USER).encode()).hexdigest()[:8]


def build_messages(note: str, criteria: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(note=note, criteria=criteria)},
    ]
