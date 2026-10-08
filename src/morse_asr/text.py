"""The original 44-symbol Cyrillic/digit vocabulary and CTC blank."""

SYMBOLS = "ЯЮЭЬЫЪЩШЧЦХФУТСРПОНМЛКЙИЗЖЕДГВБА9876543210# "
BLANK = len(SYMBOLS)


def encode(text):
    if not text:
        raise ValueError("Transcripts must not be empty")
    unknown = set(text) - set(SYMBOLS)
    if unknown:
        raise ValueError(f"Unsupported characters: {sorted(unknown)}")
    return [SYMBOLS.index(c) for c in text]


def collapse_ctc(path):
    """Collapse adjacent repeats BEFORE removing blanks: A, blank, A -> AA."""
    result, previous = [], None
    for index in path:
        index = int(index)
        if not 0 <= index <= BLANK:
            raise ValueError("Token outside vocabulary")
        if index != previous and index != BLANK:
            result.append(SYMBOLS[index])
        previous = index
    return "".join(result)


def decode(logits, lengths):
    paths = logits.argmax(-1).detach().cpu()
    return [collapse_ctc(p[: int(n)]) for p, n in zip(paths, lengths)]


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def metrics(references, hypotheses):
    if not references or len(references) != len(hypotheses) or any(not x for x in references):
        raise ValueError("Expected paired, nonempty reference transcripts")
    errors = [edit_distance(r, h) for r, h in zip(references, hypotheses)]
    return {
        "examples": len(references),
        "cer": sum(errors) / sum(map(len, references)),
        "mean_utterance_cer": sum(e / len(r) for e, r in zip(errors, references)) / len(errors),
        "exact_match": sum(r == h for r, h in zip(references, hypotheses)) / len(errors),
    }
