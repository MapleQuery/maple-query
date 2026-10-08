"""Name normalisation shared by every linker.

Sources spell one person many ways: "Hon. Pierre Poilievre",
"POILIEVRE, Pierre", "Pierre Poilievre, M.P.", "Poilièvre, Pierre J.".
`normalize` maps them all to "pierre poilievre": casefolded, accents
stripped, honorifics and post-nominals removed, "Last, First" flipped,
single-letter middle initials dropped. Matching is exact on this form;
nothing here guesses.
"""

from __future__ import annotations

import re
import unicodedata

_HONORIFICS = frozenset(
    {"hon", "honourable", "honorable", "the", "right", "rt", "mr", "mrs", "ms", "miss", "dr", "sir"}
    | {"mp", "pc", "qc", "kc", "ecc", "mla", "phd", "jr", "sr"}
)
_NON_WORD = re.compile(r"[^a-z0-9' -]+")
_SPACES = re.compile(r"\s+")
_RIGHT_QUOTE = chr(0x2019)  # typographic apostrophe, as in O'Regan


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.casefold()


def normalize(name: str) -> str:
    """Canonical 'given family' form, or '' when nothing is left."""
    text = _fold(name or "").replace(_RIGHT_QUOTE, "'")
    # Trailing post-nominals after a comma ("Pierre Poilievre, M.P., P.C.")
    # are a suffix, not a "Last, First" flip.
    parts = [p.strip() for p in text.split(",")]
    while len(parts) > 1 and all(w in _HONORIFICS for w in re.findall(r"[a-z]+", parts[-1].replace(".", ""))):
        parts.pop()
    # "Poilievre, Pierre" -> "Pierre Poilievre"
    text = f"{parts[1]} {parts[0]}" if len(parts) > 1 else parts[0]
    text = text.replace(".", " ").replace("-", " ")
    text = _NON_WORD.sub(" ", text)
    words = [w.strip("'") for w in _SPACES.split(text) if w.strip("'")]
    words = [w for w in words if w not in _HONORIFICS]
    # Drop single-letter middle initials, but never the first or last word.
    if len(words) > 2:
        words = [words[0], *[w for w in words[1:-1] if len(w) > 1], words[-1]]
    return " ".join(words)


def variants(name: str, given: str | None = None, family: str | None = None) -> set[str]:
    """Normalised forms a source might use for one person."""
    out = {normalize(name)}
    if given and family:
        out.add(normalize(f"{given} {family}"))
        # First given name only ("Mary Jane Smith" filed as "Smith, Mary").
        first = given.split()[0]
        out.add(normalize(f"{first} {family}"))
    out.discard("")
    return out
