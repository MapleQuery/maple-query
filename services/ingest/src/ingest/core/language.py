"""Resource language detection and dataset-level EN/FR pairing.

See PRD 2.2 §9 (algorithm) and parent §4.2 (English-by-default policy).
"""
from __future__ import annotations

import re
from typing import Literal

from ingest.clients.ckan import Dataset, Resource

Language = Literal["en", "fr", "unknown"]

_EN_CODES = frozenset({"en", "eng"})
_FR_CODES = frozenset({"fr", "fra"})

_FR_URL_RE = re.compile(r"(/fr/|/fr-|-fra?(\.|$|/))")
_EN_URL_RE = re.compile(r"(/en/|/en-|-eng?(\.|$|/))")


def detect_language(resource: Resource) -> Language:
    """Return the resource's language per PRD §9.1 precedence.

    English wins ties (per parent PRD §4.2): a resource with both `en`
    and `fr` in its declared list is treated as English.
    """
    declared = {code.lower() for code in resource.languages_declared}
    if declared & _EN_CODES:
        return "en"
    if declared & _FR_CODES:
        return "fr"

    for source in (resource.url, resource.name or ""):
        if _FR_URL_RE.search(source):
            return "fr"
        if _EN_URL_RE.search(source):
            return "en"

    return "unknown"


def filter_resources_by_pairing(dataset: Dataset) -> list[tuple[Resource, Language]]:
    """Apply dataset-level EN/FR pairing per PRD §9.2 + §9.3.

    - If the dataset has any English (or unknown-defaulting-to-English)
      resource: ingest those; skip every French resource.
    - If the dataset has only French resources: ingest them as net-new.

    The returned tuples carry the **detected** language (including
    `'unknown'`); we don't relabel — BQ stores what we actually detected.
    """
    detected = [(r, detect_language(r)) for r in dataset.resources]
    has_english_or_unknown = any(lang in ("en", "unknown") for _, lang in detected)

    if has_english_or_unknown:
        return [(r, lang) for r, lang in detected if lang in ("en", "unknown")]
    return [(r, lang) for r, lang in detected if lang == "fr"]
