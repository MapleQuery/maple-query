"""Resource language detection and dataset-level EN/FR pairing.

English is ingested by default. French is ingested **only** when it's
net-new — i.e. the CKAN dataset has no English (or unknown-defaulting-
to-English) sibling resource. Bilingual mirrors aren't double-stored.
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
    """Return the resource's language.

    Order of precedence (first hit wins):
      1. `resource.languages_declared` — list of CKAN language tags.
         `en`/`eng` → 'en'; `fr`/`fra` → 'fr'. English wins ties, so a
         resource tagged `['en', 'fr']` is treated as English.
      2. URL suffix sniff: matches `-fra?(.|$|/)` or `/fr/` or `/fr-` → 'fr';
         `-eng?(.|$|/)` or `/en/` or `/en-` → 'en'.
      3. Filename (`resource.name`) — same suffix sniff.
      4. Otherwise 'unknown'.
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
    """Apply the dataset-level EN/FR pairing rule.

    - If the dataset has any English (or unknown-defaulting-to-English)
      resource: ingest those; skip every French resource.
    - If the dataset has only French resources: ingest them as net-new.

    Returned tuples carry the **detected** language (including
    `'unknown'`); we don't relabel — downstream sees what was actually
    detected.
    """
    detected = [(r, detect_language(r)) for r in dataset.resources]
    has_english_or_unknown = any(lang in ("en", "unknown") for _, lang in detected)

    if has_english_or_unknown:
        return [(r, lang) for r, lang in detected if lang in ("en", "unknown")]
    return [(r, lang) for r, lang in detected if lang == "fr"]
