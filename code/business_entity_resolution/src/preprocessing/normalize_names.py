"""
normalize_names.py — Business name normalization.

Operations (in order):
1. Lowercase
2. Strip leading/trailing whitespace
3. Replace '&' → 'and'
4. Strip punctuation except hyphens inside words
5. Expand common legal suffixes to canonical form
6. Collapse whitespace
"""

from __future__ import annotations
import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal-suffix canonicalization map
# All values are the canonical (full) form.
# ---------------------------------------------------------------------------
_LEGAL_SUFFIX_MAP: dict[str, str] = {
    # Corporation variants
    r"\bcorp\b": "corporation",
    r"\bcorporated\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bincorporated\b": "incorporated",
    # Limited / LLC / LLP variants
    r"\bltd\b": "limited",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\blp\b": "limited partnership",
    # Private variants
    r"\bpvt\b": "private",
    r"\bprivate\b": "private",
    r"\bpte\b": "private",           # Singapore-style
    # Public variants
    r"\bplc\b": "public limited company",
    # Generic
    r"\bco\b": "company",
    r"\bco\.\b": "company",
    r"\bcompany\b": "company",
    r"\bgroup\b": "group",
    r"\bholdings\b": "holdings",
    r"\benterprises\b": "enterprises",
    r"\benterprise\b": "enterprise",
    r"\bindustries\b": "industries",
    r"\bindustry\b": "industry",
    r"\bservices\b": "services",
    r"\bservice\b": "service",
    r"\bsolutions\b": "solutions",
    r"\bsolution\b": "solution",
    r"\btechnologies\b": "technologies",
    r"\btechnology\b": "technology",
    r"\btech\b": "technology",
    r"\bsystems\b": "systems",
    r"\bsystem\b": "system",
    r"\binternationals\b": "international",
    r"\binternational\b": "international",
    r"\bintl\b": "international",
    r"\bintl\.\b": "international",
    r"\bnational\b": "national",
    r"\busa\b": "usa",
    r"\bus\b": "us",
}

_COMPILED_SUFFIX = [(re.compile(pat, re.IGNORECASE), repl)
                    for pat, repl in _LEGAL_SUFFIX_MAP.items()]

# Punctuation to strip (keep hyphens between word chars handled separately)
_PUNCT_RE = re.compile(r"[^\w\s-]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_AMP_RE = re.compile(r"&")
_DBA_RE = re.compile(r"\bdba\b.*", re.IGNORECASE)   # strip "dba ..." tails


def _unicode_to_ascii(text: str) -> str:
    """Transliterate accented characters (e.g. é → e) for fuzzy matching."""
    return (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def normalize_name(raw: str) -> str:
    """Return a canonical, lowercase, normalized business name string."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    s = _unicode_to_ascii(s)
    s = _AMP_RE.sub(" and ", s)
    s = _DBA_RE.sub("", s)          # drop DBA alias tail
    s = _PUNCT_RE.sub(" ", s)       # strip punctuation
    for pat, repl in _COMPILED_SUFFIX:
        s = pat.sub(repl, s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def normalize_name_series(series: "pd.Series") -> "pd.Series":
    """Vectorized wrapper for use on a DataFrame column."""
    return series.fillna("").apply(normalize_name)
