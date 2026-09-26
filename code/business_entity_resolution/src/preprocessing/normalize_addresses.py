"""
normalize_addresses.py — Address normalization.

Operations (in order):
1. Lowercase + unicode → ASCII
2. Expand common abbreviations (St→Street, Rd→Road, Blvd→Boulevard, etc.)
3. Strip punctuation; collapse whitespace
4. Reorder tokens alphabetically (word-order swap robustness)
   NOTE: we keep the original normalized form as the primary and also expose
   the sorted-token form for TF-IDF; callers decide which to use.
"""

from __future__ import annotations
import re
import unicodedata

_ABBREV_MAP: dict[str, str] = {
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\brd\b": "road",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfwy\b": "freeway",
    r"\bexpy\b": "expressway",
    r"\brt\b": "route",
    r"\bfl\b": "floor",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bunit\b": "unit",
    r"\bbldg\b": "building",
    r"\bpo\b": "post office",
    r"\bp\.o\.\b": "post office",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
    # India-specific
    r"\bnagar\b": "nagar",
    r"\bmarg\b": "marg",
    r"\bcolony\b": "colony",
    r"\bsociety\b": "society",
    # Generic noise
    r"\bno\b": "number",
    r"\bno\.\b": "number",
}

_COMPILED_ABBREV = [(re.compile(pat, re.IGNORECASE), repl)
                    for pat, repl in _ABBREV_MAP.items()]

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DIGIT_PREFIX_RE = re.compile(r"^\d+")  # leading house number


def _unicode_to_ascii(text: str) -> str:
    return (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def normalize_address(raw: str) -> str:
    """Return a canonical, lowercase, abbreviation-expanded address string."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    s = _unicode_to_ascii(s)
    for pat, repl in _COMPILED_ABBREV:
        s = pat.sub(repl, s)
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def normalize_address_sorted(raw: str) -> str:
    """Return token-sorted canonical address (robust to component reordering)."""
    s = normalize_address(raw)
    tokens = sorted(s.split())
    return " ".join(tokens)


def normalize_address_series(series: "pd.Series") -> "pd.Series":
    return series.fillna("").apply(normalize_address)


def normalize_address_sorted_series(series: "pd.Series") -> "pd.Series":
    return series.fillna("").apply(normalize_address_sorted)
