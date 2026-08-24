"""German text normalizer for natural TTS reading.

Handles:
- abbreviations (z.B., usw., ca., d.h., bzw., Dr., Mr., Mrs., ...)
- numbers and years (1990 -> neunzehnhundertneunzig)
- currency (50 € -> fünfzig Euro)
- percentages (10 % -> zehn Prozent)
- punctuation pacing (ellipsis -> comma, spacing cleanup)
"""

import re

from num2words import num2words

_ABBREVIATIONS = [
    ("z.B.", "zum Beispiel"),
    ("u.a.", "unter anderem"),
    ("usw.", "und so weiter"),
    ("ca.", "zirka"),
    ("d.h.", "das heißt"),
    ("bzw.", "beziehungsweise"),
    ("etc.", "und so weiter"),
    ("Dr.", "Doktor"),
    ("Dr", "Doktor"),
    ("Mr.", "Mister"),
    ("Mrs.", "Misses"),
    ("Nr.", "Nummer"),
    ("std.", "Stunde"),
    ("min.", "Minute"),
    ("Tel.", "Telefon"),
]

_CURRENCY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:Euro|EUR|€)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(?<![\d\w])(\d{1,9})(?!\d)")
_ORDINAL_YEAR_RE = re.compile(r"(?<![\d\w])(\d{1,2})\.(?=\s)")


def _german_number(n):
    """Convert an integer to German words; 4-digit years 1000-1999 read year-style."""
    if 1000 <= n <= 1999:
        hundreds = n // 100
        rest = n % 100
        if rest == 0:
            return num2words(hundreds, lang="de") + "hundert"
        return num2words(hundreds, lang="de") + "hundert" + num2words(rest, lang="de")
    return num2words(n, lang="de")


def _german_decimal(s):
    if "," in s:
        whole, frac = s.split(",", 1)
    elif "." in s:
        whole, frac = s.split(".", 1)
    else:
        whole, frac = s, None
    if frac is None:
        return _german_number(int(s)), True
    return (_german_number(int(whole)) + " Komma "
            + " ".join(num2words(int(d), lang="de") for d in frac)), True


def _replace_currency(text):
    def repl(m):
        amount, done = _german_decimal(m.group(1))
        return amount + " Euro"
    return _CURRENCY_RE.sub(repl, text)


def _replace_percent(text):
    def repl(m):
        amount, done = _german_decimal(m.group(1))
        return amount + " Prozent"
    return _PERCENT_RE.sub(repl, text)


def _replace_abbreviations(text):
    for abbr, full in _ABBREVIATIONS:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        text = pattern.sub(full, text)
    return text


def _replace_numbers(text):
    def repl(m):
        return _german_number(int(m.group(1)))
    return _NUMBER_RE.sub(repl, text)


def _fix_pacing(text):
    text = text.replace("...", ", ")
    text = _ORDINAL_YEAR_RE.sub(r"\1", text)  # drop trailing period of plain ordinals
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=[\wÄÖÜäöüß])", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_german(text: str) -> str:
    if not text or not text.strip():
        return text
    text = text.replace("€", " €")
    text = _replace_currency(text)
    text = _replace_percent(text)
    text = _replace_abbreviations(text)
    text = _replace_numbers(text)
    text = _fix_pacing(text)
    return text