"""Canonical book identifiers.

Internally every book is identified by its USFM/Paratext three-letter code
(GEN, PSA, 1SA, MAT, ...). The corpora name books differently:

- BHSA (hebrew): the raw ``book`` feature is Latin (Psalmi, Samuel_I), which is
  what search templates match, while Text-Fabric's section API
  (``T.nodeFromSection`` / ``T.sectionFromNode``) uses English (Psalms, 1_Samuel).
- Nestle 1904 (greek): already USFM codes in both places.

``resolve_book`` turns any of these spellings (plus common English variants)
into the code; ``section_name`` and ``template_name`` turn a code back into the
corpus-native form. Display names live in ``book_name`` keyed by language so
other languages can be added without touching callers.
"""

import difflib
import re
from dataclasses import dataclass

DEFAULT_LANG = "en"


class UnknownBookError(ValueError):
    """Raised when a book reference cannot be resolved to a known book."""


@dataclass(frozen=True)
class Book:
    code: str
    corpus: str
    names: dict[str, str]  # display name per language
    section_name: str  # as used by Text-Fabric's section API
    template_name: str  # value of the raw `book` feature (search templates)


# (code, English display name, BHSA section name, BHSA `book` feature value)
_HEBREW = [
    ("GEN", "Genesis", "Genesis", "Genesis"),
    ("EXO", "Exodus", "Exodus", "Exodus"),
    ("LEV", "Leviticus", "Leviticus", "Leviticus"),
    ("NUM", "Numbers", "Numbers", "Numeri"),
    ("DEU", "Deuteronomy", "Deuteronomy", "Deuteronomium"),
    ("JOS", "Joshua", "Joshua", "Josua"),
    ("JDG", "Judges", "Judges", "Judices"),
    ("1SA", "1 Samuel", "1_Samuel", "Samuel_I"),
    ("2SA", "2 Samuel", "2_Samuel", "Samuel_II"),
    ("1KI", "1 Kings", "1_Kings", "Reges_I"),
    ("2KI", "2 Kings", "2_Kings", "Reges_II"),
    ("ISA", "Isaiah", "Isaiah", "Jesaia"),
    ("JER", "Jeremiah", "Jeremiah", "Jeremia"),
    ("EZK", "Ezekiel", "Ezekiel", "Ezechiel"),
    ("HOS", "Hosea", "Hosea", "Hosea"),
    ("JOL", "Joel", "Joel", "Joel"),
    ("AMO", "Amos", "Amos", "Amos"),
    ("OBA", "Obadiah", "Obadiah", "Obadia"),
    ("JON", "Jonah", "Jonah", "Jona"),
    ("MIC", "Micah", "Micah", "Micha"),
    ("NAM", "Nahum", "Nahum", "Nahum"),
    ("HAB", "Habakkuk", "Habakkuk", "Habakuk"),
    ("ZEP", "Zephaniah", "Zephaniah", "Zephania"),
    ("HAG", "Haggai", "Haggai", "Haggai"),
    ("ZEC", "Zechariah", "Zechariah", "Sacharia"),
    ("MAL", "Malachi", "Malachi", "Maleachi"),
    ("PSA", "Psalms", "Psalms", "Psalmi"),
    ("JOB", "Job", "Job", "Iob"),
    ("PRO", "Proverbs", "Proverbs", "Proverbia"),
    ("RUT", "Ruth", "Ruth", "Ruth"),
    ("SNG", "Song of Songs", "Song_of_songs", "Canticum"),
    ("ECC", "Ecclesiastes", "Ecclesiastes", "Ecclesiastes"),
    ("LAM", "Lamentations", "Lamentations", "Threni"),
    ("EST", "Esther", "Esther", "Esther"),
    ("DAN", "Daniel", "Daniel", "Daniel"),
    ("EZR", "Ezra", "Ezra", "Esra"),
    ("NEH", "Nehemiah", "Nehemiah", "Nehemia"),
    ("1CH", "1 Chronicles", "1_Chronicles", "Chronica_I"),
    ("2CH", "2 Chronicles", "2_Chronicles", "Chronica_II"),
]

# (code, English display name); the corpus uses the code everywhere
_GREEK = [
    ("MAT", "Matthew"),
    ("MRK", "Mark"),
    ("LUK", "Luke"),
    ("JHN", "John"),
    ("ACT", "Acts"),
    ("ROM", "Romans"),
    ("1CO", "1 Corinthians"),
    ("2CO", "2 Corinthians"),
    ("GAL", "Galatians"),
    ("EPH", "Ephesians"),
    ("PHP", "Philippians"),
    ("COL", "Colossians"),
    ("1TH", "1 Thessalonians"),
    ("2TH", "2 Thessalonians"),
    ("1TI", "1 Timothy"),
    ("2TI", "2 Timothy"),
    ("TIT", "Titus"),
    ("PHM", "Philemon"),
    ("HEB", "Hebrews"),
    ("JAS", "James"),
    ("1PE", "1 Peter"),
    ("2PE", "2 Peter"),
    ("1JN", "1 John"),
    ("2JN", "2 John"),
    ("3JN", "3 John"),
    ("JUD", "Jude"),
    ("REV", "Revelation"),
]

# Extra spellings that are not a code, display name, section name or Latin name.
_EXTRA_ALIASES = {
    "PSA": ["Psalm"],
    "SNG": ["Song of Solomon", "Canticles", "Canticle of Canticles"],
    "ECC": ["Qoheleth"],
    "ACT": ["Acts of the Apostles"],
    "REV": ["Revelations", "Revelation of John", "Apocalypse"],
    "JUD": ["JDE"],  # older versions of the system prompt used JDE
}

BOOKS: dict[str, Book] = {}
for _code, _en, _section, _latin in _HEBREW:
    BOOKS[_code] = Book(_code, "hebrew", {"en": _en}, _section, _latin)
for _code, _en in _GREEK:
    BOOKS[_code] = Book(_code, "greek", {"en": _en}, _code, _code)


_ORDINALS = {
    "first": "1", "1st": "1", "i": "1",
    "second": "2", "2nd": "2", "ii": "2",
    "third": "3", "3rd": "3", "iii": "3",
}  # fmt: skip
_ORDINAL_RE = re.compile(r"^(first|second|third|1st|2nd|3rd|iii|ii|i)\s+(?=\S)")
_DIGIT_RE = re.compile(r"^([123])(?=[a-z])")


def _norm(value: str) -> str:
    """Normalise a book spelling: '1_Samuel', 'I Samuel', '1sam' -> '1 samuel'."""
    s = re.sub(r"[\s_.\-]+", " ", value.casefold()).strip()
    s = _ORDINAL_RE.sub(lambda m: _ORDINALS[m.group(1)] + " ", s)
    s = _DIGIT_RE.sub(lambda m: m.group(1) + " ", s)
    return s


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}

    def add(spelling: str, code: str) -> None:
        key = _norm(spelling)
        if aliases.setdefault(key, code) != code:
            raise RuntimeError(
                f"Book alias {spelling!r} is ambiguous: {aliases[key]} and {code}"
            )

    for book in BOOKS.values():
        add(book.code, book.code)
        add(book.section_name, book.code)
        add(book.template_name, book.code)
        for name in book.names.values():
            add(name, book.code)
    for code, extras in _EXTRA_ALIASES.items():
        for extra in extras:
            add(extra, code)
    return aliases


_ALIASES = _build_aliases()


def resolve_book(value: str) -> str:
    """Return the USFM code for a code, English name or Latin (BHSA) name.

    Case-insensitive; tolerates spaces/underscores/dots and ordinals
    ('1 Samuel', '1_Samuel', 'I Samuel', 'first samuel').
    """
    if not isinstance(value, str) or not value.strip():
        raise UnknownBookError("Book must be a non-empty string")
    key = _norm(value)
    code = _ALIASES.get(key)
    if code is None:
        close = difflib.get_close_matches(key, _ALIASES, n=3, cutoff=0.7)
        hint = ""
        if close:
            hint = " Did you mean " + ", ".join(
                dict.fromkeys(_ALIASES[c] for c in close)
            ) + "?"
        raise UnknownBookError(f"Unknown book {value!r}.{hint}")
    return code


def get_book(value: str) -> Book:
    """Resolve any accepted spelling to its Book record."""
    return BOOKS[resolve_book(value)]


def book_name(value: str, lang: str = DEFAULT_LANG) -> str:
    """Display name of a book in the given language (only 'en' for now)."""
    book = get_book(value)
    try:
        return book.names[lang]
    except KeyError:
        raise UnknownBookError(
            f"No {lang!r} name for {book.code}; available: {sorted(book.names)}"
        ) from None


def book_fields(value: str, lang: str = DEFAULT_LANG) -> dict[str, str]:
    """Response fields for a book: its code and its display name in ``lang``."""
    book = get_book(value)
    return {"book": book.code, "book_name": book_name(book.code, lang)}


def _in_corpus(value: str, corpus: str) -> Book:
    book = get_book(value)
    if book.corpus != corpus:
        raise UnknownBookError(f"{book.code} is not in the {corpus} corpus")
    return book


def section_name(value: str, corpus: str) -> str:
    """Book name as Text-Fabric's section API expects it (nodeFromSection)."""
    return _in_corpus(value, corpus).section_name


def template_name(value: str, corpus: str) -> str:
    """Book value for search templates: ``book book=<value>``."""
    return _in_corpus(value, corpus).template_name


def codes_for_corpus(corpus: str) -> list[str]:
    """USFM codes of the books in a corpus, in corpus order."""
    return [b.code for b in BOOKS.values() if b.corpus == corpus]
