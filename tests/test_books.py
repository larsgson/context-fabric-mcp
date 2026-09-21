"""Tests for the canonical book helper.

The corpus-consistency tests need locally available corpus data, like
test_cf_engine.py; the rest are pure.
"""

import pytest

from context_fabric_mcp.books import (
    BOOKS,
    UnknownBookError,
    book_fields,
    book_name,
    codes_for_corpus,
    resolve_book,
    section_name,
    template_name,
)
from context_fabric_mcp.cf_engine import CFEngine


class TestTable:
    def test_counts(self):
        assert len(BOOKS) == 66
        assert len(codes_for_corpus("hebrew")) == 39
        assert len(codes_for_corpus("greek")) == 27

    def test_codes_are_three_uppercase_chars(self):
        for code in BOOKS:
            assert len(code) == 3 and code == code.upper()

    def test_every_book_has_an_english_name(self):
        for code in BOOKS:
            assert book_name(code)


class TestResolve:
    def test_round_trips_every_spelling(self):
        for code, book in BOOKS.items():
            for spelling in (
                code,
                code.lower(),
                book.section_name,
                book.template_name,
                book.names["en"],
                book.names["en"].upper(),
            ):
                assert resolve_book(spelling) == code, spelling

    @pytest.mark.parametrize(
        "value, code",
        [
            ("Psalms", "PSA"),
            ("Psalm", "PSA"),
            ("Psalmi", "PSA"),
            ("Deuteronomium", "DEU"),
            ("1_Samuel", "1SA"),
            ("1 Samuel", "1SA"),
            ("1Samuel", "1SA"),
            ("I Samuel", "1SA"),
            ("first samuel", "1SA"),
            ("Samuel_II", "2SA"),
            ("Song_of_songs", "SNG"),
            ("Song of Solomon", "SNG"),
            ("Canticum", "SNG"),
            ("  matthew ", "MAT"),
            ("Revelation", "REV"),
            ("JDE", "JUD"),
            ("Jude", "JUD"),
            ("Job", "JOB"),
            ("Iob", "JOB"),
        ],
    )
    def test_aliases(self, value, code):
        assert resolve_book(value) == code

    def test_isaiah_not_confused_with_roman_numeral(self):
        assert resolve_book("Isaiah") == "ISA"
        assert resolve_book("Jesaia") == "ISA"

    @pytest.mark.parametrize("value", ["", "   ", "Atlantis", "Psalmz1234"])
    def test_unknown_raises(self, value):
        with pytest.raises(UnknownBookError):
            resolve_book(value)

    def test_unknown_suggests_close_match(self):
        with pytest.raises(UnknownBookError, match="PSA"):
            resolve_book("Psalmz")

    def test_is_a_value_error(self):
        assert issubclass(UnknownBookError, ValueError)


class TestNames:
    def test_display_name_accepts_any_spelling(self):
        assert book_name("Psalmi") == "Psalms"
        assert book_name("1sa") == "1 Samuel"
        assert book_name("SNG") == "Song of Songs"

    def test_unavailable_language_raises(self):
        with pytest.raises(UnknownBookError, match="'xx'"):
            book_name("GEN", lang="xx")


class TestBookFields:
    def test_code_and_display_name(self):
        assert book_fields("Psalmi") == {"book": "PSA", "book_name": "Psalms"}
        assert book_fields("1_Samuel") == {"book": "1SA", "book_name": "1 Samuel"}

    def test_unavailable_language_raises(self):
        with pytest.raises(UnknownBookError):
            book_fields("GEN", lang="xx")


class TestCorpusNames:
    def test_hebrew_forms(self):
        assert section_name("PSA", "hebrew") == "Psalms"
        assert template_name("PSA", "hebrew") == "Psalmi"
        assert section_name("Psalmi", "hebrew") == "Psalms"

    def test_greek_forms(self):
        assert section_name("Matthew", "greek") == "MAT"
        assert template_name("MAT", "greek") == "MAT"

    def test_wrong_corpus_raises(self):
        with pytest.raises(UnknownBookError, match="not in the greek"):
            section_name("PSA", "greek")
        with pytest.raises(UnknownBookError, match="not in the hebrew"):
            template_name("MAT", "hebrew")


@pytest.fixture(scope="module")
def engine():
    return CFEngine()


@pytest.mark.parametrize("corpus", ["hebrew", "greek"])
class TestMatchesCorpus:
    """The table must agree with what the loaded corpus actually contains."""

    def test_book_list_and_order(self, engine: CFEngine, corpus: str):
        names = [b.name for b in engine.list_books(corpus)]
        assert [resolve_book(n) for n in names] == codes_for_corpus(corpus)

    def test_section_and_template_names(self, engine: CFEngine, corpus: str):
        api = engine._ensure_loaded(corpus)
        codes = codes_for_corpus(corpus)
        for node, code in zip(api.F.otype.s("book"), codes):
            assert api.T.sectionFromNode(node)[0] == section_name(code, corpus)
            assert api.F.book.v(node) == template_name(code, corpus)
            assert api.T.nodeFromSection((section_name(code, corpus),)) == node
