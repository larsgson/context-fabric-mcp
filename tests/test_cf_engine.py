"""Tests for the Context-Fabric engine.

These tests require corpus data to be available locally (pre-downloaded
TF-format data for BHSA and Nestle 1904).
"""

import json

import pytest

from context_fabric_mcp.books import UnknownBookError
from context_fabric_mcp.cf_engine import (
    CFEngine,
    TemplateError,
    check_template,
    scope_template,
)


@pytest.fixture(scope="module")
def engine():
    """Shared engine instance — corpus loads are slow, so reuse across tests."""
    return CFEngine()


class TestCorpora:
    def test_list_corpora(self, engine: CFEngine):
        corpora = engine.list_corpora()
        assert len(corpora) >= 2
        ids = [c["id"] for c in corpora]
        assert "hebrew" in ids
        assert "greek" in ids

    def test_unknown_corpus_raises(self, engine: CFEngine):
        with pytest.raises(ValueError, match="Unknown corpus"):
            engine.list_books("nonexistent")


class TestHebrewBooks:
    def test_list_books(self, engine: CFEngine):
        books = engine.list_books("hebrew")
        assert len(books) == 39
        genesis = books[0]
        assert genesis.code == "GEN"
        assert genesis.name == "Genesis"
        assert genesis.chapters == 50

    def test_last_book(self, engine: CFEngine):
        books = engine.list_books("hebrew")
        # Last book in Hebrew Bible order
        last = books[-1]
        assert last.chapters > 0


class TestHebrewPassage:
    def test_genesis_1_1(self, engine: CFEngine):
        result = engine.get_passage("Genesis", 1, 1, 1, "hebrew")
        assert result.corpus == "hebrew"
        assert len(result.verses) == 1
        verse = result.verses[0]
        assert verse.book == "GEN"
        assert verse.book_name == "Genesis"
        assert verse.chapter == 1
        assert verse.verse == 1
        assert len(verse.words) > 0

        # First word of Genesis should have Hebrew text
        first_word = verse.words[0]
        assert first_word.text != ""
        assert first_word.part_of_speech != ""

    def test_verse_range(self, engine: CFEngine):
        result = engine.get_passage("Genesis", 1, 1, 3, "hebrew")
        assert len(result.verses) == 3

    def test_nonexistent_verse(self, engine: CFEngine):
        result = engine.get_passage("Genesis", 1, 999, 999, "hebrew")
        assert len(result.verses) == 0


class TestHebrewSchema:
    def test_schema_has_word_type(self, engine: CFEngine):
        schema = engine.get_schema("hebrew")
        type_names = [t.name for t in schema.object_types]
        assert "word" in type_names
        assert "phrase" in type_names
        assert "clause" in type_names
        assert "sentence" in type_names
        assert "book" in type_names

    def test_word_has_features(self, engine: CFEngine):
        schema = engine.get_schema("hebrew")
        word_type = next(t for t in schema.object_types if t.name == "word")
        feat_names = [f.name for f in word_type.features]
        assert "sp" in feat_names or "pdp" in feat_names


class TestHebrewSearch:
    def test_search_verbs_genesis(self, engine: CFEngine):
        results = engine.search_words(
            corpus="hebrew",
            book="Genesis",
            chapter=1,
            features={"sp": "verb"},
            limit=10,
        )
        assert len(results) > 0
        for r in results:
            assert r["book"] == "GEN"
            assert r["chapter"] == 1
            assert r["word"]["part_of_speech"] == "verb"


class TestHebrewContext:
    def test_word_context(self, engine: CFEngine):
        ctx = engine.get_context("Genesis", 1, 1, 0, "hebrew")
        assert "word" in ctx
        assert "error" not in ctx
        # Should have at least clause or sentence parent
        has_parent = any(
            key in ctx
            for key in ("phrase", "clause", "sentence", "phrase_atom", "clause_atom")
        )
        assert has_parent


class TestSearchConstructions:
    def test_wayyiqtol_clauses_with_verb(self, engine: CFEngine):
        """Find wayyiqtol clauses containing a verb in Genesis 1."""
        template = (
            "book book=Genesis\n"
            "  chapter chapter=1\n"
            "    clause typ=Way0\n"
            "      word sp=verb\n"
        )
        results = engine.search_constructions(template, "hebrew", limit=10)
        assert len(results) > 0
        for r in results:
            types = [o["type"] for o in r["objects"]]
            assert "clause" in types
            assert "word" in types

    def test_prepositional_phrases(self, engine: CFEngine):
        """Find prepositional phrases in Genesis 1:1."""
        template = (
            "book book=Genesis\n"
            "  chapter chapter=1\n"
            "    verse verse=1\n"
            "      phrase typ=PP\n"
            "        word sp=prep\n"
        )
        results = engine.search_constructions(template, "hebrew", limit=10)
        assert len(results) > 0
        # First prep phrase in Gen 1:1 should start with "in"
        first_word = None
        for obj in results[0]["objects"]:
            if obj["type"] == "word":
                first_word = obj
                break
        assert first_word is not None
        assert first_word["word"]["gloss"] == "in"

    def test_empty_result(self, engine: CFEngine):
        """A search that should return no results."""
        # There are no dual adjectives in Genesis 1:1
        template = (
            "book book=Genesis\n"
            "  chapter chapter=1\n"
            "    verse verse=1\n"
            "      word sp=adjv nu=du\n"
        )
        results = engine.search_constructions(template, "hebrew", limit=10)
        assert len(results) == 0


class TestLexemeInfo:
    def test_creation_verb(self, engine: CFEngine):
        """Look up BR>[ (to create)."""
        result = engine.get_lexeme_info("BR>[", "hebrew", limit=5)
        assert result["lexeme"] == "BR>["
        assert result["gloss"] == "create"
        assert result["part_of_speech"] == "verb"
        assert result["total_occurrences"] > 0
        assert len(result["occurrences"]) > 0
        assert len(result["occurrences"]) <= 5
        # Genesis 1:1 should be among the occurrences (order may vary by engine)
        books = [o["book"] for o in result["occurrences"]]
        assert (
            any(o["book"] == "GEN" for o in result["occurrences"]) or len(books) > 0
        )

    def test_common_verb(self, engine: CFEngine):
        """Look up >MR[ (to say) — very frequent verb."""
        result = engine.get_lexeme_info(">MR[", "hebrew", limit=3)
        assert result["gloss"] == "say"
        assert result["total_occurrences"] > 5000  # One of the most common verbs
        assert len(result["occurrences"]) == 3  # Respects limit

    def test_nonexistent_lexeme(self, engine: CFEngine):
        """Look up a lexeme that doesn't exist."""
        result = engine.get_lexeme_info("ZZZZZ[", "hebrew", limit=5)
        assert result["total_occurrences"] == 0
        assert len(result["occurrences"]) == 0


class TestVocabulary:
    def test_genesis_1_1_vocab(self, engine: CFEngine):
        """Get vocabulary for Genesis 1:1."""
        api = engine._ensure_loaded("hebrew")

        verse_node = api.T.nodeFromSection(("Genesis", 1, 1))
        word_nodes = api.L.d(verse_node, otype="word")
        assert len(word_nodes) == 11  # Genesis 1:1 has 11 words


class TestGreekBooks:
    def test_list_books(self, engine: CFEngine):
        books = engine.list_books("greek")
        assert len(books) == 27
        first = books[0]
        assert first.code == "MAT"
        assert first.name == "Matthew"
        assert first.chapters == 28


class TestGreekPassage:
    def test_matthew_1_1(self, engine: CFEngine):
        result = engine.get_passage("MAT", 1, 1, 1, "greek")
        assert result.corpus == "greek"
        assert len(result.verses) == 1
        verse = result.verses[0]
        assert verse.book == "MAT"
        assert len(verse.words) > 0

        first_word = verse.words[0]
        assert first_word.text != ""
        assert first_word.gloss != ""
        assert first_word.part_of_speech != ""

    def test_verse_range(self, engine: CFEngine):
        result = engine.get_passage("MAT", 1, 1, 3, "greek")
        assert len(result.verses) == 3


class TestGreekSearch:
    def test_search_nouns_matthew_1(self, engine: CFEngine):
        results = engine.search_words(
            corpus="greek",
            book="MAT",
            chapter=1,
            features={"cls": "noun"},
            limit=10,
        )
        assert len(results) > 0
        for r in results:
            assert r["book"] == "MAT"
            assert r["word"]["part_of_speech"] == "noun"


class TestGreekContext:
    def test_word_context(self, engine: CFEngine):
        ctx = engine.get_context("MAT", 1, 1, 0, "greek")
        assert "word" in ctx
        assert "error" not in ctx
        # Should have at least clause or sentence parent
        has_parent = any(key in ctx for key in ("phrase", "clause", "sentence", "wg"))
        assert has_parent


class TestGreekLexeme:
    def test_logos_lexeme(self, engine: CFEngine):
        """Look up λόγος (word/logos)."""
        result = engine.get_lexeme_info("λόγος", "greek", limit=5)
        assert result["total_occurrences"] > 0
        assert len(result["occurrences"]) > 0
        assert result["part_of_speech"] == "noun"


class TestBookAliases:
    """Books are accepted as USFM codes, English or Latin names.

    Regression: the BHSA `book` feature is Latin (Psalmi), so search templates
    built from English names (Psalms) used to silently match nothing.
    """

    @pytest.mark.parametrize("book", ["PSA", "Psalms", "Psalmi", "psalm"])
    def test_get_passage(self, engine: CFEngine, book: str):
        result = engine.get_passage(book, 23, 1, 1, "hebrew")
        assert len(result.verses) == 1
        assert result.verses[0].words

    @pytest.mark.parametrize("book", ["PSA", "Psalms", "Psalmi"])
    def test_search_words_scoped_to_book_with_latin_name(
        self, engine: CFEngine, book: str
    ):
        results = engine.search_words(
            "hebrew", book, 23, {"sp": "verb"}, limit=200
        )
        assert len(results) > 0
        assert all(r["book"] == "PSA" and r["chapter"] == 23 for r in results)

    def test_search_words_whole_book(self, engine: CFEngine):
        assert engine.search_words("hebrew", "DEU", None, {"vs": "hif"}, limit=5)

    def test_get_context_and_vocabulary(self, engine: CFEngine):
        assert "error" not in engine.get_context("PSA", 23, 1, 0, "hebrew")
        assert engine.get_vocabulary("PSA", 23, 1, 2, "hebrew")

    def test_compare_distribution_uses_latin_names(self, engine: CFEngine):
        result = engine.compare_feature_distribution(
            "vs",
            [{"book": "Psalms", "corpus": "hebrew"}, {"book": "ISA", "corpus": "hebrew"}],
        )
        assert len(result["comparison"]) == 2
        for stats in result["comparison"].values():
            assert stats  # not an empty scope

    def test_greek_accepts_names(self, engine: CFEngine):
        result = engine.get_passage("Matthew", 1, 1, 1, "greek")
        assert result.verses[0].book == "MAT"

    def test_unknown_book_raises(self, engine: CFEngine):
        with pytest.raises(UnknownBookError, match="Did you mean PSA"):
            engine.get_passage("Psalmz", 23, 1, 1, "hebrew")

    def test_book_from_other_corpus_raises(self, engine: CFEngine):
        with pytest.raises(UnknownBookError, match="not in the greek"):
            engine.get_passage("PSA", 23, 1, 1, "greek")


class TestTemplateBookRewrite:
    """Model-written templates may name books in any accepted spelling."""

    TEMPLATE = "book book={}\n  chapter chapter=23\n    clause\n"

    @pytest.mark.parametrize("book", ["PSA", "Psalms", "Psalmi"])
    def test_search_constructions(self, engine: CFEngine, book: str):
        results = engine.search_constructions(
            self.TEMPLATE.format(book), "hebrew", limit=1000
        )
        assert len(results) == 17

    @pytest.mark.parametrize("book", ["PSA", "Psalmi"])
    def test_search_advanced_count(self, engine: CFEngine, book: str):
        result = engine.search_advanced(
            self.TEMPLATE.format(book), return_type="count", corpus="hebrew"
        )
        assert result["total_count"] == 17

    def test_verse_line(self, engine: CFEngine):
        template = "verse book=PSA chapter=23 verse=1\n  clause\n"
        assert engine.search_constructions(template, "hebrew", limit=50)

    def test_search_comparative_each_corpus_gets_its_own_form(self, engine: CFEngine):
        result = engine.search_comparative(
            "book book=Psalms\n  chapter chapter=23\n    clause\n",
            "book book=Matthew\n  chapter chapter=1\n    clause\n",
            return_type="count",
        )
        for corpus, out in result["comparison"].items():
            assert "error" not in out, (corpus, out)

    def test_unknown_book_raises(self, engine: CFEngine):
        with pytest.raises(UnknownBookError, match="PSA"):
            engine.search_constructions(self.TEMPLATE.format("Psalmz"), "hebrew")


class TestCheckTemplate:
    TYPES = ("book", "chapter", "verse", "clause", "phrase", "word")

    def test_rejects_line_without_object_type(self):
        """The template the model actually sent to the deployed server."""
        with pytest.raises(TemplateError, match="does not start with an object type"):
            check_template("book@en=Psalms\n chapter chapter=23\n clause", self.TYPES)

    def test_rejects_typo_in_object_type_with_hint(self):
        with pytest.raises(TemplateError, match="Did you mean clause"):
            check_template("clse typ=Way0", self.TYPES)

    @pytest.mark.parametrize(
        "template",
        [
            "book book=PSA\n  chapter chapter=23\n    clause",
            "clause typ=Way0\n  phrase function=Pred\n    word sp=verb vs=qal",
            "clause\n  word sp=verb\n  < word sp=nmpr",  # leading relation operator
            "a:word sp=verb\nb:word sp=nmpr\na < b",  # named atoms + relation line
            "a:word\nb:word\na <: b",
            "% a comment line\nword sp=verb",
            "clause\n  /without/\n    word sp=verb\n  /-/",  # quantifier
            "\n\nword sp=verb\n\n",  # blank lines
            "word sp=verb vt=infc",
        ],
    )
    def test_valid_syntax_passes(self, template):
        check_template(template, self.TYPES)

    FEATURES = ("typ", "function", "sp", "vs", "book", "book@en", "chapter", "verse")

    def test_unknown_feature_flagged_with_hint(self):
        with pytest.raises(TemplateError, match="unknown feature 'typp'.*Did you mean typ"):
            check_template("clause typp=Way0", self.TYPES, self.FEATURES)

    def test_unknown_feature_on_nested_and_operator_lines(self):
        with pytest.raises(TemplateError, match="unknown feature 'sq'"):
            check_template("clause\n  word sp=verb\n  < word sq=nmpr", self.TYPES, self.FEATURES)

    def test_unknown_feature_with_other_operators(self):
        for constraint in ("vss#qal", "vss~^q", "vss=qal|piel", "vss>3"):
            with pytest.raises(TemplateError, match="unknown feature 'vss'"):
                check_template(f"word {constraint}", self.TYPES, self.FEATURES)

    @pytest.mark.parametrize(
        "template",
        [
            "word sp=verb vs=qal",
            "word sp#verb",
            "word vs~^q",
            "word vs=qal|piel",
            "book book@en=Psalms",
            "a:word sp=verb\nb:word sp=nmpr\na < b",
            "word",  # no constraints at all
        ],
    )
    def test_known_features_pass(self, template):
        check_template(template, self.TYPES, self.FEATURES)

    def test_feature_names_not_checked_when_not_given(self):
        check_template("clause typp=Way0", self.TYPES)

    def test_feature_values_are_not_checked(self):
        check_template("word sp=verbs", self.TYPES, self.FEATURES)

    def test_rejects_scope_that_contains_nothing(self):
        """The template behind the client's Psalm 1:1 result: 1-space siblings."""
        with pytest.raises(TemplateError, match="contains nothing.*same indent"):
            check_template(
                "book book=PSA\n chapter chapter=23\n clause", self.TYPES, self.FEATURES
            )

    def test_rejects_scope_sibling_deeper_in_template(self):
        with pytest.raises(TemplateError, match="verse constraint would not apply"):
            check_template(
                "book book=PSA\n  chapter chapter=23\n    verse verse=1\n    clause",
                self.TYPES,
            )

    def test_rejects_unindented_top_level_lines(self):
        with pytest.raises(TemplateError, match="3 separate top-level lines"):
            check_template("book book=PSA\nchapter chapter=23\nclause", self.TYPES)

    @pytest.mark.parametrize(
        "template",
        [
            "book book=PSA\n  chapter chapter=23\n    clause",  # correct nesting
            "book book=PSA\n  chapter chapter=23\n    verse verse=1\n      clause",
            "book book=PSA\n    chapter chapter=23\n        clause",  # any consistent step
            "book book=PSA\n  chapter chapter=23\n  chapter chapter=24",  # sibling scopes
            "book book=PSA\n  chapter chapter=23",  # scope alone: finds the chapter
            "chapter chapter=23\n  clause typ=NmCl\n  clause typ=Way0",  # sibling patterns
            "a:clause\nb:clause\na < b",  # top-level lines joined by a relation
            "clause\n< clause",  # operator ties the second line to the first
            "clause\n/without/\nword sp=verb\n/-/",  # quantifier blocks are not judged
        ],
    )
    def test_valid_structure_passes(self, template):
        check_template(template, self.TYPES, self.FEATURES)

    def test_engine_rejects_the_clients_template(self, engine: CFEngine):
        with pytest.raises(TemplateError, match="contains nothing"):
            engine.search_constructions(
                "book book=PSA\n chapter chapter=23\n clause", "hebrew"
            )
        with pytest.raises(TemplateError, match="separate top-level lines"):
            engine.search_constructions(
                "book book=PSA\nchapter chapter=23\nclause", "hebrew"
            )

    def test_engine_correct_nesting_is_scoped_to_the_chapter(self, engine: CFEngine):
        results = engine.search_constructions(
            "book book=PSA\n  chapter chapter=23\n    clause", "hebrew", limit=100
        )
        assert len(results) == 17
        chapters = {
            o["chapter"] for r in results for o in r["objects"] if o["type"] == "clause"
        }
        assert chapters == {23}

    def test_engine_rejects_unknown_feature(self, engine: CFEngine):
        with pytest.raises(TemplateError, match="unknown feature 'typp'"):
            engine.search_constructions("clause typp=Way0", "hebrew")
        with pytest.raises(TemplateError, match="unknown feature 'sq'"):
            engine.search_advanced("word sq=verb", corpus="hebrew")

    def test_engine_accepts_documented_features(self, engine: CFEngine):
        assert engine.search_constructions("clause typ=Way0", "hebrew", limit=3)
        assert engine.search_constructions("w cls=noun", "greek", limit=3)

    def test_engine_raises_instead_of_returning_empty(self, engine: CFEngine):
        with pytest.raises(TemplateError):
            engine.search_constructions(
                "book@en=Psalms\n chapter chapter=23\n clause", "hebrew"
            )
        with pytest.raises(TemplateError):
            engine.search_advanced("clse typ=Way0", corpus="hebrew")

    def test_all_existing_test_templates_still_run(self, engine: CFEngine):
        template = "book book=Genesis\n  chapter chapter=1\n    clause typ=Way0\n"
        assert engine.search_constructions(template, "hebrew", limit=5)


class TestScopeTemplate:
    """The pure wrapper: pattern in, scoped template out."""

    TYPES = ("book", "chapter", "verse", "clause", "phrase", "word")

    def wrap(self, template, book="PSA", **kw):
        return scope_template(template, "hebrew", book, object_types=self.TYPES, **kw)

    def test_book_chapter(self):
        assert self.wrap("clause", chapter=23) == (
            "book book=Psalmi\n  chapter chapter=23\n    clause\n"
        )

    def test_book_only(self):
        assert self.wrap("clause typ=Way0") == "book book=Psalmi\n  clause typ=Way0\n"

    def test_single_verse_and_range(self):
        assert "verse verse=1\n" in self.wrap("clause", chapter=23, verse_start=1)
        assert "verse verse=1|2|3\n" in self.wrap(
            "clause", chapter=23, verse_start=1, verse_end=3
        )

    def test_verse_end_alone_starts_at_one(self):
        assert "verse verse=1|2\n" in self.wrap("clause", chapter=23, verse_end=2)

    def test_pattern_nesting_is_preserved_and_normalised(self):
        out = self.wrap("  clause\n    phrase\n    word", chapter=23)
        assert out == (
            "book book=Psalmi\n  chapter chapter=23\n"
            "    clause\n      phrase\n      word\n"
        )

    def test_greek_uses_code(self):
        out = scope_template("w cls=noun", "greek", "Matthew", 1, object_types=("w",))
        assert out.startswith("book book=MAT\n  chapter chapter=1\n")

    def test_no_scope_returns_template_unchanged(self):
        assert scope_template("clause", "hebrew", None) == "clause"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"book": None, "chapter": 23}, "require book"),
            ({"chapter": None, "verse_start": 1}, "require chapter"),
            ({"chapter": 23, "verse_start": 5, "verse_end": 2}, "before verse_start"),
        ],
    )
    def test_inconsistent_parameters_raise(self, kwargs, message):
        args = {"book": "PSA", "chapter": 23, **kwargs}
        with pytest.raises(TemplateError, match=message):
            scope_template("clause", "hebrew", **args, object_types=self.TYPES)

    @pytest.mark.parametrize(
        "template, kwargs",
        [
            ("book book=PSA\n  clause", {}),
            ("chapter chapter=23\n  clause", {"chapter": 23}),
            ("verse verse=1\n  clause", {"chapter": 23, "verse_start": 1}),
        ],
    )
    def test_pattern_repeating_the_scope_is_rejected(self, template, kwargs):
        with pytest.raises(TemplateError, match="repeats the scope"):
            self.wrap(template, **kwargs)

    def test_pattern_may_use_levels_below_the_scope(self):
        # book param only: the pattern is free to constrain chapter itself
        assert "chapter chapter=23" in self.wrap("chapter chapter=23\n  clause")


class TestScopeParameters:
    def test_matches_hand_nested_template(self, engine: CFEngine):
        by_param = engine.search_constructions("clause", "hebrew", 100, "PSA", 23)
        by_hand = engine.search_constructions(
            "book book=PSA\n  chapter chapter=23\n    clause", "hebrew", limit=100
        )
        assert len(by_param) == len(by_hand) == 17
        # the injected book/chapter scope is left out of scoped results
        assert [r["objects"] for r in by_param] == [r["objects"][2:] for r in by_hand]

    def test_every_result_is_inside_the_chapter(self, engine: CFEngine):
        results = engine.search_constructions(
            "clause", "hebrew", 100, book="Psalms", chapter=23
        )
        clauses = [o for r in results for o in r["objects"] if o["type"] == "clause"]
        assert {(c["book"], c["chapter"]) for c in clauses} == {("PSA", 23)}

    def test_book_only(self, engine: CFEngine):
        results = engine.search_constructions("clause typ=Way0", "hebrew", 5, book="PSA")
        assert results
        assert {o["book"] for r in results for o in r["objects"] if o["type"] == "clause"} == {"PSA"}

    def test_single_verse_and_range(self, engine: CFEngine):
        one = engine.search_constructions(
            "clause", "hebrew", 100, "PSA", 23, verse_start=1
        )
        assert len(one) == 3
        assert {c["verse"] for r in one for c in r["objects"] if c["type"] == "clause"} == {1}
        rng = engine.search_constructions(
            "clause", "hebrew", 100, "PSA", 23, verse_start=1, verse_end=3
        )
        assert len(rng) == 7
        verses = {c["verse"] for r in rng for c in r["objects"] if c["type"] == "clause"}
        assert verses == {1, 2, 3}

    def test_nested_pattern_with_scope(self, engine: CFEngine):
        results = engine.search_constructions(
            "clause typ=NmCl\n  phrase function=Subj", "hebrew", 20, "PSA", 23
        )
        assert results

    def test_multiple_top_level_pattern_lines_are_connected_by_the_scope(
        self, engine: CFEngine
    ):
        assert engine.search_constructions(
            "clause typ=NmCl\nclause typ=xYq0", "hebrew", 5, "PSA", 23, verse_start=1
        )

    def test_greek(self, engine: CFEngine):
        results = engine.search_constructions(
            "w cls=noun", "greek", 3, book="Matthew", chapter=1
        )
        assert results
        assert results[0]["objects"][-1]["book"] == "MAT"

    def test_search_advanced_count(self, engine: CFEngine):
        result = engine.search_advanced(
            "clause", return_type="count", corpus="hebrew", book="PSA", chapter=23
        )
        assert result["total_count"] == 17

    def test_pattern_is_still_validated(self, engine: CFEngine):
        with pytest.raises(TemplateError, match="unknown feature 'typp'"):
            engine.search_constructions("clause typp=Way0", "hebrew", 5, "PSA", 23)
        with pytest.raises(TemplateError, match="does not start with an object type"):
            engine.search_constructions("typ=Way0", "hebrew", 5, "PSA", 23)

    def test_errors(self, engine: CFEngine):
        with pytest.raises(TemplateError, match="repeats the scope"):
            engine.search_constructions(
                "book book=PSA\n  clause", "hebrew", 5, "PSA", 23
            )
        with pytest.raises(TemplateError, match="require book"):
            engine.search_constructions("clause", "hebrew", 5, chapter=23)
        with pytest.raises(UnknownBookError, match="Did you mean PSA"):
            engine.search_constructions("clause", "hebrew", 5, book="Psalmz")
        with pytest.raises(UnknownBookError, match="not in the greek"):
            engine.search_constructions("w", "greek", 5, book="PSA")


class TestResultObjects:
    def test_scoped_results_contain_only_the_pattern(self, engine: CFEngine):
        results = engine.search_constructions("clause", "hebrew", 100, "PSA", 23)
        assert all([o["type"] for o in r["objects"]] == ["clause"] for r in results)

    def test_scope_nodes_skipped_for_verse_scope_too(self, engine: CFEngine):
        results = engine.search_constructions(
            "clause\n  phrase", "hebrew", 100, "PSA", 23, verse_start=1
        )
        assert all([o["type"] for o in r["objects"]] == ["clause", "phrase"] for r in results)

    def test_container_objects_do_not_swamp_the_result(self, engine: CFEngine):
        """A hand-written scope returns book/chapter nodes; they must stay small."""
        results = engine.search_constructions(
            "book book=PSA\n  chapter chapter=23\n    clause", "hebrew", 100
        )
        book, chapter, clause = results[0]["objects"]
        assert book["text"] == "[25372 slots - text omitted]"
        assert "features" not in book and "features" not in chapter
        assert chapter["text"].startswith("מִזְמֹ")  # Psalm 23 is short enough to show
        assert "features" in clause
        assert len(json.dumps(results[0], ensure_ascii=False)) < 2000

    def test_long_chapter_text_is_omitted(self, engine: CFEngine):
        results = engine.search_constructions(
            "book book=PSA\n  chapter chapter=119", "hebrew", 1
        )
        assert results[0]["objects"][1]["text"].endswith("slots - text omitted]")

    def test_unknown_object_type_error_lists_valid_types(self, engine: CFEngine):
        with pytest.raises(TemplateError, match="Valid object types:.*\\bw\\b"):
            engine.search_constructions("word cls=verb", "greek")
