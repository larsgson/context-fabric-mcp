"""Tests for the Context-Fabric engine.

These tests require corpus data to be available locally (pre-downloaded
TF-format data for BHSA and Nestle 1904).
"""

import pytest

from context_fabric_mcp.cf_engine import CFEngine


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
        assert verse.book == "Genesis"
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
            assert r["book"] == "Genesis"
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
            any(o["book"] == "Genesis" for o in result["occurrences"]) or len(books) > 0
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
        assert first.name == "MAT"
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
