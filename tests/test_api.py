"""Tests for the FastAPI HTTP layer."""

import pytest
from fastapi.testclient import TestClient

from context_fabric_mcp.api import app


@pytest.fixture(scope="module")
def client():
    """Shared test client — corpora load once per module."""
    return TestClient(app)


# --- Corpora ---


class TestCorpora:
    def test_list_corpora(self, client):
        resp = client.get("/api/corpora")
        assert resp.status_code == 200
        data = resp.json()
        ids = [c["id"] for c in data]
        assert "hebrew" in ids
        assert "greek" in ids


# --- Books ---


class TestBooks:
    def test_hebrew_books(self, client):
        resp = client.get("/api/books", params={"corpus": "hebrew"})
        assert resp.status_code == 200
        books = resp.json()
        assert len(books) == 39
        assert books[0]["code"] == "GEN"
        assert books[0]["name"] == "Genesis"
        assert books[0]["chapters"] == 50

    def test_greek_books(self, client):
        resp = client.get("/api/books", params={"corpus": "greek"})
        assert resp.status_code == 200
        books = resp.json()
        assert len(books) == 27


# --- Passage ---


class TestPassage:
    def test_genesis_1_1(self, client):
        resp = client.get(
            "/api/passage",
            params={
                "book": "Genesis",
                "chapter": 1,
                "verse_start": 1,
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["corpus"] == "hebrew"
        assert len(data["verses"]) == 1
        verse = data["verses"][0]
        assert verse["book"] == "GEN"
        assert verse["book_name"] == "Genesis"
        assert verse["chapter"] == 1
        assert verse["verse"] == 1
        # Genesis 1:1 has 11 words in Hebrew
        assert len(verse["words"]) == 11
        first_word = verse["words"][0]
        assert first_word["text"] != ""
        assert first_word["part_of_speech"] != ""

    def test_verse_range(self, client):
        resp = client.get(
            "/api/passage",
            params={
                "book": "Genesis",
                "chapter": 1,
                "verse_start": 1,
                "verse_end": 3,
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["verses"]) == 3

    def test_greek_passage(self, client):
        resp = client.get(
            "/api/passage",
            params={
                "book": "MAT",
                "chapter": 1,
                "verse_start": 1,
                "corpus": "greek",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["corpus"] == "greek"
        assert len(data["verses"]) == 1
        assert len(data["verses"][0]["words"]) > 0

    def test_nonexistent_verse(self, client):
        resp = client.get(
            "/api/passage",
            params={
                "book": "Genesis",
                "chapter": 1,
                "verse_start": 999,
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["verses"]) == 0


# --- Schema ---


class TestSchema:
    def test_hebrew_schema(self, client):
        resp = client.get("/api/schema", params={"corpus": "hebrew"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["corpus"] == "hebrew"
        type_names = [t["name"] for t in data["object_types"]]
        assert "word" in type_names
        assert "clause" in type_names


# --- Search ---


class TestSearch:
    def test_search_words(self, client):
        resp = client.post(
            "/api/search/words",
            json={
                "corpus": "hebrew",
                "book": "Genesis",
                "chapter": 1,
                "features": {"sp": "verb"},
                "limit": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert data[0]["book"] == "GEN"
        assert data[0]["chapter"] == 1
        assert data[0]["word"]["part_of_speech"] == "verb"

    def test_search_constructions(self, client):
        resp = client.post(
            "/api/search/constructions",
            json={
                "template": "clause typ=Way0\n  phrase function=Pred\n    word sp=verb\n",
                "corpus": "hebrew",
                "limit": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        # Each result should have objects
        assert "objects" in data[0]
        assert len(data[0]["objects"]) == 3  # clause, phrase, word


# --- Lexeme ---


class TestLexeme:
    def test_hebrew_lexeme(self, client):
        resp = client.get(
            "/api/lexeme/BR>%5B",  # BR>[ URL-encoded
            params={"corpus": "hebrew", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["lexeme"] == "BR>["
        assert data["gloss"] == "create"
        assert data["total_occurrences"] > 0
        assert len(data["occurrences"]) <= 5


# --- Vocabulary ---


class TestVocabulary:
    def test_genesis_1_1_vocab(self, client):
        resp = client.get(
            "/api/vocabulary",
            params={
                "book": "Genesis",
                "chapter": 1,
                "verse_start": 1,
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        # Each entry should have required fields
        entry = data[0]
        assert "lexeme" in entry
        assert "gloss" in entry
        assert "part_of_speech" in entry
        assert "count" in entry


# --- Context ---


class TestContext:
    def test_word_context(self, client):
        resp = client.get(
            "/api/context",
            params={
                "book": "Genesis",
                "chapter": 1,
                "verse": 1,
                "word_index": 0,
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "word" in data
        assert data["word"]["text"] != ""
        # Should have at least one parent type (phrase, clause, etc.)
        parent_keys = [k for k in data.keys() if k != "word"]
        assert len(parent_keys) > 0


# --- Search Syntax Guide ---


class TestSearchSyntaxGuide:
    def test_overview(self, client):
        resp = client.get("/api/search/syntax-guide")
        assert resp.status_code == 200
        data = resp.json()
        assert "sections" in data
        assert "basics" in data["sections"]

    def test_specific_section(self, client):
        resp = client.get("/api/search/syntax-guide", params={"section": "basics"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["section"] == "basics"
        assert "content" in data


# --- Features ---


class TestFeatures:
    def test_list_all(self, client):
        resp = client.get("/api/features", params={"corpus": "hebrew"})
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert len(data["features"]) > 0

    def test_list_word_features(self, client):
        resp = client.get(
            "/api/features", params={"node_types": "word", "corpus": "hebrew"}
        )
        assert resp.status_code == 200
        names = [f["name"] for f in resp.json()["features"]]
        assert "sp" in names

    def test_describe_sp(self, client):
        resp = client.get("/api/features/sp", params={"corpus": "hebrew"})
        assert resp.status_code == 200
        data = resp.json()
        assert "sample_values" in data or "samples" in data


# --- Advanced Search ---


class TestAdvancedSearch:
    def test_count(self, client):
        resp = client.post(
            "/api/search/advanced",
            json={
                "template": "word sp=verb\n",
                "return_type": "count",
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] > 0

    def test_statistics(self, client):
        resp = client.post(
            "/api/search/advanced",
            json={
                "template": "book book=Genesis\n  chapter chapter=1\n    word sp=verb\n",
                "return_type": "statistics",
                "aggregate_features": ["vs", "vt"],
                "corpus": "hebrew",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] > 0


# --- Edges ---


class TestEdges:
    def test_list_edge_features(self, client):
        resp = client.get("/api/edges", params={"corpus": "hebrew"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        names = [e["name"] for e in data]
        assert "mother" in names

    def test_get_edges(self, client):
        resp = client.get(
            "/api/edges/mother",
            params={"node": 1, "direction": "to", "corpus": "hebrew"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["edge_feature"] == "mother"
        assert "edges" in data


class TestBookResolution:
    def test_passage_accepts_code_and_latin_name(self, client):
        for book in ("PSA", "Psalms", "Psalmi"):
            resp = client.get(
                "/api/passage",
                params={"corpus": "hebrew", "book": book, "chapter": 23, "verse_start": 1},
            )
            assert resp.status_code == 200
            verse = resp.json()["verses"][0]
            assert (verse["book"], verse["book_name"]) == ("PSA", "Psalms")

    def test_unknown_book_is_404_with_hint(self, client):
        resp = client.get(
            "/api/passage",
            params={"corpus": "hebrew", "book": "Psalmz", "chapter": 23, "verse_start": 1},
        )
        assert resp.status_code == 404
        assert "PSA" in resp.json()["detail"]


class TestTemplateErrors:
    def test_malformed_template_is_400_with_hint(self, client):
        resp = client.post(
            "/api/search/constructions",
            json={"template": "book@en=Psalms\n chapter chapter=23\n clause", "corpus": "hebrew"},
        )
        assert resp.status_code == 400
        assert "object type" in resp.json()["detail"]


class TestScopedSearch:
    def test_constructions_with_scope_parameters(self, client):
        resp = client.post(
            "/api/search/constructions",
            json={"template": "clause", "book": "PSA", "chapter": 23},
        )
        assert resp.status_code == 200
        clauses = [
            o for r in resp.json() for o in r["objects"] if o["type"] == "clause"
        ]
        assert len(clauses) == 17
        assert {c["chapter"] for c in clauses} == {23}

    def test_advanced_with_scope_parameters(self, client):
        resp = client.post(
            "/api/search/advanced",
            json={"template": "clause", "return_type": "count", "book": "Psalms", "chapter": 23},
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 17

    def test_scope_conflict_is_400(self, client):
        resp = client.post(
            "/api/search/constructions",
            json={"template": "book book=PSA\n  clause", "book": "PSA", "chapter": 23},
        )
        assert resp.status_code == 400
        assert "repeats the scope" in resp.json()["detail"]


class TestCompareDistributionAPI:
    def test_exact_counts_with_filter(self, client):
        resp = client.post(
            "/api/compare/distribution",
            json={"feature": "vt", "sections": [{"book": "GEN"}], "features": {"sp": "verb"}},
        )
        assert resp.status_code == 200
        assert resp.json()["comparison"]["GEN (hebrew)"]["total_count"] == 5060

    def test_unknown_feature_is_400(self, client):
        resp = client.post(
            "/api/compare/distribution",
            json={"feature": "vtt", "sections": [{"book": "GEN"}]},
        )
        assert resp.status_code == 400
        assert "Did you mean vt" in resp.json()["detail"]
