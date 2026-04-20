"""Tests for quiz models, storage, and API endpoints."""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from context_fabric_mcp.api import app
from context_fabric_mcp.cf_engine import CFEngine
from context_fabric_mcp.quiz_engine import QuizStore, generate_session
from context_fabric_mcp.quiz_models import (
    FeatureConfig,
    FeatureVisibility,
    QuizDefinition,
)


@pytest.fixture(scope="module")
def engine():
    return CFEngine()


@pytest.fixture
def tmp_store(tmp_path):
    return QuizStore(directory=tmp_path)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --- Model tests ---


class TestQuizModel:
    def test_default_quiz(self):
        q = QuizDefinition()
        assert q.title == "Untitled Quiz"
        assert q.corpus == "hebrew"
        assert len(q.id) == 12

    def test_custom_quiz(self):
        q = QuizDefinition(
            title="Hebrew Verbs",
            corpus="hebrew",
            book="Genesis",
            chapter_start=1,
            chapter_end=1,
            search_template="word sp=verb",
            features=[
                FeatureConfig(name="gloss", visibility=FeatureVisibility.show),
                FeatureConfig(
                    name="part_of_speech", visibility=FeatureVisibility.request
                ),
                FeatureConfig(name="verbal_stem", visibility=FeatureVisibility.request),
            ],
        )
        assert q.title == "Hebrew Verbs"
        assert len(q.features) == 3


# --- Store tests ---


class TestQuizStore:
    def test_save_and_load(self, tmp_store):
        q = QuizDefinition(title="Test Quiz")
        tmp_store.save(q)
        loaded = tmp_store.load(q.id)
        assert loaded.title == "Test Quiz"
        assert loaded.id == q.id

    def test_list_all(self, tmp_store):
        q1 = QuizDefinition(title="Quiz A")
        q2 = QuizDefinition(title="Quiz B")
        tmp_store.save(q1)
        tmp_store.save(q2)
        all_quizzes = tmp_store.list_all()
        titles = [q["title"] for q in all_quizzes]
        assert "Quiz A" in titles
        assert "Quiz B" in titles

    def test_delete(self, tmp_store):
        q = QuizDefinition(title="To Delete")
        tmp_store.save(q)
        tmp_store.delete(q.id)
        with pytest.raises(FileNotFoundError):
            tmp_store.load(q.id)

    def test_load_nonexistent(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.load("nonexistent")


# --- Session generation tests ---


class TestQuizGeneration:
    def test_generate_hebrew_verb_quiz(self, engine):
        quiz = QuizDefinition(
            title="Genesis 1 Verbs",
            corpus="hebrew",
            book="Genesis",
            chapter_start=1,
            chapter_end=1,
            search_template="word sp=verb",
            features=[
                FeatureConfig(name="gloss", visibility=FeatureVisibility.show),
                FeatureConfig(
                    name="part_of_speech", visibility=FeatureVisibility.request
                ),
            ],
            max_questions=5,
            randomize=False,
        )
        session = generate_session(quiz, engine)
        assert session.quiz_title == "Genesis 1 Verbs"
        assert len(session.questions) == 5
        for q in session.questions:
            assert q.book == "Genesis"
            assert q.chapter == 1
            assert q.word_text != ""
            assert "gloss" in q.shown_features
            assert "part_of_speech" in q.requested_features
            assert q.requested_features["part_of_speech"] == "verb"

    def test_generate_with_verse_range(self, engine):
        quiz = QuizDefinition(
            corpus="hebrew",
            book="Genesis",
            chapter_start=1,
            chapter_end=1,
            verse_start=1,
            verse_end=3,
            search_template="word sp=verb",
            features=[
                FeatureConfig(name="verbal_stem", visibility=FeatureVisibility.request),
            ],
            max_questions=0,
            randomize=False,
        )
        session = generate_session(quiz, engine)
        assert len(session.questions) > 0
        for q in session.questions:
            assert 1 <= q.verse <= 3

    def test_generate_greek_quiz(self, engine):
        quiz = QuizDefinition(
            corpus="greek",
            book="MAT",
            chapter_start=1,
            chapter_end=1,
            search_template="w cls=noun",
            features=[
                FeatureConfig(name="gloss", visibility=FeatureVisibility.show),
                FeatureConfig(
                    name="part_of_speech", visibility=FeatureVisibility.request
                ),
            ],
            max_questions=3,
        )
        session = generate_session(quiz, engine)
        assert len(session.questions) <= 3
        assert len(session.questions) > 0


# --- API tests ---


class TestQuizAPI:
    def test_crud_lifecycle(self, client):
        # Create
        quiz_data = {
            "title": "API Test Quiz",
            "corpus": "hebrew",
            "book": "Genesis",
            "chapter_start": 1,
            "chapter_end": 1,
            "search_template": "word sp=verb",
            "features": [
                {"name": "gloss", "visibility": "show"},
                {"name": "part_of_speech", "visibility": "request"},
            ],
        }
        resp = client.post("/api/quizzes", json=quiz_data)
        assert resp.status_code == 200
        created = resp.json()
        quiz_id = created["id"]
        assert created["title"] == "API Test Quiz"

        # Read
        resp = client.get(f"/api/quizzes/{quiz_id}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "API Test Quiz"

        # Update
        quiz_data["title"] = "Updated Quiz"
        quiz_data["id"] = quiz_id
        resp = client.put(f"/api/quizzes/{quiz_id}", json=quiz_data)
        assert resp.status_code == 200
        assert resp.json()["title"] == "Updated Quiz"

        # List
        resp = client.get("/api/quizzes")
        assert resp.status_code == 200
        titles = [q["title"] for q in resp.json()]
        assert "Updated Quiz" in titles

        # Generate session
        resp = client.post(f"/api/quizzes/{quiz_id}/generate")
        assert resp.status_code == 200
        session = resp.json()
        assert session["quiz_id"] == quiz_id
        assert len(session["questions"]) > 0

        # Delete
        resp = client.delete(f"/api/quizzes/{quiz_id}")
        assert resp.status_code == 200

    def test_get_nonexistent_quiz(self, client):
        resp = client.get("/api/quizzes/nonexistent")
        assert resp.status_code == 404
