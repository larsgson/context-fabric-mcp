"""Context-Fabric data access layer for biblical texts.

Drop-in replacement for tf_engine.py that uses Context Fabric's memory-mapped
engine instead of Text-Fabric. The public interface (method signatures, return
types) is identical so that chat.py, quiz_engine.py, and api.py work unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import cfabric
from cfabric.core.api import Api

from context_fabric_mcp.models import (
    BookInfo,
    FeatureInfo,
    ObjectTypeInfo,
    PassageResult,
    SchemaResult,
    VerseResult,
    WordInfo,
)

logger = logging.getLogger(__name__)

# Corpus registry: name -> (path-finding callable, display name)
# Paths are resolved at load time from the TF data cache or a configured location.
CORPORA = {
    "hebrew": ("ETCBC/bhsa", "Biblical Hebrew (BHSA)"),
    "greek": ("ETCBC/nestle1904", "Greek New Testament (Nestle 1904)"),
}

# Word-level features to retrieve per corpus.
# Maps our canonical names to the TF/CF feature names.
WORD_FEATURES = {
    "hebrew": {
        "text": "g_word_utf8",
        "trailer": "trailer_utf8",
        "lexeme": "lex",
        "lexeme_utf8": "lex_utf8",
        "gloss": "gloss",
        "part_of_speech": "sp",
        "gender": "gn",
        "number": "nu",
        "person": "ps",
        "state": "st",
        "verbal_stem": "vs",
        "verbal_tense": "vt",
        "language": "language",
    },
    "greek": {
        "text": "unicode",
        "trailer": "after",
        "lexeme": "lemma",
        "lexeme_utf8": "lemma",
        "gloss": "gloss",
        "part_of_speech": "cls",
        "gender": "gender",
        "number": "number",
        "person": "person",
        "state": "",
        "verbal_stem": "voice",
        "verbal_tense": "tense",
        "language": "",
    },
}

# The word-level object type name differs per corpus
WORD_TYPE = {
    "hebrew": "word",
    "greek": "w",
}

# Features to exclude per corpus when loading.
# The Nestle 1904 "nodeId" feature has int64 values that overflow CF's int32
# storage (as of CF 0.5.x).  We don't use it, so skip it.
_EXCLUDE_FEATURES: dict[str, set[str]] = {
    "greek": {"nodeId"},
}


def _find_corpus_path(org_repo: str) -> str:
    """Locate a TF-format corpus on disk.

    Searches the standard text-fabric-data locations that both TF and CF
    use when corpora have been pre-downloaded.
    """
    home = os.environ.get("HOME", str(Path.home()))
    # Standard TF download layout: ~/text-fabric-data/github/ORG/REPO/tf/VERSION
    base = Path(home) / "text-fabric-data" / "github" / org_repo / "tf"
    if base.exists():
        # Pick the latest version directory
        versions = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if versions:
            return str(versions[0])

    # Fallback: try cfabric cache directory
    try:
        cache_dir = cfabric.get_cache_dir()
        alt = Path(cache_dir) / org_repo
        if alt.exists():
            return str(alt)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Corpus data not found for {org_repo}. "
        f"Searched: {base}. HOME={home}. "
        "Ensure the corpus has been pre-downloaded."
    )


class CFEngine:
    """Manages Context-Fabric corpus loading and queries.

    Public API is identical to the former TFEngine so that all callers
    (chat.py, quiz_engine.py, api.py, tools/*) work without changes.
    """

    def __init__(self) -> None:
        self._apis: dict[str, Api] = {}
        self._fabrics: dict[str, cfabric.Fabric] = {}
        self._load_lock = threading.Lock()

    def _ensure_loaded(self, corpus: str) -> Api:
        """Load a corpus if not already loaded, return the CF Api."""
        if corpus not in CORPORA:
            raise ValueError(
                f"Unknown corpus '{corpus}'. Available: {list(CORPORA.keys())}"
            )
        # Fast path: already loaded (no lock needed)
        if corpus in self._apis:
            return self._apis[corpus]
        # Slow path: acquire lock, double-check, then load
        with self._load_lock:
            if corpus in self._apis:
                return self._apis[corpus]
            org_repo, display_name = CORPORA[corpus]
            logger.info(
                "Loading %s (%s) via Context-Fabric ...", display_name, org_repo
            )

            path = _find_corpus_path(org_repo)
            logger.info("Corpus path: %s", path)

            # Hide excluded .tf files so CF never sees them during scan
            # or compilation (CF auto-compiles ALL .tf it finds).
            exclude = _EXCLUDE_FEATURES.get(corpus, set())
            hidden: list[tuple[Path, Path]] = []
            for feat_name in exclude:
                tf_file = Path(path) / f"{feat_name}.tf"
                skip_file = tf_file.with_suffix(".tf._skip")
                if tf_file.exists():
                    tf_file.rename(skip_file)
                    hidden.append((skip_file, tf_file))
                    logger.info("Temporarily hidden: %s", tf_file)

            try:
                CF = cfabric.Fabric(locations=path, silent="deep")
                api = CF.loadAll(silent="deep")
            except Exception as e:
                logger.error("Context-Fabric load failed for %s: %s", path, e)
                raise RuntimeError(
                    f"Failed to load corpus '{corpus}' from {path}: {e}"
                ) from e
            finally:
                # Always restore hidden files
                for skip_file, tf_file in hidden:
                    if skip_file.exists():
                        skip_file.rename(tf_file)
                        logger.info("Restored: %s", tf_file)

            if api is None or not hasattr(api, "T") or not hasattr(api.F, "otype"):
                logger.error(
                    "Corpus loaded but API incomplete: api=%s, hasT=%s, hasOtype=%s",
                    api is not None,
                    hasattr(api, "T") if api else False,
                    hasattr(api.F, "otype") if api and hasattr(api, "F") else False,
                )
                raise RuntimeError(
                    f"Failed to load corpus '{corpus}' from {path}. "
                    "Context-Fabric API did not initialize correctly."
                )

            self._fabrics[corpus] = CF
            self._apis[corpus] = api

            # Register with cfabric_mcp corpus_manager so built-in tools
            # (search, describe_feature, etc.) can access our loaded corpora.
            try:
                from cfabric_mcp.corpus_manager import corpus_manager as cm

                if not cm.is_loaded(corpus):
                    cm._corpora[corpus] = (CF, api)
                    if cm.current is None:
                        cm._current = corpus
            except ImportError:
                pass

            logger.info("Loaded %s", display_name)

        return self._apis[corpus]

    # ------------------------------------------------------------------
    # Public query methods — identical signatures to TFEngine
    # ------------------------------------------------------------------

    def list_corpora(self) -> list[dict[str, str]]:
        """Return available corpora."""
        return [{"id": cid, "name": display} for cid, (_, display) in CORPORA.items()]

    def list_books(self, corpus: str = "hebrew") -> list[BookInfo]:
        """Return all books with chapter counts for a corpus."""
        api = self._ensure_loaded(corpus)
        books = []
        for book_node in api.F.otype.s("book"):
            book_name = api.T.sectionFromNode(book_node)[0]
            chapter_nodes = api.L.d(book_node, otype="chapter")
            books.append(BookInfo(name=book_name, chapters=len(chapter_nodes)))
        return books

    def get_passage(
        self,
        book: str,
        chapter: int,
        verse_start: int = 1,
        verse_end: int | None = None,
        corpus: str = "hebrew",
    ) -> PassageResult:
        """Get words with features for a verse range."""
        api = self._ensure_loaded(corpus)
        feat_map = WORD_FEATURES.get(corpus, WORD_FEATURES["hebrew"])

        if verse_end is None:
            verse_end = verse_start

        wtype = WORD_TYPE.get(corpus, "word")

        verses: list[VerseResult] = []
        for verse_num in range(verse_start, verse_end + 1):
            verse_node = api.T.nodeFromSection((book, chapter, verse_num))
            if verse_node is None:
                continue

            word_nodes = api.L.d(verse_node, otype=wtype)
            words: list[WordInfo] = []
            for w in word_nodes:
                words.append(self._word_info(api, w, feat_map))

            verses.append(
                VerseResult(
                    book=book,
                    chapter=chapter,
                    verse=verse_num,
                    words=words,
                )
            )

        return PassageResult(corpus=corpus, verses=verses)

    def get_schema(self, corpus: str = "hebrew") -> SchemaResult:
        """Return object types and their features for a corpus."""
        api = self._ensure_loaded(corpus)

        object_types: list[ObjectTypeInfo] = []
        for otype in api.F.otype.all:
            nodes = api.F.otype.s(otype)
            count = len(nodes)
            if count == 0:
                continue

            sample_node = nodes[0]
            features: list[FeatureInfo] = []
            for feat_name in sorted(api.Fall()):
                feat_obj = api.Fs(feat_name)
                if feat_obj is None:
                    continue
                val = feat_obj.v(sample_node)
                if val is not None:
                    features.append(FeatureInfo(name=feat_name))

            object_types.append(
                ObjectTypeInfo(name=otype, count=count, features=features)
            )

        return SchemaResult(corpus=corpus, object_types=object_types)

    def search_words(
        self,
        corpus: str = "hebrew",
        book: str | None = None,
        chapter: int | None = None,
        features: dict[str, str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search for words matching feature constraints."""
        api = self._ensure_loaded(corpus)
        wtype = WORD_TYPE.get(corpus, "word")

        # Build search template
        constraints = []
        if features:
            for feat, val in features.items():
                constraints.append(f"  {feat}={val}")

        constraint_str = "\n".join(constraints)

        if book and chapter:
            template = (
                f"book book={book}\n"
                f"  chapter chapter={chapter}\n"
                f"    {wtype}\n{constraint_str}\n"
            )
        elif book:
            template = f"book book={book}\n  {wtype}\n{constraint_str}\n"
        else:
            template = f"{wtype}\n{constraint_str}\n"

        results = list(api.S.search(template))

        feat_map = WORD_FEATURES.get(corpus, WORD_FEATURES["hebrew"])
        output = []
        for result_tuple in results[:limit]:
            w = result_tuple[-1]
            section = api.T.sectionFromNode(w)
            info = self._word_info(api, w, feat_map)
            output.append(
                {
                    "book": section[0],
                    "chapter": section[1],
                    "verse": section[2],
                    "word": info.model_dump(),
                }
            )

        return output

    def get_context(
        self,
        book: str,
        chapter: int,
        verse: int,
        word_index: int = 0,
        corpus: str = "hebrew",
    ) -> dict:
        """Get the hierarchical context (phrase, clause, sentence) for a word."""
        api = self._ensure_loaded(corpus)

        wtype = WORD_TYPE.get(corpus, "word")

        verse_node = api.T.nodeFromSection((book, chapter, verse))
        if verse_node is None:
            return {"error": f"Verse not found: {book} {chapter}:{verse}"}

        word_nodes = api.L.d(verse_node, otype=wtype)
        if word_index >= len(word_nodes):
            return {
                "error": f"Word index {word_index} out of range (max {len(word_nodes) - 1})"
            }

        w = word_nodes[word_index]
        feat_map = WORD_FEATURES.get(corpus, WORD_FEATURES["hebrew"])

        context: dict[str, Any] = {
            "word": self._word_info(api, w, feat_map).model_dump(),
        }

        all_types = [
            t for t in api.F.otype.all if t not in ("book", "chapter", "verse", wtype)
        ]
        for parent_type in all_types:
            parents = api.L.u(w, otype=parent_type)
            if parents:
                parent = parents[0]
                parent_features = {}
                for feat_name in sorted(api.Fall()):
                    feat_obj = api.Fs(feat_name)
                    if feat_obj is None:
                        continue
                    val = feat_obj.v(parent)
                    if val is not None:
                        parent_features[feat_name] = str(val)
                context[parent_type] = {
                    "node": int(parent),
                    "features": parent_features,
                    "text": api.T.text(parent),
                }

        return context

    def search_constructions(
        self,
        template: str,
        corpus: str = "hebrew",
        limit: int = 50,
    ) -> list[dict]:
        """Search using a Text-Fabric search template for structural patterns."""
        api = self._ensure_loaded(corpus)
        feat_map = WORD_FEATURES.get(corpus, WORD_FEATURES["hebrew"])
        wtype = WORD_TYPE.get(corpus, "word")

        results = list(api.S.search(template))

        output = []
        for result_tuple in results[:limit]:
            entry: dict[str, Any] = {"objects": []}
            for node in result_tuple:
                otype = api.F.otype.v(node)
                section = api.T.sectionFromNode(node)
                obj: dict[str, Any] = {
                    "type": otype,
                    "book": section[0] if len(section) > 0 else "",
                    "chapter": section[1] if len(section) > 1 else 0,
                    "verse": section[2] if len(section) > 2 else 0,
                    "text": api.T.text(node),
                }
                if otype == wtype:
                    obj["word"] = self._word_info(api, node, feat_map).model_dump()
                else:
                    features = {}
                    for feat_name in sorted(api.Fall()):
                        feat_obj = api.Fs(feat_name)
                        if feat_obj is None:
                            continue
                        val = feat_obj.v(node)
                        if val is not None:
                            features[feat_name] = str(val)
                    obj["features"] = features
                entry["objects"].append(obj)
            output.append(entry)

        return output

    def get_lexeme_info(
        self,
        lexeme: str,
        corpus: str = "hebrew",
        limit: int = 50,
    ) -> dict:
        """Look up a lexeme and return its occurrences with context."""
        api = self._ensure_loaded(corpus)
        feat_map = WORD_FEATURES.get(corpus, WORD_FEATURES["hebrew"])

        lex_feat = feat_map.get("lexeme", "lex")
        gloss_feat = feat_map.get("gloss", "gloss")
        sp_feat = feat_map.get("part_of_speech", "sp")
        lex_utf8_feat = feat_map.get("lexeme_utf8", "lex_utf8")

        wtype = WORD_TYPE.get(corpus, "word")
        template = f"{wtype} {lex_feat}={lexeme}\n"
        results = list(api.S.search(template))
        corpus_count = len(results)

        first_gloss = ""
        first_sp = ""
        first_utf8 = ""
        matches = []

        for result_tuple in results[:limit]:
            w = result_tuple[0]

            if not first_gloss and gloss_feat:
                feat_obj = api.Fs(gloss_feat)
                if feat_obj:
                    g = feat_obj.v(w)
                    if g:
                        first_gloss = str(g)
            if not first_sp and sp_feat:
                feat_obj = api.Fs(sp_feat)
                if feat_obj:
                    s = feat_obj.v(w)
                    if s:
                        first_sp = str(s)
            if not first_utf8 and lex_utf8_feat:
                feat_obj = api.Fs(lex_utf8_feat)
                if feat_obj:
                    u = feat_obj.v(w)
                    if u:
                        first_utf8 = str(u)

            section = api.T.sectionFromNode(w)
            matches.append(
                {
                    "book": section[0],
                    "chapter": section[1],
                    "verse": section[2],
                    "word": self._word_info(api, w, feat_map).model_dump(),
                }
            )

        return {
            "lexeme": lexeme,
            "lexeme_utf8": first_utf8,
            "gloss": first_gloss,
            "part_of_speech": first_sp,
            "total_occurrences": corpus_count,
            "occurrences": matches,
        }

    def get_vocabulary(
        self,
        book: str,
        chapter: int,
        verse_start: int = 1,
        verse_end: int | None = None,
        corpus: str = "hebrew",
    ) -> list[dict]:
        """Get unique lexemes in a passage with frequency and gloss."""
        api = self._ensure_loaded(corpus)
        feat_map = WORD_FEATURES.get(corpus, WORD_FEATURES["hebrew"])

        if verse_end is None:
            verse_end = verse_start

        wtype = WORD_TYPE.get(corpus, "word")
        lexemes: dict[str, dict] = {}

        for v in range(verse_start, verse_end + 1):
            verse_node = api.T.nodeFromSection((book, chapter, v))
            if verse_node is None:
                continue
            for w in api.L.d(verse_node, otype=wtype):
                lex_feat = feat_map.get("lexeme", "lex")
                feat_obj = api.Fs(lex_feat)
                lex = feat_obj.v(w) if feat_obj else ""
                lex = lex or ""
                if not lex or lex in lexemes:
                    if lex in lexemes:
                        lexemes[lex]["count"] += 1
                    continue

                gloss_feat = feat_map.get("gloss", "gloss")
                gloss_obj = api.Fs(gloss_feat) if gloss_feat else None
                gloss_val = gloss_obj.v(w) if gloss_obj else None

                sp_feat = feat_map.get("part_of_speech", "sp")
                sp_obj = api.Fs(sp_feat) if sp_feat else None
                sp_val = sp_obj.v(w) if sp_obj else None

                lex_utf8_feat = feat_map.get("lexeme_utf8", "lex_utf8")
                utf8_obj = api.Fs(lex_utf8_feat) if lex_utf8_feat else None
                lex_utf8_val = utf8_obj.v(w) if utf8_obj else None

                freq_obj = api.Fs("freq_lex")
                freq_val = freq_obj.v(w) if freq_obj else None

                lexemes[lex] = {
                    "lexeme": lex,
                    "lexeme_utf8": str(lex_utf8_val) if lex_utf8_val else "",
                    "gloss": str(gloss_val) if gloss_val else "",
                    "part_of_speech": str(sp_val) if sp_val else "",
                    "corpus_frequency": int(freq_val) if freq_val else 0,
                    "count": 1,
                }

        return sorted(
            lexemes.values(), key=lambda x: x["corpus_frequency"], reverse=True
        )

    def _word_info(self, api: Api, w: int, feat_map: dict[str, str]) -> WordInfo:
        """Extract word features into a WordInfo model."""

        def _get(canonical: str) -> str:
            tf_name = feat_map.get(canonical, "")
            if not tf_name:
                return ""
            feat_obj = api.Fs(tf_name)
            if feat_obj is None:
                return ""
            val = feat_obj.v(w)
            return str(val) if val is not None else ""

        return WordInfo(
            monad=w,
            text=_get("text"),
            trailer=_get("trailer"),
            lexeme=_get("lexeme"),
            lexeme_utf8=_get("lexeme_utf8"),
            gloss=_get("gloss"),
            part_of_speech=_get("part_of_speech"),
            gender=_get("gender"),
            number=_get("number"),
            person=_get("person"),
            state=_get("state"),
            verbal_stem=_get("verbal_stem"),
            verbal_tense=_get("verbal_tense"),
            language=_get("language"),
        )

    # ------------------------------------------------------------------
    # Discovery & advanced search — delegates to cfabric_mcp built-ins
    # ------------------------------------------------------------------

    def get_search_syntax_guide(self, section: str | None = None) -> dict:
        """Return search syntax documentation.

        Args:
            section: Section name (basics, structure, relations, quantifiers,
                     examples). None returns overview with section list.
        """
        from cfabric_mcp.tools import search_syntax_guide

        return search_syntax_guide(section)

    def describe_feature(
        self,
        features: str | list[str],
        sample_limit: int = 20,
        corpus: str = "hebrew",
    ) -> dict:
        """Get detailed info about one or more features with sample values.

        Args:
            features: Feature name or list of names (e.g. "sp" or ["sp", "vt"])
            sample_limit: Max sample values per feature (default 20).
            corpus: Corpus name.
        """
        self._ensure_loaded(corpus)
        from cfabric_mcp.tools import describe_features

        return describe_features(features, sample_limit, corpus)

    def list_features(
        self,
        kind: str = "all",
        node_types: list[str] | None = None,
        corpus: str = "hebrew",
    ) -> dict:
        """List features with optional filtering by kind and node type.

        Args:
            kind: "all", "node", or "edge".
            node_types: Filter to features for these types (e.g. ["word"]).
            corpus: Corpus name.
        """
        self._ensure_loaded(corpus)
        from cfabric_mcp.tools import list_features

        return list_features(kind, node_types, corpus)

    def search_advanced(
        self,
        template: str,
        return_type: str = "results",
        aggregate_features: list[str] | None = None,
        group_by_section: bool = False,
        top_n: int = 50,
        limit: int = 100,
        corpus: str = "hebrew",
    ) -> dict:
        """Search with advanced return types.

        Args:
            template: Search template string.
            return_type: "results", "count", "statistics", or "passages".
            aggregate_features: For statistics — which features to aggregate.
            group_by_section: For statistics — include distribution by book.
            top_n: For statistics — max values per feature (default 50).
            limit: For results/passages — page size (default 100).
            corpus: Corpus name.
        """
        self._ensure_loaded(corpus)
        from cfabric_mcp.tools import search as cf_search

        return cf_search(
            template=template,
            return_type=return_type,
            aggregate_features=aggregate_features,
            group_by_section=group_by_section,
            top_n=top_n,
            limit=limit,
            corpus=corpus,
        )

    def search_continue(
        self,
        cursor_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> dict:
        """Continue paginated search using a cursor ID from search_advanced."""
        from cfabric_mcp.tools import search_continue as cf_search_continue

        return cf_search_continue(cursor_id=cursor_id, offset=offset, limit=limit)

    def search_comparative(
        self,
        template_hebrew: str,
        template_greek: str,
        return_type: str = "count",
        limit: int = 50,
    ) -> dict:
        """Search same or adapted pattern across both corpora.

        Args:
            template_hebrew: Search template for Hebrew corpus.
            template_greek: Search template for Greek corpus.
            return_type: "count" or "statistics" (most useful for comparison).
            limit: Max results per corpus.
        """
        from cfabric_mcp.tools import search as cf_search

        results = {}
        for corpus_id, template in [
            ("hebrew", template_hebrew),
            ("greek", template_greek),
        ]:
            try:
                self._ensure_loaded(corpus_id)
                results[corpus_id] = cf_search(
                    template=template,
                    return_type=return_type,
                    limit=limit,
                    corpus=corpus_id,
                )
            except Exception as e:
                results[corpus_id] = {"error": str(e)}
        return {"comparison": results}

    def list_edge_features(self, corpus: str = "hebrew") -> list[dict]:
        """List available edge features for a corpus."""
        api = self._ensure_loaded(corpus)
        result = []
        for name in api.Eall():
            eobj = api.Es(name)
            if eobj is None:
                continue
            meta = getattr(eobj, "meta", {}) or {}
            has_values = getattr(eobj, "doValues", False)
            result.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "has_values": has_values,
                }
            )
        return result

    def get_edge_features(
        self,
        node: int,
        edge_feature: str,
        direction: str = "from",
        corpus: str = "hebrew",
    ) -> dict:
        """Get edges for a node using a specific edge feature.

        Args:
            node: Node ID.
            edge_feature: Edge feature name.
            direction: "from" (outgoing) or "to" (incoming).
            corpus: Corpus name.
        """
        api = self._ensure_loaded(corpus)
        eobj = api.Es(edge_feature)
        if eobj is None:
            return {"error": f"Edge feature '{edge_feature}' not found"}

        if direction == "to":
            edges = eobj.t(node)
        else:
            edges = eobj.f(node)

        if edges is None:
            edges = set()

        has_values = getattr(eobj, "doValues", False)
        results = []
        for edge in edges:
            if has_values and isinstance(edge, tuple):
                target_node, value = edge
            else:
                target_node = edge
                value = None

            otype = api.F.otype.v(target_node)
            section = api.T.sectionFromNode(target_node)
            entry: dict[str, Any] = {
                "node": int(target_node),
                "type": otype,
                "text": api.T.text(target_node),
                "section": {
                    "book": section[0] if len(section) > 0 else "",
                    "chapter": section[1] if len(section) > 1 else 0,
                    "verse": section[2] if len(section) > 2 else 0,
                },
            }
            if value is not None:
                entry["value"] = str(value)
            results.append(entry)

        return {
            "node": node,
            "edge_feature": edge_feature,
            "direction": direction,
            "source_type": api.F.otype.v(node),
            "edges": results,
        }

    def compare_feature_distribution(
        self,
        feature: str,
        sections: list[dict],
        node_type: str = "word",
        top_n: int = 20,
    ) -> dict:
        """Compare feature value distributions across sections.

        Args:
            feature: Feature name (e.g. "sp", "vs").
            sections: List of dicts with book (and optionally chapter, corpus).
            node_type: Object type to count (default "word").
            top_n: Max values per distribution.
        """
        from cfabric_mcp.tools import search as cf_search

        results = {}
        for sec in sections:
            corpus = sec.get("corpus", "hebrew")
            book = sec["book"]
            chapter = sec.get("chapter")

            if chapter:
                template = (
                    f"book book={book}\n  chapter chapter={chapter}\n    {node_type}\n"
                )
            else:
                template = f"book book={book}\n  {node_type}\n"

            self._ensure_loaded(corpus)
            stats = cf_search(
                template=template,
                return_type="statistics",
                aggregate_features=[feature],
                top_n=top_n,
                corpus=corpus,
            )

            label = f"{book}" + (f" {chapter}" if chapter else "") + f" ({corpus})"
            results[label] = stats

        return {"feature": feature, "comparison": results}
