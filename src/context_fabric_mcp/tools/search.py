"""MCP tools for linguistic search."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from context_fabric_mcp.cf_engine import CFEngine


def register(mcp: FastMCP, engine: CFEngine) -> None:
    @mcp.tool()
    def search_words(
        corpus: str = "hebrew",
        book: str | None = None,
        chapter: int | None = None,
        features: dict[str, str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search for words matching morphological feature constraints.

        Example features for Hebrew: {"sp": "verb", "vs": "hif", "vt": "perf"}
        Example features for Greek: {"sp": "verb", "tense": "aorist"}

        Common Hebrew feature names:
        - sp: part of speech (verb, subs, prep, adjv, advb, conj, art, prps, prde, prin, intj, nega, inrg, nmpr)
        - vs: verbal stem (qal, nif, piel, pual, hif, hof, hit, etc.)
        - vt: verbal tense (perf, impf, wayq, impv, infa, infc, ptca, ptcp)
        - gn: gender (m, f)
        - nu: number (sg, pl, du)
        - ps: person (p1, p2, p3)
        - st: state (a=absolute, c=construct, e=emphatic)
        - language: (Hebrew, Aramaic)

        Args:
            corpus: "hebrew" or "greek"
            book: Limit search to a specific book (e.g. "Genesis")
            chapter: Limit search to a specific chapter (requires book)
            features: Dict of feature name -> value constraints
            limit: Max results to return (default 100)
        """
        return engine.search_words(corpus, book, chapter, features, limit)

    @mcp.tool()
    def search_constructions(
        template: str,
        corpus: str = "hebrew",
        limit: int = 50,
    ) -> list[dict]:
        """Search for structural/syntactic patterns using Text-Fabric search templates.

        Templates express hierarchical containment through indentation.
        Each line specifies an object type and optional feature constraints.
        Indented lines are contained within the parent line above them.

        Example templates for Hebrew:

        1. Find nominal clauses with a predicate phrase containing a verb:
           clause typ=NmCl
             phrase function=Pred
               word sp=verb

        2. Find clauses where a verb is followed by a proper noun:
           clause
             word sp=verb
             < word sp=nmpr

        3. Find infinitive constructs in construct phrases:
           phrase typ=PP
             word sp=prep
             word sp=verb vt=infc

        4. Find all participles in Genesis:
           book book=Genesis
             word sp=verb vt=ptca

        Common object types (Hebrew): word, phrase, clause, sentence, book, chapter, verse
        Common phrase features: typ (NP/VP/PP/CP), function (Subj/Objc/Pred), det, rela
        Common clause features: typ, kind (NC/VC), rela, txt, domain
        The '<' operator means "followed by" (adjacency).

        Args:
            template: Text-Fabric search template string
            corpus: "hebrew" or "greek"
            limit: Max results to return (default 50)
        """
        return engine.search_constructions(template, corpus, limit)
