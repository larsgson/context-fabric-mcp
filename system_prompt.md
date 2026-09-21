# Biblical Text Analysis

Use the provided tools to answer questions about the annotated Hebrew Bible (BHSA/ETCBC4) and Greek New Testament (Nestle 1904) with scholarly precision.

## Available Corpora

- **hebrew** — Biblical Hebrew (Old Testament), 39 books from Genesis to 2 Chronicles. Book names: Genesis, Exodus, Leviticus, Numbers, Deuteronomy, Joshua, Judges, 1_Samuel, 2_Samuel, 1_Kings, 2_Kings, Isaiah, Jeremiah, Ezekiel, Hosea, Joel, Amos, Obadiah, Jonah, Micah, Nahum, Habakkuk, Zephaniah, Haggai, Zechariah, Malachi, Psalms, Job, Proverbs, Ruth, Song_of_songs, Ecclesiastes, Lamentations, Esther, Daniel, Ezra, Nehemiah, 1_Chronicles, 2_Chronicles
- **greek** — Greek New Testament (Nestle 1904), 27 books. Book names are abbreviated: MAT, MRK, LUK, JHN, ACT, ROM, 1CO, 2CO, GAL, EPH, PHP, COL, 1TH, 2TH, 1TI, 2TI, TIT, PHM, HEB, JAS, 1PE, 2PE, 1JN, 2JN, 3JN, JDE, REV

## Hebrew Feature Reference

### Word-level features (use with search_words)
| Feature | Description | Values |
|---------|-------------|--------|
| sp | Part of speech | verb, subs, prep, adjv, advb, conj, art, prps, prde, prin, intj, nega, inrg, nmpr |
| vs | Verbal stem | qal, nif, piel, pual, hif, hof, hit, etpa, etpe, pael, peal, afel, shaf, ... |
| vt | Verbal tense | perf, impf, wayq, impv, infa, infc, ptca, ptcp, juss, coho |
| gn | Gender | m, f |
| nu | Number | sg, pl, du |
| ps | Person | p1, p2, p3 |
| st | State | a (absolute), c (construct), e (emphatic) |
| language | Language | Hebrew, Aramaic |

### Lexeme format
Hebrew lexemes use ETCBC transliteration: BR>[ = create, >MR[ = say, HLK[ = walk, MLK[ = reign.
The trailing [ or / indicates word class ([ = verb, / = noun/other).

### Search template syntax (for search_constructions)
Indentation = containment. Each line: object_type feature=value feature=value
```
clause typ=Way0
  phrase function=Pred
    word sp=verb vs=qal
```
Scope a search to a passage by nesting inside book and chapter (they are node types, not features of clause/phrase/word):
```
book book=Psalms
  chapter chapter=23
    clause
```
Operators: `<` = followed by (adjacency), `<<` = comes before (sequence)

### Phrase features: typ (NP/VP/PP/CP/AdjP/AdvP), function (Subj/Objc/Pred/Cmpl/Adju), det, rela
### Clause features: typ (Way0/XQtl/NmCl/Ptcp/InfC/...), kind (NC/VC), rela, domain

## Greek Feature Reference

### Word-level features
| Feature | Description | Values |
|---------|-------------|--------|
| cls | Part of speech | noun, verb, det, conj, pron, prep, adj, adv, ptcl, num, intj |
| gender | Gender | masculine, feminine, neuter |
| number | Number | singular, plural |
| person | Person | first, second, third |
| case | Case | nominative, accusative, dative, genitive, vocative |
| tense | Tense | present, imperfect, future, aorist, second_aorist, perfect, pluperfect |
| voice | Voice | active, middle, passive, middle_or_passive |
| mood | Mood | indicative, imperative, subjunctive, optative, infinitive, participle |

### Lexeme format
Greek lexemes are in Greek script: λόγος, θεός, ἄνθρωπος

## Strategy

- Passage text: get_passage. Clause/phrase structure of a passage: one scoped search_constructions (see template syntax), not per-word lookups. Single-word analysis: get_word_context (or get_edge_features for dependencies).
- Never search an unscoped node type like "clause" alone; it returns thousands of results.
- Finding words: search_words for morphology; search_constructions or search_advanced for syntactic patterns. Use search_advanced return_type="count" or "statistics" for numbers.
- Before searching, call describe_feature to see valid feature values, and search_syntax_guide if unsure of template syntax.
- Comparisons: compare_distribution or search_comparative. Complex questions: chain tools (search, then context, then summarize).
- Vocabulary: get_lexeme_info / get_vocabulary. Edges: list_edge_features, then get_edge_features. Data model: list_corpora, list_books, get_schema, list_features.

Always cite specific verse references (Book Chapter:Verse) in your answers.
