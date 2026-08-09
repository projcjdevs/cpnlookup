# cpnlookup

<p>

  <img src="https://img.shields.io/badge/version-3.0.0--beta-562bc2?style=flat-square" alt="Version 3.0.0-beta"/>

  <img src="https://img.shields.io/badge/codename-Forge-7c3aed?style=flat-square" alt="Forge"/>

  <img src="https://img.shields.io/badge/python-3.9+-3572a5?style=flat-square" alt="Python 3.9+"/>

  <img src="https://img.shields.io/badge/license-MIT-2d9e75?style=flat-square" alt="MIT License"/>

  <img src="https://img.shields.io/badge/status-work%20in%20progress-b45309?style=flat-square" alt="Work in Progress"/>

</p>

```
 ██████╗██████╗ ███╗   ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗
██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝
╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║
 ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝
```

*Smarter retrieval. Deeper context. Persistent memory.*

**cpnlookup** is a local-first CLI tool for indexing and querying GitHub repositories through natural language — without ever running `git clone`. It combines FAISS vector similarity search, BM25 sparse retrieval, cross-encoder reranking, and a SQLite-backed static call graph to deliver a multi-stage Hybrid RAG pipeline that understands both the semantic meaning and structural relationships of a codebase, entirely on your own machine.

> This project is under active development. See [Known Limitations](#known-limitations) and [Roadmap](#roadmap) for current status.

---

## Features

**Remote indexing** — Fetches repository file trees and raw content via the GitHub API. Supports Python source files and Markdown documentation.

**Logic-aware chunking** — Uses Python's `ast` module to parse source files into meaningful units (functions and classes) rather than splitting on arbitrary character counts or line boundaries.

**Static call graph construction** — Performs static analysis to identify function invocations and stores a directed call graph in SQLite. Used at query time to expand retrieval context with structurally adjacent code.

**Hybrid RAG retrieval** — Combines BM25 keyword retrieval with FAISS vector similarity search. Results from both retrieval methods are merged before being passed through a cross-encoder reranker, which scores each candidate against the query jointly and re-sorts the final context.

**Cross-encoder reranking** — Scores retrieved candidates against the query after the initial retrieval stage, filtering and prioritizing the most relevant chunks before they reach the LLM.

**Graph-aware context expansion** — Retrieved chunks can be expanded through their callers and callees in the static call graph, restoring structural context that semantic retrieval alone may miss.

**Incremental indexing** — File content is hashed on index. Re-indexing only fetches, chunks, and embeds files that have actually changed. Unchanged files reuse their stored embedding vectors.

**Write consistency** — An `index_status` flag is written to SQLite at the start of every indexing run and cleared only on clean completion. Interrupted runs are detected and reported on next use.

**Persistent conversation memory** — Q&A history is stored separately in `~/.cpnlookup/memory.db`, allowing conversation history to survive `lookup drop`. The last four turns are automatically injected into the LLM prompt for subsequent questions.

**Local inference** — All inference is handled by a locally running Ollama instance (default: Mistral). No source code, queries, or embeddings leave your machine.

**Global registry** — A registry of all locally indexed repositories is maintained, allowing management of multiple indexes from a single interface.

---

## Architecture

```
GitHub API  (file tree + SHAs — one call)
    |
    v
Change detection  (compare SHAs against stored file_hash in SQLite)
    |
    |-- Unchanged files  -->  reuse stored embedding BLOBs
    |-- New / changed    -->  fetch content  -->  chunk  -->  embed
    v
AST / Regex Chunker
    |-- Python   -->  ast module       -->  function / class chunks
    |-- Markdown -->  regex headers     -->  section chunks
    v
Embedding Model  (sentence-transformers/all-MiniLM-L6-v2)
    |
    +---> Embedding BLOBs stored in SQLite  (chunks.embedding)
    +---> FAISS index rebuilt from all BLOBs  (faiss.index)
    +---> Static call graph  (SQLite: graph_nodes, graph_edges)
    |
    v
Query pipeline
    |-- BM25 search       (keyword-based retrieval)
    |-- FAISS search      (dense vector retrieval)
    |-- Result merge      (combine sparse + dense candidates)
    |-- Cross-encoder     (joint query-candidate relevance scoring)
    |-- Graph expansion   (callers + callees, bidirectional)
    |-- Memory injection  (last 4 conversation turns)
    v
Local LLM via Ollama  -->  Response
    |
    v
Persistent memory  (~/.cpnlookup/memory.db)
```

Each indexed repository produces a `.cpnlookup/` directory containing `index.db` and `faiss.index`. A global registry at `~/.cpnlookup/registry.json` tracks all indexes on the local system.

Conversation memory is stored separately at `~/.cpnlookup/memory.db` and is not removed when an individual repository index is dropped.

---

## Installation

**Prerequisites:** Python 3.9+, [Ollama](https://ollama.ai) running locally, and a GitHub Personal Access Token (Classic) with `repo` scope.

```bash
ollama pull mistral
pip install cpnlookup
lookup auth <your_github_token>
```

---

## Usage

```bash
lookup                             # Welcome screen and quick start
lookup desc                        # Project info, version history, acknowledgements
lookup profile <username>          # Browse a user's repositories
lookup init <username>/<repo>      # Index a repository
lookup ask "How does X work?"      # Query the codebase in natural language
lookup functions                   # List all indexed functions and classes
lookup indexed                     # List all locally indexed repositories
lookup config <key> <value>        # Configure model or search depth
lookup clone <username>/<repo>     # Git clone a repository
lookup drop                        # Remove the local index for the current repository
lookup history                     # View saved conversation history
lookup forget                      # Remove saved conversation history
```

---

## Known Limitations

The following limitations are understood, documented for transparency, and tracked in the roadmap.

**Call graph accuracy.** The call graph is built via name-only static analysis. When a call to `process()` is detected, an edge is recorded to any indexed function named `process` regardless of module or class. Codebases with common function names will produce false edges. Dynamic dispatch — `getattr`, decorator-wrapped functions, factory patterns — is invisible to this analysis. The graph is a useful structural approximation, not a semantically precise call graph.

**Language support.** Only Python source files and Markdown documentation are supported. All other file types are skipped during indexing. Multi-language support via Tree-sitter remains planned.

**Cold-start latency.** Commands that invoke the embedding model still carry a load cost on first use per session. Lazy imports (shipped in v1.2.2) eliminated startup lag for non-model commands; a persistent model daemon to eliminate the remaining load cost is planned for a future release.

**FAISS scalability.** `IndexFlatL2` performs exhaustive linear search. For repositories producing more than roughly 10,000 indexed chunks, query latency will degrade. Migration to `IndexIVFFlat` remains planned for a future release.

**Conversation memory scope.** Only the last four conversation turns are automatically injected into the LLM prompt. Older turns remain stored in `~/.cpnlookup/memory.db` but are not automatically included in the active context.

---

## Roadmap

### Version 2.0 — Forge

| Item                          | Status  | Description                                                                                                                                    |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Write consistency             | Shipped | `index_status` flag prevents corrupt index state on interrupted runs.                                                                          |
| Incremental indexing          | Shipped | SHA-based change detection. Only new or changed files are re-fetched, re-chunked, and re-embedded. Embedding BLOBs stored in SQLite for reuse. |
| Batched embedding             | Shipped | All chunks encoded in a single batched `model.encode()` call.                                                                                  |
| Pre-index filtering           | Shipped | Build artifacts, compiled output, and cache directories excluded. Test files retained.                                                         |
| Bidirectional graph traversal | Shipped | Callers and callees both included in context expansion.                                                                                        |
| Persistent model daemon       | Planned | Background process keeps embedding model warm between commands, eliminating remaining cold-start cost.                                         |

### Version 3.0 — Intelligence and Retrieval

| Item                        | Status  | Description                                                                                                                           |
| --------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| BM25 sparse retrieval       | Shipped | Keyword-based retrieval pass merged with FAISS dense search. Improves recall on exact symbol name queries.                            |
| Cross-encoder reranking     | Shipped | Post-retrieval scoring of each candidate chunk against the query jointly. Filters low-relevance context before it reaches the LLM.    |
| Conversation memory         | Shipped | Persistent Q&A history stored separately from repository indexes. The last four turns are automatically injected into the LLM prompt. |
| Graph-aware retrieval       | Shipped | Retrieved candidates are expanded through their callers and callees to restore structural context.                                    |
| Tree-sitter parser          | Planned | Multi-language support — JS, TS, Rust, Go, C++. Replaces `ast`-based chunking.                                                        |
| FAISS scalability upgrade   | Planned | Migrate from `IndexFlatL2` to `IndexIVFFlat` with product quantization for large repositories.                                        |
| Hierarchical indexing depth | Planned | User-selectable modes: "focused" (main files, faster) and "comprehensive" (full context, slower).                                     |
| Mermaid.js visualisation    | Planned | Visual call graph diagrams generated from `graph_edges` data.                                                                         |
| Local LLM upgrade           | Planned | Evaluate `qwen2.5-coder` as an alternative default for improved code comprehension.                                                   |

---

## Version History

| Version     | Codename           | Highlights                                                                                                          |
| ----------- | ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| v1.0.0      | —                  | Initial release. Hybrid RAG pipeline, GitHub API indexing, AST chunking, FAISS search, Ollama inference.            |
| v1.1.1      | Optimized Indexing | Pre-index filtering, batched embedding, corrected runtime dependency declarations.                                  |
| v1.2.2      | Lazy Imports       | Heavy ML imports moved inside command functions. Non-model commands start instantly.                                |
| v2.0.0      | Forge              | Write consistency, incremental indexing, embedding BLOB storage, bidirectional graph traversal, `faiss_id` mapping. |
| v3.0.0-beta | Flux                  | BM25 + FAISS hybrid retrieval, cross-encoder reranking, graph-aware retrieval, and persistent conversation memory.  |

---

## Privacy

No source code, query text, or generated embeddings are transmitted to any external service. Embedding generation runs locally via `sentence-transformers`. Inference is handled by your local Ollama instance. The GitHub API is used only to fetch the content of repositories you explicitly choose to index.

Conversation history is stored locally in `~/.cpnlookup/memory.db` and is not transmitted to any external service except as part of the locally generated LLM prompt to your Ollama instance.

---

## Acknowledgements

Developed by [@projcjdevs](https://github.com/projcjdevs).

Special thanks to [@2nieGarcia](https://github.com/2nieGarcia), [@renzv-compsci](https://github.com/renzv-compsci), and [@NIghtIngale340](https://github.com/NIghtIngale340) for feedback and testing.
