# cpnlookup

<p>
  <img src="https://img.shields.io/badge/version-1.0.0-562bc2?style=flat-square" alt="Version 1.0.0"/>
  <img src="https://img.shields.io/badge/python-3.9+-3572a5?style=flat-square" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/license-MIT-2d9e75?style=flat-square" alt="MIT License"/>
  <img src="https://img.shields.io/badge/status-work%20in%20progress-b45309?style=flat-square" alt="Work in Progress"/>
</p>

**cpnlookup** is a local-first AI engineering tool for indexing and querying GitHub repositories without cloning them. It combines vector similarity search with static call graph analysis to deliver a Hybrid RAG pipeline that understands both the semantic meaning and structural relationships of a codebase — entirely on your own machine.

> This project is under active development. Version 1.0 is a functional proof-of-concept with known architectural limitations documented below.

---

## Features

**Remote indexing** — Fetches repository file trees and raw content via the GitHub API. Supports Python source files and Markdown documentation.

**Logic-aware chunking** — Uses Python's `ast` module to parse source files into meaningful units (functions and classes) rather than splitting on arbitrary character counts or line boundaries.

**Static call graph construction** — Performs static analysis to identify function invocations and stores a directed call graph in SQLite. This graph is used at query time to expand retrieval context with structurally adjacent code.

**Hybrid RAG retrieval** — Combines FAISS vector similarity search with call graph traversal. Top-k vector matches are retrieved first, then their graph neighbours are included to provide the LLM with architectural context beyond what semantic similarity alone would surface.

**Local inference** — All inference is handled by a locally running Ollama instance (default: Mistral). No source code, queries, or embeddings leave your machine.

**Global registry** — A registry of all locally indexed repositories is maintained, allowing management of multiple indexes from a single interface.

---

## Architecture

```
GitHub API  (file tree + raw content)
    |
    v
AST / Regex Chunker
    |-- Python  --> ast module       --> function / class chunks
    |-- Markdown --> regex headers   --> section chunks
    v
Embedding Model  (sentence-transformers/all-MiniLM-L6-v2)
    |
    +---> FAISS index          (faiss.index)
    +---> Static call graph    (SQLite: index.db / graph_edges)
    |
    v
Query pipeline
    |-- Vector search   (top-k FAISS matches)
    |-- Graph traversal (caller / callee expansion)
    v
Local LLM via Ollama  -->  Response
```

Each indexed repository produces a `.cpnlookup/` directory containing `index.db` and `faiss.index`. A global registry file tracks all indexes on the local system. The CLI is built with `click` and `rich`.

---

## Installation

**Prerequisites:** Python 3.9+, [Ollama](https://ollama.ai) running locally, and a GitHub Personal Access Token (Classic) with `repo` scope.

```bash
ollama pull mistral
pip install cpnlookup
```

> The `pyproject.toml` dependency declaration is being completed as part of the V1 stabilisation pass. If the PyPI install is incomplete, install dependencies directly:
> ```bash
> pip install click rich requests sentence-transformers faiss-cpu numpy
> ```

```bash
lookup auth <your_github_token>
```

---

## Usage

```bash
lookup profile <username>          # Browse a user's repositories
lookup init <username>/<repo>      # Index a repository (fetch, chunk, embed, graph)
lookup ask "How does X work?"      # Query the codebase in natural language
lookup indexed                     # List all locally indexed repositories
lookup functions                   # List all indexed functions and classes
lookup drop                        # Remove the local index for the current repository
```

---

## Known Limitations

Version 1.0 is a proof-of-concept. The limitations below are understood, documented for transparency, and addressed in the roadmap.

**Call graph accuracy.** The call graph is built via name-only static analysis. When a call to `process()` is detected, an edge is recorded to any indexed function named `process` regardless of module or class. Codebases with common function names will produce false edges. Dynamic dispatch patterns — `getattr`, decorator-wrapped functions, factory patterns — are invisible to this analysis. The graph is a useful structural approximation, not a semantically precise call graph.

**Language support.** V1 supports Python and Markdown only. All other file types are skipped during indexing.

**Cold-start latency.** Each CLI command cold-loads the embedding model from disk, adding approximately 8–12 seconds of latency per command. This is a model loading cost, not a retrieval cost.

**No incremental indexing.** Re-indexing performs a full fetch and re-embedding of all files. There is no change detection mechanism. For actively developed repositories this is a significant friction point.

**SQLite and FAISS consistency.** The two stores are written in separate operations with no shared transaction boundary. An interrupted indexing run can produce a permanently inconsistent state that requires a full re-index to recover.

**FAISS scalability.** `IndexFlatL2` performs exhaustive linear search and degrades on repositories producing more than roughly 10,000 indexed chunks.

---

## Roadmap

Items marked `[new]` were identified during architectural review and are not in the original plan.

### Version 2.0 — Performance and Reliability

| Item | Description |
|---|---|
| Persistent model process | Background socket server keeps the embedding model warm between commands, eliminating cold-start latency. |
| Batched embedding | Replace per-chunk `model.encode()` calls with batched encoding — O(n) passes become one vectorised operation. |
| Incremental indexing `[new]` | Store a content hash per file in SQLite. On re-index, only changed files are re-processed. The highest-impact UX improvement in this cycle. |
| Pre-index filtering | Skip build artifacts, compiled output, and auto-generated files. Test files are intentionally retained as executable documentation. |
| Write consistency `[new]` | Two-phase commit pattern for SQLite and FAISS writes with a recoverable state flag to handle interrupted indexing runs. |

### Version 3.0 — Intelligence and Scalability

| Item | Description |
|---|---|
| Tree-sitter parser | Replace `ast`-based chunking with Tree-sitter for multi-language support (JS, TS, Rust, Go, C++). Name-only resolution remains a limitation; full semantic accuracy requires language server integration, planned for a later cycle. |
| BM25 sparse retrieval `[new]` | Keyword-based retrieval pass alongside FAISS dense search. BM25 outperforms dense retrieval on exact symbol name queries; results are merged before context assembly. |
| Cross-encoder reranking `[new]` | Apply a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to the candidate set after hybrid retrieval. Scores each chunk against the query jointly, filtering low-relevance context before it reaches the LLM. |
| FAISS scalability upgrade `[new]` | Migrate from `IndexFlatL2` to `IndexIVFFlat` with product quantization for repositories exceeding ~10,000 chunks. |
| Hierarchical indexing depth | User-selectable modes: "focused" (main source files, faster) and "comprehensive" (full repository context, slower). |
| Conversation memory | Persist and index chat history across sessions. Older turns are summarised rather than retained verbatim to avoid context window bloat. |
| Mermaid.js visualisation | Generate visual call graph diagrams from `graph_edges` data, renderable in any Mermaid-compatible viewer. |
| Local LLM upgrade | Evaluate `qwen2.5-coder` as an alternative default. Code-specialised models consistently outperform general-purpose models on comprehension and synthesis tasks at equivalent parameter counts. |

---

## Privacy

No source code, query text, or generated embeddings are transmitted to any external service. Embedding generation runs locally via `sentence-transformers`. Inference is handled by your local Ollama instance. The GitHub API is used only to fetch the content of repositories you explicitly choose to index.

---

*Developed by [@projcjdevs](https://github.com/projcjdevs)*