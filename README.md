# cpnlookup

<p>
  <img src="https://img.shields.io/badge/version-3.1.0-562bc2?style=flat-square" alt="Version 3.1.0"/>
  <img src="https://img.shields.io/badge/codename-Featherweight-7c3aed?style=flat-square" alt="Featherweight"/>
  <img src="https://img.shields.io/badge/python-3.11+-3572a5?style=flat-square" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-MIT-2d9e75?style=flat-square" alt="MIT License"/>
</p>

```
 ██████╗██████╗ ███╗   ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗
██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝
╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║
 ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝
```

*Fast to start. Answers as they're written. Local by design.*

**cpnlookup** lets you ask questions about any GitHub repository from your terminal, without cloning it. It reads the code through the GitHub API, splits it into functions, classes and documentation sections, and answers with a hybrid retrieval pipeline (vector search, keyword search, a reranker and a static call graph) on top of your local [Ollama](https://ollama.com) model. Your code, questions and embeddings never leave your machine.

```bash
lookup init psf/requests
lookup ask "How are retries configured for a Session?"
lookup chat                 # or keep asking in one session
```

---

## Installation

You need Python 3.11+, [Ollama](https://ollama.com) running locally, and a GitHub personal access token (classic) with the `repo` scope.

```bash
pip install cpnlookup
ollama pull mistral
lookup auth <your_github_token>
lookup                      # welcome screen; shows whether Ollama is running
```

The first `init` downloads the embedding model and the first `ask` downloads the reranker (about 90 MB each) into `~/.cpnlookup/models`. Both run on ONNX Runtime, so the install has no PyTorch. Each download is pinned to a fixed revision and checked against a known checksum. After that, everything runs offline except reading repositories from GitHub.

If Ollama runs on another port or machine, set `OLLAMA_HOST` the same way you would for Ollama itself (for example `OLLAMA_HOST=127.0.0.1:11500`).

---

## Usage

Run `init` inside the directory where you want the index to live; every command then works on that directory's index.

| Command | What it does |
|---|---|
| `lookup` | Welcome screen and Ollama status |
| `lookup help` | Setup guide and the full command list |
| `lookup desc` | About, version history, acknowledgements |
| `lookup --version` | Print the installed version |
| `lookup auth <token>` | Save your GitHub token |
| `lookup profile <user> [--all]` | List a user's or organization's repositories (20 most recent, or all) |
| `lookup init <owner/repo>` | Index a repository into `./.cpnlookup`. Also accepts GitHub URLs. Re-running only processes changed files |
| `lookup ask "<question>"` | Answer a question from the indexed code. The answer appears as it is written, followed by the `file:line` sources it was based on and how long the search and the model took |
| `lookup chat` | Ask several questions in one session. The models load once and earlier turns are remembered. `exit` or Ctrl+C to leave |
| `lookup functions` | List every indexed function, class and documentation section |
| `lookup indexed` | List every index on this machine, with the cpnlookup version that built it and whether `lookup init` is suggested |
| `lookup config [key] [value]` | Show settings, or change one (see below) |
| `lookup clone <owner/repo>` | `git clone` a repository |
| `lookup drop` | Delete this directory's index (conversation memory is kept) |
| `lookup history` | Show past questions and answers for this repository |
| `lookup forget [--all]` | Clear conversation memory for this repository, or for all |

### Settings

Settings are stored per directory in `.cpnlookup/config.json`.

| Key | Default | Meaning |
|---|---|---|
| `model` | `mistral` | Ollama model used for answers |
| `top_k` | `5` | Chunks kept after reranking |
| `max_neighbors` | `3` | Callers and callees added per chunk from the call graph |
| `num_ctx` | `8192` | Context window requested from Ollama; also sets how much code fits in the prompt |
| `rerank` | `on` | `off` skips the cross-encoder: questions start about a second sooner, but the best chunk is found less often (21 of 25 test questions instead of 25 of 25) |

```bash
lookup config model qwen2.5-coder
lookup config num_ctx 16384
```

---

## How it works

```
lookup init owner/repo
  GitHub tree API ── one call: every path + blob SHA
        │
  Change detection ── compare SHAs with the last run; unchanged files are skipped
        │
  Fetch ── download only new or changed .py / .md files
        │
  Chunk ── Python via ast (functions, classes, methods) · Markdown by header
        │
  Store (SQLite .cpnlookup/index.db)
        ├── chunks + FTS5 keyword index
        ├── call graph edges (name( → function or class)
        └── embeddings (all-MiniLM-L6-v2 on ONNX Runtime) → FAISS index

lookup ask "question"
  FAISS (meaning)  ─┐
                    ├─ merge ─ cross-encoder rerank ─ top_k chunks
  FTS5 BM25 (words) ┘                                     │
                                  call graph: + callers and callees
                                                          │
             + last 4 Q&A turns ─ local Ollama model ─ streamed answer + sources
```

- **Hybrid retrieval.** Vector search finds code by meaning ("what handles authentication?"); BM25 keyword search finds exact symbols ("where is `parse_token` defined?"). A cross-encoder then reads the question and each candidate together to pick the most relevant ones.
- **Call-graph context.** For each selected chunk, the functions it calls and the functions that call it are added, within a size budget that fits your model's context window.
- **Incremental indexing.** GitHub blob SHAs tell cpnlookup exactly which files changed, so re-indexing only downloads and embeds those.
- **Safe interruption.** An index is marked incomplete until a run finishes; `ask` refuses to use a half-written index.
- **Conversation memory.** Questions and answers are kept in `~/.cpnlookup/memory.db`, separate from indexes, so they survive `lookup drop`. The last four turns go into each new prompt.

### Files

```
~/.cpnlookup/auth.json        GitHub token
~/.cpnlookup/registry.json    every indexed directory
~/.cpnlookup/memory.db        conversation history
~/.cpnlookup/models/          embedding + reranker models
<dir>/.cpnlookup/index.db     chunks, keyword index, call graph, metadata
<dir>/.cpnlookup/faiss.index  vector index
<dir>/.cpnlookup/config.json  settings for this directory
```

### Upgrading from 3.0

- **Indexes keep working.** 3.1 produces the same embeddings as 3.0, so nothing has to be rebuilt. cpnlookup suggests running `lookup init` once per index to refresh it; that only fetches files that changed.
- **Python 3.11 or newer** is required. On Python 3.10, `pip` keeps installing 3.0.0.
- **Old model files.** The PyTorch model files that 3.0 downloaded are no longer used. You can delete `~/.cpnlookup/models` to remove them; the ONNX files are downloaded again on next use.

### Upgrading from 2.x or 3.0.0b1

The index format changed in 3.0.0. The first `lookup init` after upgrading rebuilds each index from scratch once; after that, re-indexing is incremental again. Conversation history is kept.

---

## Known limitations

- **Languages:** only Python and Markdown files are indexed for now.
- **Generation speed:** cpnlookup reaches the model in about two seconds, but how fast the answer is written depends on your Ollama model and hardware. A smaller model (`lookup config model <name>`) answers faster.
- **Call graph is name-based:** a call to `process()` links to every indexed function named `process`, regardless of module. Dynamic dispatch isn't detected.
- **Large repositories:** vector search is exhaustive (`IndexFlatL2`); beyond roughly 10,000 chunks queries slow down.
- **Memory:** the last four turns are included verbatim; older turns are stored but not summarised.

## Roadmap

| Version | Codename | Focus |
|---|---|---|
| 3.2 | Bridge | cpnlookup as a tool for coding agents: search output for scripts and agents, an MCP server |
| 3.3 | Polyglot | JavaScript, TypeScript, Go, Rust, C/C++ and Java via Tree-sitter |
| 3.4 | Depth | Focused vs. full indexing, indexing local folders, parallel downloads |
| 3.5 | Memory+ | Token-budgeted memory with summaries of older turns |
| 3.6 | Graph | Call-graph export as Mermaid diagrams |
| 3.7 | Model UX | Model picker and built-in answer-quality evaluation |
| 4.0 | Switchboard | OpenAI-compatible local backends, questions across several repositories |

## Version history

| Version | Codename | Highlights |
|---|---|---|
| v1.0.0 | — | GitHub API indexing, AST chunking, FAISS search, SQLite call graph, Ollama answers |
| v1.1.1 | Optimized Indexing | Pre-index filtering, batched embedding, fixed runtime dependencies |
| v1.2.2 | Lazy Imports | Heavy ML imports moved into functions; light commands start instantly |
| v2.0.0 | Forge | Incremental indexing, write consistency, stored embeddings, `faiss_id` mapping, callers + callees |
| v3.0.0 | Flux | Hybrid retrieval (FAISS + FTS5 BM25 + cross-encoder), conversation memory, `history` / `forget`, context budget, new index format, leaner and faster codebase |
| v3.1.0 | Featherweight | Models on ONNX Runtime instead of PyTorch (a question reaches the model in about 2 s instead of 10), streamed answers, `lookup chat`, source citations, `OLLAMA_HOST`, `rerank` setting, clearer Ollama errors |

## Privacy

Embeddings, reranking and answers are computed on your machine. The GitHub API is used only to read the repositories you choose to index, and model files are downloaded once from Hugging Face. Conversation history stays in `~/.cpnlookup/memory.db`.

One exception is yours to make: if you point `OLLAMA_HOST` at another machine, your questions and the retrieved code are sent to that machine.

## License

MIT © 2026 Charles Cabatian. See [LICENSE](https://github.com/projcjdevs/cpnlookup/blob/main/LICENSE).

## Acknowledgements

Developed by [@projcjdevs](https://github.com/projcjdevs). Thanks to [@2nieGarcia](https://github.com/2nieGarcia), [@renzv-compsci](https://github.com/renzv-compsci) and [@NIghtIngale340](https://github.com/NIghtIngale340) for feedback and testing.
