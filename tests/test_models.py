"""Embedder, reranker and Ollama client with fakes: no model files, no network."""
import hashlib
import json
import numpy as np
import pytest
from cpnlookup import llm
from cpnlookup.indexer import embedder

class Enc:
    def __init__(self, ids, width): self.ids = ids + [0] * (width - len(ids)); self.attention_mask = [1] * len(ids) + [0] * (width - len(ids)); self.type_ids = [0] * width

class FakeTok:
    """One token per character; pads each batch to its longest item."""
    def encode_batch(self, inputs):
        ids = [[ord(ch) for ch in (t if isinstance(t, str) else "".join(t))] for t in inputs]
        return [Enc(i, max(map(len, ids))) for i in ids]

class FakeSess:
    """Hidden state = token id in every dimension, with garbage on padding that pooling must ignore."""
    def __init__(self, dim): self.dim, self.batches = dim, []
    def get_inputs(self): return [type("I", (), {"name": n}) for n in ("input_ids", "attention_mask")]
    def run(self, _, feed):
        assert set(feed) == {"input_ids", "attention_mask"}                         # token_type_ids dropped when the model has no such input
        ids = feed["input_ids"].astype(np.float32); self.batches.append(ids.shape)
        return [np.where(feed["attention_mask"][..., None] == 1, ids[..., None] + np.arange(self.dim), 999.0).astype(np.float32)]

def test_embed_texts_pools_real_tokens_normalises_and_keeps_order(monkeypatch):
    sess = FakeSess(embedder.DIM); monkeypatch.setattr(embedder, "_load", lambda kind: (FakeTok(), sess))
    texts = ["b", "aaaaaaaa", "cc", "a"]
    out = embedder.embed_texts(texts, batch_size=2)
    assert out.shape == (4, embedder.DIM) and out.dtype == np.float32 and np.allclose(np.linalg.norm(out, axis=1), 1)
    assert sess.batches == [(2, 8), (2, 1)]                                         # sorted by length: long texts batched together
    expected = lambda t: (v := ord(t[0]) + np.arange(embedder.DIM, dtype=np.float32)) / np.linalg.norm(v)
    for row, t in zip(out, texts): assert np.allclose(row, expected(t), atol=1e-6), t   # padding ignored, original order restored
    assert embedder.embed_texts([]).shape == (0, embedder.DIM)

def test_rerank_scores_and_order(monkeypatch):
    from cpnlookup.retrieval import search
    class Sess(FakeSess):
        def run(self, _, feed): return [feed["attention_mask"].sum(1, keepdims=True).astype(np.float32)]   # longer pair = higher score
    monkeypatch.setattr(embedder, "_load", lambda kind: (FakeTok(), Sess(1)))
    cands = [{"id": i, "file_path": "a.py", "name": n, "source_code": s} for i, (n, s) in enumerate([("short", "x"), ("long", "x" * 50), ("mid", "x" * 9)])]
    assert [c["name"] for c in search.rerank("q", cands, 2)] == ["long", "mid"]
    assert embedder.rerank_scores([("q", "t"), ("q", "tt")]) == [2.0, 3.0]

def test_model_file_prefers_pinned_snapshot_and_checks_downloads(isolated, monkeypatch):
    import huggingface_hub
    repo, rev = "org/name", "abc123"
    snap = embedder.models_dir() / "models--org--name" / "snapshots" / rev; snap.mkdir(parents=True); (snap / "tokenizer.json").write_text("{}")
    def no_network(*a, **kw): raise AssertionError("the pinned file is on disk; no download expected")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    assert embedder.model_file(repo, rev, "tokenizer.json") == str(snap / "tokenizer.json")
    seen = {}
    def fake_download(repo_id, name, revision, cache_dir): seen.update(revision=revision, cache_dir=cache_dir); p = snap / name; p.write_bytes(b"tampered"); return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    good = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(embedder.ModelError, match="integrity check"): embedder.model_file(repo, rev, "model.onnx", "0" * 64)
    assert not (snap / "model.onnx").exists() and seen == {"revision": rev, "cache_dir": str(embedder.models_dir())}
    assert embedder.model_file(repo, rev, "model.onnx", good) == str(snap / "model.onnx")
    def offline(*a, **kw): raise OSError("getaddrinfo failed\nlong traceback text")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(embedder.ModelError, match="downloaded once.*Details: getaddrinfo failed$"): embedder.model_file(repo, rev, "other.bin")

def test_models_are_pinned_to_full_commits_and_hashes():
    for repo, rev, max_len, sha, tok_sha in embedder.MODELS.values(): assert "/" in repo and len(rev) == 40 and len(sha) == len(tok_sha) == 64 and max_len in (256, 512)

@pytest.mark.parametrize("env,url", [
    (None, "http://127.0.0.1:11434"), ("", "http://127.0.0.1:11434"), ("0.0.0.0", "http://127.0.0.1:11434"), ("0.0.0.0:8080", "http://127.0.0.1:8080"),
    (":9000", "http://127.0.0.1:9000"), ("localhost:5", "http://localhost:5"), ("[::]:7", "http://[::1]:7"), ("192.168.1.5", "http://192.168.1.5:11434"), ("http://box:1234/", "http://box:1234"),
    ("https://ollama.example.com", "https://ollama.example.com:443"), ("[::1]:11434", "http://[::1]:11434")])
def test_ollama_host_forms(monkeypatch, env, url):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    if env is not None: monkeypatch.setenv("OLLAMA_HOST", env)
    assert llm.host() == url

@pytest.mark.parametrize("env", ["ftp://x:1", "host:notaport", "http://x:99999999"])
def test_ollama_host_rejects(monkeypatch, env):
    monkeypatch.setenv("OLLAMA_HOST", env)
    with pytest.raises(llm.OllamaError, match="not a valid address"): llm.host()
    assert llm.check_ollama() is False

class Reply:
    def __init__(self, status=200, data=None, lines=None, text=""): self.status_code, self.ok, self._data, self._lines, self.text = status, status < 400, data, lines or [], text
    def json(self):
        if self._data is None: raise ValueError("no json")
        return self._data
    def iter_lines(self): yield from self._lines
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_ensure_ready_names_the_exact_fix(monkeypatch):
    import requests, shutil
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    def down(*a, **kw): raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", down); monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(llm.OllamaError, match="not installed.*ollama.com"): llm.ensure_ready("mistral")
    monkeypatch.setattr(shutil, "which", lambda name: "C:/ollama.exe")
    with pytest.raises(llm.OllamaError, match="not running at http://127.0.0.1:11434.*ollama serve"): llm.ensure_ready("mistral")
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.9"); monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(llm.OllamaError, match="not running at http://10.0.0.9:11434"): llm.ensure_ready("mistral")   # remote: never "not installed"
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Reply(data={"models": [{"name": "mistral:latest"}, {"name": "qwen2.5-coder:3b"}]}))
    llm.ensure_ready("mistral"); llm.ensure_ready("mistral:latest"); llm.ensure_ready("qwen2.5-coder:3b")
    with pytest.raises(llm.OllamaError, match="ollama pull llama3"): llm.ensure_ready("llama3")
    for bad in (Reply(data=None), Reply(status=500, data={}), Reply(data=["x"])):
        monkeypatch.setattr(requests, "get", lambda *a, bad=bad, **kw: bad)
        with pytest.raises(llm.OllamaError, match="did not answer like an Ollama server"): llm.ensure_ready("mistral")

def lines(*parts): return [json.dumps(p).encode() for p in parts]

def test_stream_yields_pieces_strips_control_characters_and_reports_failures(monkeypatch):
    import requests
    sent = {}
    def post(url, json, stream, timeout): sent.update(url=url, **json); return reply[0]
    monkeypatch.setattr(requests, "post", post); monkeypatch.setenv("OLLAMA_HOST", "box:1")
    reply = [Reply(lines=lines({"response": "Hel"}, {"response": "lo \x1b]52;c;evil\x07\x1b[2J\r\n\tok"}, {"response": "", "done": True}))]
    assert "".join(llm.stream_ollama("p", model="m", num_ctx=4096)) == "Hello ]52;c;evil[2J\n\tok"
    assert sent == {"url": "http://box:1/api/generate", "model": "m", "prompt": "p", "stream": True, "options": {"num_ctx": 4096}}
    for bad, match in ((Reply(lines=lines({"response": "partial"})), "before the answer was complete"),
                       (Reply(lines=lines({"response": "a"}, {"error": "out of memory"})), "Ollama stopped: out of memory"),
                       (Reply(lines=[b"{not json"]), "could not be read"),
                       (Reply(status=404, data={"error": "model 'm' not found"}), "404: model 'm' not found. Try: ollama pull m"),
                       (Reply(status=500, data=None, text="boom"), "500: boom")):
        reply[0] = bad
        with pytest.raises(llm.OllamaError, match=match): list(llm.stream_ollama("p", model="m"))
    def dropped(*a, **kw): raise requests.ConnectionError("reset")
    monkeypatch.setattr(requests, "post", dropped)
    with pytest.raises(llm.OllamaError, match="Lost the connection to Ollama at http://box:1"): list(llm.stream_ollama("p"))
