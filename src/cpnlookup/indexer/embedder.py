"""MiniLM embeddings and the ms-marco cross-encoder on ONNX Runtime."""
import os
from functools import cache
from pathlib import Path
from cpnlookup.config import get_global_dir

EMBED_ID, DIM = "all-MiniLM-L6-v2", 384     # EMBED_ID is stored in repo_meta.embed_model
# kind -> (Hugging Face repo, pinned commit, max tokens, sha256 of onnx/model.onnx, sha256 of tokenizer.json).
# The pin and the hash mean a changed upstream repository can never swap the model that runs here.
MODELS = {"embed": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", 256,
                    "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452", "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
          "rerank": ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a", 512,
                     "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66")}
QUIET_ENV = {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}

class ModelError(Exception): pass

def models_dir() -> Path: return get_global_dir() / "models"

def model_file(repo: str, rev: str, name: str, sha256: str = "") -> str:
    """Path of a pinned model file under ~/.cpnlookup/models. Downloads it (and checks its hash) only when missing."""
    p = models_dir() / f"models--{repo.replace('/', '--')}" / "snapshots" / rev / name
    if p.exists(): return str(p)
    for k, v in QUIET_ENV.items(): os.environ.setdefault(k, v)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, name, revision=rev, cache_dir=str(models_dir()))
    except Exception as e:
        reason = (str(e).strip().splitlines() or [type(e).__name__])[0]
        raise ModelError(f"Could not download the search model ({repo}). It is downloaded once; after that no connection is needed. Details: {reason}") from None
    if sha256:
        import hashlib
        with open(path, "rb") as f: digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != sha256:
            for f in {os.path.realpath(path), path}:
                if os.path.lexists(f): os.remove(f)
            raise ModelError(f"The downloaded model file from {repo} failed its integrity check and was deleted. Run the command again.")
    return path

@cache
def _load(kind: str):
    """(tokenizer, ONNX session) for 'embed' or 'rerank', loaded once per process."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    repo, rev, max_len, sha, tok_sha = MODELS[kind]; ort.set_default_logger_severity(3)
    files = model_file(repo, rev, "tokenizer.json", tok_sha), model_file(repo, rev, "onnx/model.onnx", sha)
    try: tok, sess = Tokenizer.from_file(files[0]), ort.InferenceSession(files[1], providers=["CPUExecutionProvider"])
    except Exception as e: raise ModelError(f"Could not load the search model: {e} Delete {models_dir()} and run the command again.") from None
    tok.enable_truncation(max_length=max_len); tok.enable_padding()
    return tok, sess

def _run(kind: str, inputs: list):
    """(first model output, attention mask) for one padded batch of texts or (query, text) pairs."""
    import numpy as np
    tok, sess = _load(kind); enc = tok.encode_batch(inputs)
    feed = {"input_ids": [e.ids for e in enc], "attention_mask": [e.attention_mask for e in enc], "token_type_ids": [e.type_ids for e in enc]}
    feed = {k: np.array(v, dtype=np.int64) for k, v in feed.items() if k in {i.name for i in sess.get_inputs()}}
    return sess.run(None, feed)[0], feed["attention_mask"]

def embed_texts(texts: list, batch_size: int = 8):
    """L2-normalised float32 matrix, one row per text in the given order. Batches are sorted by length to limit padding."""
    import numpy as np
    order, out = sorted(range(len(texts)), key=lambda i: -len(texts[i])), np.empty((len(texts), DIM), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]; hidden, mask = _run("embed", [texts[j] for j in idx]); m = mask[..., None].astype(np.float32)
        v = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)                    # mean over real tokens only
        out[idx] = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    return out

def embed_text(c: dict) -> str:
    return f"File: {c['file_path']}\nName: {c['name']}\nType: {c['chunk_type']}\nDoc: {c['docstring']}\nCode: {c['source_code']}"

def embed_chunks(chunks: list):
    """float32 matrix of embeddings, one row per chunk, in order."""
    return embed_texts([embed_text(c) for c in chunks])

def rerank_scores(pairs: list) -> list:
    """Relevance score per (query, text) pair; higher is better.
    One pair per run: padding a batch to its longest pair took twice as long (measured, 16 candidates: 0.93 s vs 0.41 s)."""
    return [float(_run("rerank", [p])[0].reshape(-1)[0]) for p in pairs]
