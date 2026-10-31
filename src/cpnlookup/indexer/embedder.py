import os
from cpnlookup.config import get_global_dir

EMBED_MODEL = "all-MiniLM-L6-v2"
QUIET_ENV = {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error",
             "TRANSFORMERS_VERBOSITY": "error", "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1", "TOKENIZERS_PARALLELISM": "false"}

def quiet_hf() -> None:
    """Keeps HuggingFace progress bars and warnings from drawing over the Rich spinner."""
    import logging as std_logging, warnings
    for k, v in QUIET_ENV.items(): os.environ.setdefault(k, v)
    std_logging.getLogger("sentence_transformers").setLevel(std_logging.ERROR); warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from transformers.utils import logging
        from huggingface_hub.utils import logging as hub_logging
        logging.set_verbosity_error(); logging.disable_progress_bar(); hub_logging.set_verbosity_error()
    except ImportError: pass

def models_dir() -> str: return str(get_global_dir() / "models")

def load_cached(cls, name: str, **kw):
    """Loads a sentence-transformers model from ~/.cpnlookup/models without network calls; downloads only if missing."""
    quiet_hf()
    try: return cls(name, cache_folder=models_dir(), local_files_only=True, **kw)
    except Exception: return cls(name, cache_folder=models_dir(), **kw)

def get_model():
    """all-MiniLM-L6-v2 (downloads ~90 MB on first use)."""
    from sentence_transformers import SentenceTransformer
    return load_cached(SentenceTransformer, EMBED_MODEL)

def embed_text(c: dict) -> str:
    return f"File: {c['file_path']}\nName: {c['name']}\nType: {c['chunk_type']}\nDoc: {c['docstring']}\nCode: {c['source_code']}"

def embed_chunks(chunks: list):
    """float32 matrix of embeddings, one row per chunk, in order."""
    return get_model().encode([embed_text(c) for c in chunks], batch_size=32, show_progress_bar=False, convert_to_numpy=True)
