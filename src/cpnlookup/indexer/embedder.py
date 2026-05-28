from sentence_transformers import SentenceTransformer
from cpnlookup.utils.config import get_global_dir

def get_model():
    """Loads/Downloads the all-MiniLM-L6-v2 model locally."""
    global_models = get_global_dir() / "models"
    return SentenceTransformer('all-MiniLM-L6-v2', cache_folder=str(global_models))

def embed_chunks(chunks: list):
    """Converts code chunks into a matrix of embeddings using batched processing."""
    model = get_model()

    texts = [
        f"File: {c['file_path']}\nName: {c['name']}\nType: {c['chunk_type']}\n"
        f"Doc: {c['docstring']}\nCode: {c['source_code']}"
        for c in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings