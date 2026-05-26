from cpnlookup.utils.config import get_global_dir

def get_model():
    """Lazy loads the transformer model."""
    from sentence_transformers import SentenceTransformer
    global_models = get_global_dir() / "models"
    return SentenceTransformer('all-MiniLM-L6-v2', cache_folder=str(global_models))

def embed_chunks(chunks: list):
    """Batched encoding logic."""
    model = get_model()
    
    texts = [
        f"File: {c['file_path']}\nName: {c['name']}\nType: {c['chunk_type']}\nDoc: {c['docstring']}" 
        for c in chunks
    ]
    
    embeddings = model.encode(
        texts, 
        batch_size=32, 
        show_progress_bar=True
    )
    
    return embeddings