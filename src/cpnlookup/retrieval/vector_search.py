from cpnlookup.indexer.storage import load_faiss_index

def search_chunks(query: str, top_k: int = 5):
    import numpy as np
    from cpnlookup.indexer.embedder import get_model

    index = load_faiss_index()
    if not index:
        return []

    model = get_model()
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, top_k)

    return indices[0].tolist()