"""Graph expansion and prompt assembly with a size budget."""
from cpnlookup.indexer.storage import CHUNK_COLS, rows_to_chunks

COLS = ", ".join(CHUNK_COLS)
CHARS_PER_CTX_TOKEN = 2.5   # code context gets roughly 60-80% of num_ctx; the rest is instruction, memory and the answer
MEMORY_ANSWER_CHARS = 1000

def expand(conn, chunks: list, max_neighbors: int) -> list:
    """[(label, chunk)]: every match first, then up to max_neighbors callees and callers per match. Dedup by chunk id."""
    out, seen = [], set()
    def add(label, c):
        if c["id"] not in seen: seen.add(c["id"]); out.append((label, c))
    for c in chunks: add("MATCH", c)
    for c in chunks:
        callees = conn.execute(f"SELECT {COLS} FROM chunks WHERE chunk_type != 'documentation' AND id != ? AND name IN "
                               "(SELECT target_name FROM graph_edges WHERE source_id = ?) LIMIT ?", (c["id"], c["id"], max_neighbors))
        callers = conn.execute(f"SELECT {COLS} FROM chunks WHERE id IN (SELECT source_id FROM graph_edges WHERE target_name = ?) LIMIT ?",
                               (c["name"], max_neighbors))
        for n in rows_to_chunks(callees): add("CALLEE", n)
        for n in rows_to_chunks(callers): add("CALLER", n)
    return out

def build_prompt(question: str, items: list, turns: list, num_ctx: int) -> str:
    """Adds context blocks in order until the character budget is spent; the first match is always included."""
    budget, blocks = int(num_ctx * CHARS_PER_CTX_TOKEN), []
    for label, c in items:
        block = f"--- {label}: {c['file_path']}:{c['line_start']} | {c['name']} ---\n{c['source_code']}"
        if blocks and len(block) > budget: continue
        blocks.append(block[:budget]); budget -= len(block)
    memory = "".join(f"Q: {q}\nA: {a[:MEMORY_ANSWER_CHARS]}\n\n" for q, a in turns)
    return ("You are a code assistant. Answer the question using the code context below. "
            "If the context does not contain the answer, say so.\n\nCode context:\n" + "\n\n".join(blocks)
            + (f"\n\nPrevious conversation:\n{memory}" if memory else "") + f"\n\nQuestion: {question}")
