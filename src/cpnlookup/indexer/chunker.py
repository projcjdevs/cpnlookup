import ast
import re
from pathlib import PurePosixPath

HEADER = re.compile(r"^(#{1,6})\s+(.*)")

def chunk_python_code(file_path: str, code: str) -> list:
    """Functions, async functions and classes as chunks (ast.walk, so methods are chunks too)."""
    try: tree = ast.parse(code)
    except (SyntaxError, ValueError): return []
    chunks = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): continue
        source = ast.get_source_segment(code, node)
        if not source: continue
        chunks.append({"name": node.name, "file_path": file_path, "line_start": node.lineno, "line_end": getattr(node, "end_lineno", node.lineno),
                       "chunk_type": "class" if isinstance(node, ast.ClassDef) else "function", "source_code": source, "docstring": ast.get_docstring(node) or ""})
    return chunks

def chunk_markdown(file_path: str, content: str) -> list:
    """One chunk per header section, named '<file stem>: <header>'. Headers inside code fences are ignored."""
    title, lines = PurePosixPath(file_path).stem, content.splitlines()
    chunks, header, start, buf, fence = [], "Introduction", 1, [], False
    def flush(end):
        text = "\n".join(buf).strip()
        if text: chunks.append({"name": f"{title}: {header}", "file_path": file_path, "line_start": start, "line_end": end,
                                "chunk_type": "documentation", "source_code": text, "docstring": f"Markdown documentation section: {header}"})
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")): fence = not fence
        m = None if fence else HEADER.match(line)
        if m: flush(i - 1); header, start, buf = m.group(2).strip().rstrip("#").strip() or header, i, []
        else: buf.append(line)
    flush(len(lines)); return chunks

def chunk_file(file_path: str, content: str) -> list:
    p = file_path.lower()
    if p.endswith(".py"): return chunk_python_code(file_path, content)
    if p.endswith(".md"): return chunk_markdown(file_path, content)
    return []
