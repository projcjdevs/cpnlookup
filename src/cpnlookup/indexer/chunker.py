import ast
from typing import List, Dict

def chunk_python_code(file_path: str, code: str) -> List[Dict]:
    """
    Parses Python code and extracts functions and classes as individual chunks.
    """
    chunks = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            source = ast.get_source_segment(code, node)
            
            if not source:
                continue

            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            
            docstring = ast.get_docstring(node) or ""

            chunks.append({
                "name": node.name,
                "file_path": file_path,
                "line_start": node.lineno,
                "line_end": getattr(node, 'end_lineno', node.lineno),
                "chunk_type": chunk_type,
                "source_code": source,
                "docstring": docstring
            })
            
    return chunks