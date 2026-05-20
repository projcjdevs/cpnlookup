import ast
import re
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

def chunk_markdown(file_path: str, content: str) -> List[Dict]:
    """Splits markdown files by headers to preserve context."""
    chunks = []
    sections = re.split(r'(^#+\s.*)', content, flags=re.MULTILINE)
    
    current_header = "Introduction"
    for i in range(len(sections)):
        section = sections[i].strip()
        if not section: continue
        
        if section.startswith('#'):
            current_header = section.replace('#', '').strip()
            continue
            
        chunks.append({
            "name": f"README: {current_header}",
            "file_path": file_path,
            "line_start": 0, 
            "line_end": 0,
            "chunk_type": "documentation",
            "source_code": section,
            "docstring": f"Markdown documentation section: {current_header}"
        })

    return chunks