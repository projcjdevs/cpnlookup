def generate_mermaid_graph(chunks: list) -> str:
    """Generates a Mermaid.js flowchart string based on function calls."""
    lines = ["graph TD"]
    
    all_names = {c['name'] for c in chunks}
    
    for c in chunks:
        source = c.get('source_code', '')
        caller = c['name']

        caller_id = caller.replace(".", "_").replace(" ", "_")

        lines.append(f"    {caller_id}[{caller}]")

        for target in all_names:
            if target == caller:
                continue

            if f"{target}(" in source:
                target_id = target.replace(".", "_").replace(" ", "_")
                lines.append(f"    {caller_id} --> {target_id}")
    
    return "\n".join(lines)