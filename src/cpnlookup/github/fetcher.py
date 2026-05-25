import requests
import base64
from typing import List, Dict
from cpnlookup.utils.config import get_github_token

def should_skip_file(path: str) -> bool:
    """
    Returns True if the file or directory should be ignored.
    Keeps everything with 'test' in the name/path.
    """
    p = path.lower()

    if "test" in p:
        return False

    skip_dirs = {
        '__pycache__', 'node_modules', '.venv', 'venv', 'env', 
        'dist', 'build', '.egg-info', '.git', '.github', 
        'obj', 'bin', '.vs', '.idea', '.vscode'
    }

    path_parts = set(p.split('/'))
    if any(d in path_parts for d in skip_dirs):
        return True

    skip_exts = {
        '.pyc', '.pyo', '.pyd', '.exe', '.dll', '.so', '.dylib', # Compiled
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.pdf', # Media
        '.lock', '.lockb', '-lock.json', '.zip', '.tar.gz', '.7z' # Archives/Locks
    }
    if any(p.endswith(ext) for ext in skip_exts):
        return True
        
    return False

def fetch_repo_files(repo_full_name: str) -> List[Dict]:
    token = get_github_token()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    session = requests.Session()
    session.headers.update(headers)

    repo_url = f"https://api.github.com/repos/{repo_full_name}"
    repo_res = session.get(repo_url)
    
    if repo_res.status_code == 404:
        raise Exception(f"Repository '{repo_full_name}' not found.")
    
    repo_info = repo_res.json()
    default_branch = repo_info.get("default_branch", "main")

    tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{default_branch}?recursive=1"
    tree_res = session.get(tree_url)
    tree_data = tree_res.json()
    
    if "tree" not in tree_data:
        raise Exception(f"Could not fetch file list: {tree_data.get('message', 'Unknown error')}")

    files_to_index = []
    for item in tree_data["tree"]:
        if item["type"] == "blob":
            file_path = item["path"]
            
            # --- V2 Pre-index Filtering ---
            if should_skip_file(file_path):
                continue

            is_python = file_path.endswith('.py')
            is_markdown = file_path.lower().endswith('.md')
            
            if not (is_python or is_markdown):
                continue

            blob_url = item["url"]
            blob_res = session.get(blob_url)
            blob_data = blob_res.json()
            
            content_b64 = blob_data.get("content", "")
            try:
                content_text = base64.b64decode(content_b64.replace("\n", "")).decode('utf-8')
            except:
                continue

            files_to_index.append({
                "path": file_path,
                "content": content_text,
                "size": item.get("size", 0)
            })
            
    return files_to_index