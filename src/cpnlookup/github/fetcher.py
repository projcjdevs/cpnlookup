import requests
import base64
from typing import List, Dict
from cpnlookup.utils.config import get_github_token

def fetch_repo_files(repo_full_name: str) -> List[Dict]:
    token = get_github_token()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # Use a Session for faster downloads (keeps the connection alive)
    session = requests.Session()
    session.headers.update(headers)

    repo_url = f"https://api.github.com/repos/{repo_full_name}"
    repo_res = session.get(repo_url)
    
    if repo_res.status_code == 404:
        raise Exception(f"Repository '{repo_full_name}' not found. Check the spelling.")
    
    repo_info = repo_res.json()
    default_branch = repo_info.get("default_branch")

    if not default_branch:
        raise Exception("Could not find the default branch.")

    tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{default_branch}?recursive=1"
    tree_res = session.get(tree_url)
    tree_data = tree_res.json()
    
    if "tree" not in tree_data:
        raise Exception(f"Could not fetch file list: {tree_data.get('message', 'Unknown error')}")

    files_to_index = []
    for item in tree_data["tree"]:
        if item["type"] == "blob":
            file_path = item["path"]
            
            # --- START FILTERING LOGIC ---
            # 1. Skip obvious noise first
            if any(file_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.lock', '.json']):
                continue
            if ".github" in file_path or "node_modules" in file_path or "venv" in file_path:
                continue

            # 2. ONLY allow .py and .md files
            # This ensures we don't grab text files, configs, or binaries
            is_python = file_path.endswith('.py')
            is_markdown = file_path.lower().endswith('.md')
            
            if not (is_python or is_markdown):
                continue
            # --- END FILTERING LOGIC ---

            blob_url = item["url"]
            blob_res = session.get(blob_url)
            blob_data = blob_res.json()
            
            content_b64 = blob_data.get("content", "")
            try:
                # Remove newlines before decoding base64
                content_text = base64.b64decode(content_b64.replace("\n", "")).decode('utf-8')
            except:
                continue

            files_to_index.append({
                "path": file_path,
                "content": content_text,
                "size": item.get("size", 0)
            })
            
    return files_to_index