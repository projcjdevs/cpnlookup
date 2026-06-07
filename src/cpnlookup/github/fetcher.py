import requests
import base64
from typing import List, Dict, Tuple
from cpnlookup.utils.config import get_github_token

def should_skip_file(path: str) -> bool:
    p = path.lower()
    if "test" in p:
        return False
    skip_dirs = {
        '__pycache__', 'node_modules', '.venv', 'venv', 'env',
        'dist', 'build', '.egg-info', '.git', '.github',
        'obj', 'bin', '.vs', '.idea', '.vscode'
    }
    if any(d in set(p.split('/')) for d in skip_dirs):
        return True
    skip_exts = {
        '.pyc', '.pyo', '.pyd', '.exe', '.dll', '.so', '.dylib',
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.pdf',
        '.lock', '.lockb', '-lock.json', '.zip', '.tar.gz', '.7z'
    }
    if any(p.endswith(ext) for ext in skip_exts):
        return True
    return False

def _make_session(repo_full_name: str) -> Tuple[requests.Session, str, str, str]:
    token = get_github_token()
    owner, repo = repo_full_name.split('/')
    session = requests.Session()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    session.headers.update(headers)

    repo_res = session.get(f"https://api.github.com/repos/{repo_full_name}")
    if repo_res.status_code == 404:
        raise Exception(f"Repository '{repo_full_name}' not found.")
    branch = repo_res.json().get("default_branch", "main")
    return session, owner, repo, branch

def fetch_repo_tree(repo_full_name: str) -> List[Dict]:
    session, owner, repo, branch = _make_session(repo_full_name)
    tree_res = session.get(
        f"https://api.github.com/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
    )
    tree_data = tree_res.json()
    if "tree" not in tree_data:
        raise Exception(f"Could not fetch file tree: {tree_data.get('message', 'Unknown error')}")

    items = []
    for item in tree_data["tree"]:
        if item["type"] != "blob": continue
        path = item["path"]
        if should_skip_file(path): continue
        if not (path.endswith('.py') or path.lower().endswith('.md')): continue
        items.append({"path": path, "sha": item["sha"], "size": item.get("size", 0)})
    return items

def fetch_file_contents(repo_full_name: str, items: List[Dict]) -> List[Dict]:
    if not items:
        return []
    session, owner, repo, _ = _make_session(repo_full_name)
    results = []
    for item in items:
        blob_res = session.get(
            f"https://api.github.com/repos/{repo_full_name}/git/blobs/{item['sha']}"
        )
        blob_data = blob_res.json()
        content_b64 = blob_data.get("content", "")
        try:
            content = base64.b64decode(content_b64.replace("\n", "")).decode("utf-8")
        except Exception:
            continue
        results.append({"path": item["path"], "sha": item["sha"],
                         "content": content, "size": item["size"]})
    return results