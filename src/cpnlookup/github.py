import base64
import re
from datetime import datetime
from cpnlookup.config import get_github_token

API = "https://api.github.com"
TIMEOUT = 20
INDEXED_EXTS = (".py", ".md")
SKIP_DIRS = {"__pycache__", "node_modules", ".venv", "venv", "env", "dist", "build", ".git", ".github",
             "obj", "bin", ".vs", ".idea", ".vscode", ".tox", ".mypy_cache", ".pytest_cache", "site-packages"}

class GitHubError(Exception): pass

def should_skip_file(path: str) -> bool:
    """True for files outside INDEXED_EXTS or inside build/vendor/cache directories."""
    *dirs, name = path.lower().split("/")
    return any(d in SKIP_DIRS or d.endswith(".egg-info") for d in dirs) or not name.endswith(INDEXED_EXTS)

def parse_repo(text: str) -> str:
    """Accepts owner/repo, github.com/owner/repo or a clone URL; returns 'owner/repo'."""
    m = re.fullmatch(r"(?:(?:https?://)?(?:www\.)?github\.com[/:]|git@github\.com:)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?", text.strip())
    if not m: raise GitHubError(f"'{text}' is not a repository. Use the form owner/repo.")
    return f"{m.group(1)}/{m.group(2)}"

def session():
    import requests
    s = requests.Session(); s.headers["Accept"] = "application/vnd.github+json"
    token = get_github_token()
    if token: s.headers["Authorization"] = f"Bearer {token}"
    return s

def _get(s, url: str, what: str = "resource", **kw):
    import requests
    try: r = s.get(url, timeout=TIMEOUT, **kw)
    except requests.RequestException as e: raise GitHubError(f"Could not reach GitHub: {e}") from None
    if r.ok: return r.json()
    try: msg = r.json().get("message", "")
    except ValueError: msg = r.text[:200]
    if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
        reset = datetime.fromtimestamp(int(r.headers.get("X-RateLimit-Reset", "0"))).strftime("%H:%M")
        hint = "" if "Authorization" in s.headers else " Run lookup auth <token> to raise the limit to 5,000 requests/hour."
        raise GitHubError(f"GitHub API rate limit reached; it resets at {reset}.{hint}")
    if r.status_code == 401: raise GitHubError("GitHub rejected the saved token. Run lookup auth <token> with a valid token.")
    if r.status_code == 404: raise GitHubError(f"Not found: {what}. Check the name, or run lookup auth if it's private.")
    raise GitHubError(f"GitHub API error {r.status_code}: {msg}")

def fetch_repo_tree(s, repo: str):
    """Returns ([{path, sha, size}], truncated) for indexable files on the default branch. No content."""
    branch = _get(s, f"{API}/repos/{repo}", what=f"repository '{repo}'").get("default_branch", "main")
    data = _get(s, f"{API}/repos/{repo}/git/trees/{branch}", what=f"branch '{branch}'", params={"recursive": "1"})
    items = [{"path": i["path"], "sha": i["sha"], "size": i.get("size", 0)}
             for i in data.get("tree", []) if i["type"] == "blob" and not should_skip_file(i["path"])]
    return items, bool(data.get("truncated"))

def fetch_blob(s, repo: str, sha: str):
    """Returns the blob's UTF-8 text, or None for binary / non-UTF-8 content."""
    data = _get(s, f"{API}/repos/{repo}/git/blobs/{sha}", what=f"blob {sha[:7]}")
    try: return base64.b64decode(data.get("content", "")).decode("utf-8")
    except (ValueError, UnicodeDecodeError): return None

def get_user_repos(username: str, limit=20) -> list:
    """Most recently updated repos of a user or org; limit=None fetches every page."""
    s, repos, page = session(), [], 1
    while True:
        batch = _get(s, f"{API}/users/{username}/repos", what=f"user '{username}'",
                     params={"sort": "updated", "per_page": 100 if limit is None else limit, "page": page})
        repos += batch
        if limit is not None or len(batch) < 100: return repos
        page += 1
