import json
from pathlib import Path

def get_global_dir() -> Path:
    global_dir = Path.home() / ".cpnlookup"
    global_dir.mkdir(parents=True, exist_ok=True)
    return global_dir

def save_github_token(token: str) -> None:
    auth_file = get_global_dir() / "auth.json"
    
    data = {"github_token": token}
    with open(auth_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def get_github_token() -> str:
    auth_file = get_global_dir() / "auth.json"
    if not auth_file.exists():
        return ""
        
    with open(auth_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data.get("github_token", "")
        except json.JSONDecodeError:
            return ""