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
        
def get_local_config() -> dict:
    """Reads .cpnlookup/config.json if it exists."""
    cfg_path = Path.cwd() / ".cpnlookup" / "config.json"
    if not cfg_path.exists():
        return {"model": "mistral", "top_k": 5}
    with open(cfg_path, "r") as f:
        return json.load(f)

def save_local_config(config: dict):
    """Saves the config to .cpnlookup/config.json."""
    cfg_path = Path.cwd() / ".cpnlookup" / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)