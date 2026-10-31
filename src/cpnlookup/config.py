import json
import os
from pathlib import Path

DEFAULTS = {"model": "mistral", "top_k": 5, "num_ctx": 8192, "max_neighbors": 3}
INT_KEYS = {"top_k", "num_ctx", "max_neighbors"}

def get_global_dir() -> Path:
    d = Path.home() / ".cpnlookup"; d.mkdir(parents=True, exist_ok=True); return d

def local_dir() -> Path: return Path.cwd() / ".cpnlookup"

def _read_json(path: Path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError): return default

def _write_json(path: Path, data) -> None: path.write_text(json.dumps(data, indent=4), encoding="utf-8")

def save_github_token(token: str) -> None:
    p = get_global_dir() / "auth.json"; _write_json(p, {"github_token": token})
    try: os.chmod(p, 0o600)
    except OSError: pass

def get_github_token() -> str: return _read_json(get_global_dir() / "auth.json", {}).get("github_token", "")

def get_local_config() -> dict:
    """Per-directory settings from .cpnlookup/config.json merged over DEFAULTS."""
    return {**DEFAULTS, **_read_json(local_dir() / "config.json", {})}

def set_local_config(key: str, value: str):
    """Validates and saves one setting. Raises ValueError with a user-facing message."""
    if key not in DEFAULTS: raise ValueError(f"Unknown setting '{key}'. Valid settings: {', '.join(DEFAULTS)}.")
    if key in INT_KEYS:
        try: value = int(value)
        except ValueError: raise ValueError(f"'{key}' must be a whole number.") from None
        if value < 1: raise ValueError(f"'{key}' must be at least 1.")
    if not local_dir().exists(): raise ValueError("No index in this directory. Run lookup init <owner/repo> first.")
    p = local_dir() / "config.json"; cfg = _read_json(p, {}); cfg[key] = value; _write_json(p, cfg)
    return value

def _same_path(a: str, b: str) -> bool: return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

def update_registry(path: str, repo_name: str, add: bool = True) -> None:
    """Adds or removes a directory in ~/.cpnlookup/registry.json. Paths that differ only in case or form are merged."""
    f = get_global_dir() / "registry.json"
    reg = {k: v for k, v in _read_json(f, {}).items() if not _same_path(k, path)}
    if add: reg[os.path.abspath(path)] = repo_name
    _write_json(f, reg)

def get_registry() -> dict: return _read_json(get_global_dir() / "registry.json", {})

def registry_lookup(path: str):
    return next((v for k, v in get_registry().items() if _same_path(k, path)), None)
