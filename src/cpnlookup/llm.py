"""Ollama client: where it lives (OLLAMA_HOST), whether it can answer, and streamed generation."""
import os
import re

CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")   # terminal control characters; an answer must never be able to drive the terminal

class OllamaError(Exception): pass

def host() -> str:
    """Ollama's base URL from OLLAMA_HOST (host, host:port, :port or a full URL), default http://127.0.0.1:11434.
    Not "localhost": Windows tries IPv6 first and waits 2 s per request before reaching an IPv4-only Ollama."""
    from urllib.parse import urlsplit
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    try:
        u = urlsplit(raw if "://" in raw else f"http://{raw}"); port = u.port or (11434 if "://" not in raw else {"http": 80, "https": 443}[u.scheme])
        if u.scheme not in ("http", "https"): raise ValueError
    except (ValueError, KeyError): raise OllamaError(f"OLLAMA_HOST is not a valid address: {raw}") from None
    name = {None: "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(u.hostname, u.hostname)   # bind addresses are not ones a client can call
    return f"{u.scheme}://{f'[{name}]' if ':' in name else name}:{port}"

def check_ollama() -> bool:
    import requests
    try: return requests.get(f"{host()}/api/tags", timeout=3).ok
    except (requests.RequestException, OllamaError): return False

def ensure_ready(model: str) -> None:
    """Raises OllamaError naming the exact fix: Ollama not installed, not running, or the model not pulled."""
    import shutil
    import requests
    url = host()
    try: r = requests.get(f"{url}/api/tags", timeout=3); names = {m.get("name", "") for m in r.json().get("models", [])} if r.ok else None
    except requests.RequestException:
        if ("localhost" in url or "127.0.0.1" in url) and shutil.which("ollama") is None:
            raise OllamaError(f"Ollama is not installed. Get it from https://ollama.com, then run: ollama pull {model}") from None
        raise OllamaError(f"Ollama is not running at {url}. Start it with: ollama serve") from None
    except (ValueError, AttributeError): names = None
    if names is None: raise OllamaError(f"{url} did not answer like an Ollama server. Check OLLAMA_HOST.")
    if not {model, f"{model}:latest"} & names: raise OllamaError(f"The model '{model}' is not installed in Ollama. Run: ollama pull {model}")

def stream_ollama(prompt: str, model: str = "mistral", num_ctx: int = 8192):
    """Yields the answer piece by piece as Ollama writes it. Raises OllamaError with Ollama's own reason, also mid-answer."""
    import json
    import requests
    payload, url, done = {"model": model, "prompt": prompt, "stream": True, "options": {"num_ctx": num_ctx}}, host(), False
    try:
        with requests.post(f"{url}/api/generate", json=payload, stream=True, timeout=(5, 900)) as r:
            if not r.ok:
                try: msg = r.json().get("error", r.text)
                except ValueError: msg = r.text
                raise OllamaError(f"Ollama returned {r.status_code}: {msg.strip()}." + (f" Try: ollama pull {model}" if r.status_code == 404 else ""))
            for line in r.iter_lines():
                if not line: continue
                part = json.loads(line)
                if part.get("error"): raise OllamaError(f"Ollama stopped: {part['error']}")
                piece = CONTROL.sub("", part.get("response") or "")
                if piece: yield piece
                if part.get("done"): done = True; break
    except requests.RequestException as e: raise OllamaError(f"Lost the connection to Ollama at {url}: {e}") from None
    except (ValueError, AttributeError): raise OllamaError("Ollama sent a reply that could not be read.") from None
    if not done: raise OllamaError("Ollama closed the connection before the answer was complete.")
