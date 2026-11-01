OLLAMA = "http://localhost:11434"

class OllamaError(Exception): pass

def check_ollama() -> bool:
    import requests
    try: return requests.get(f"{OLLAMA}/api/tags", timeout=3).ok
    except requests.RequestException: return False

def chat_with_ollama(prompt: str, model: str = "mistral", num_ctx: int = 8192) -> str:
    """One non-streaming generation. Raises OllamaError with Ollama's own reason on failure."""
    import requests
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_ctx": num_ctx}}
    try: r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=(5, 900))
    except requests.RequestException as e: raise OllamaError(f"Could not reach Ollama at {OLLAMA}: {e}") from None
    if not r.ok:
        try: msg = r.json().get("error", r.text)
        except ValueError: msg = r.text
        hint = f" Try: ollama pull {model}" if r.status_code == 404 else ""
        raise OllamaError(f"Ollama returned {r.status_code}: {msg.strip()}.{hint}")
    return r.json().get("response", "").strip()
