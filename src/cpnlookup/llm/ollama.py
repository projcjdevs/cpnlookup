import requests
import json

def check_ollama():
    # Check if Ollama is running locally
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False
    
def chat_with_ollama(prompt: str, model: str = "mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model, 
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        return f"Error: Ollama returned status {response.status_code}"