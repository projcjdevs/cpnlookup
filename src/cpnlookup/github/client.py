import requests
from typing import List, Dict
from cpnlookup.utils.config import get_github_token

def get_user_repos(username: str) -> List[Dict]:
    """Fetches active repositories for a given GitHub user."""
    url = f"https://api.github.com/users/{username}/repos"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    token = get_github_token()
    if token:
        headers["Authorization"] = f"token {token}"
        
    params = {"sort": "updated", "per_page": 100}
    response = requests.get(url, headers=headers, params=params)
    
    response.raise_for_status() 
    
    return response.json()