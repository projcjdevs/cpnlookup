import base64
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own home (~/.cpnlookup) and working directory."""
    home, work = tmp_path / "home", tmp_path / "work"; home.mkdir(); work.mkdir()
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("USERPROFILE", str(home)); monkeypatch.chdir(work)
    return work

class FakeGitHub:
    """Stands in for cpnlookup.github network calls. files: {path: content}; sha changes when content changes."""
    def __init__(self, files): self.files, self.fetched, self.fail_on = dict(files), [], None
    def sha(self, path): return f"{abs(hash(self.files[path])):x}"
    def session(self): return object()
    def fetch_repo_tree(self, s, repo): return [{"path": p, "sha": self.sha(p), "size": len(c)} for p, c in self.files.items()], False
    def fetch_blob(self, s, repo, sha):
        from cpnlookup.github import GitHubError
        path = next(p for p in self.files if self.sha(p) == sha)
        if path == self.fail_on: raise GitHubError("rate limited")
        self.fetched.append(path); return self.files[path]

def fake_embed(chunks):
    """Deterministic 8-dim vectors so FAISS works without downloading a model."""
    return np.array([np.random.default_rng(abs(hash(c["name"] + c["file_path"])) % 2**32).random(8, dtype=np.float32) for c in chunks])

SAMPLE = {
    "pkg/fetch.py": 'def fetch_repo_tree(repo):\n    """List files."""\n    return parse(repo)\n\ndef parse(x):\n    return x\n',
    "pkg/app.py": "class App:\n    def __init__(self):\n        self.tree = fetch_repo_tree('a/b')\n\n    def run(self):\n        return parse(1)\n",
    "pkg/other.py": "class Other:\n    def __init__(self):\n        pass\n\ndef my_parse(y):\n    return y\n",
    "README.md": "# Demo\nIntro text.\n\n## Install\npip install demo\n",
}

@pytest.fixture
def gh(monkeypatch):
    from cpnlookup import github
    from cpnlookup.indexer import embedder
    fake, calls = FakeGitHub(SAMPLE), []
    for name in ("session", "fetch_repo_tree", "fetch_blob"): monkeypatch.setattr(github, name, getattr(fake, name))
    monkeypatch.setattr(embedder, "embed_chunks", lambda chunks: (calls.append(len(chunks)), fake_embed(chunks))[1])
    fake.embed_calls = calls
    return fake

def b64(text): return base64.b64encode(text.encode()).decode()
