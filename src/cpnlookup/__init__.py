"""cpnlookup: query GitHub repositories in natural language, locally."""
from importlib.metadata import PackageNotFoundError, version

try: __version__ = version("cpnlookup")
except PackageNotFoundError: __version__ = "0.0.0"
