"""WSGI entry point for hosts that import ``application``."""
import os
import secrets
import sys


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATA_DIR = os.environ.get("APP_DATA_DIR") or os.path.join(
    os.path.expanduser("~"), ".brainora"
)
os.makedirs(DATA_DIR, mode=0o700, exist_ok=True)
if not os.environ.get("APP_DATA_DIR"):
    os.environ["APP_DATA_DIR"] = DATA_DIR
if not os.environ.get("APP_ENV"):
    os.environ["APP_ENV"] = "production"
NLTK_DATA_DIR = os.environ.get("NLTK_DATA") or os.path.join(
    os.path.expanduser("~"), ".nltk_data"
)
os.makedirs(NLTK_DATA_DIR, mode=0o700, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_DATA_DIR

# Ensure NLTK resources are present before the app module is imported.
# This makes the deployment self-contained without a separate build step.
import nltk  # noqa: E402

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

_NLTK_PACKAGES = [
    ("punkt",                      ["tokenizers/punkt"]),
    ("punkt_tab",                  ["tokenizers/punkt_tab"]),
    ("stopwords",                  ["corpora/stopwords"]),
    ("averaged_perceptron_tagger", ["taggers/averaged_perceptron_tagger"]),
    ("averaged_perceptron_tagger_eng", ["taggers/averaged_perceptron_tagger_eng"]),
    ("wordnet",                    ["corpora/wordnet", "corpora/wordnet.zip/wordnet/"]),
]

def _nltk_resource_exists(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False

for _pkg, _options in _NLTK_PACKAGES:
    if not any(_nltk_resource_exists(r) for r in _options):
        nltk.download(_pkg, download_dir=NLTK_DATA_DIR, quiet=True)

if not os.environ.get("SECRET_KEY"):
    secret_path = os.path.join(DATA_DIR, "secret_key")
    try:
        secret_fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(secret_fd, "w", encoding="utf-8") as secret_file:
            secret_file.write(secrets.token_hex(32))
    except (FileExistsError, OSError):
        pass
    try:
        with open(secret_path, encoding="utf-8") as secret_file:
            os.environ["SECRET_KEY"] = secret_file.read().strip()
    except OSError:
        # Filesystem not writable (e.g. free-tier ephemeral host) — use a
        # per-process key. Sessions won't survive restarts, but the app runs.
        os.environ["SECRET_KEY"] = secrets.token_hex(32)

from app import app as application
