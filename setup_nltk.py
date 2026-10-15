"""Download the NLTK resources required by the quiz generator."""
import os

import nltk


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.abspath(
    os.environ.get("NLTK_DATA") or os.path.join(ROOT_DIR, ".nltk_data")
)
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


def _resource_exists(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False


RESOURCES = (
    ("punkt", ("tokenizers/punkt",)),
    ("punkt_tab", ("tokenizers/punkt_tab",)),
    ("stopwords", ("corpora/stopwords",)),
    ("averaged_perceptron_tagger", ("taggers/averaged_perceptron_tagger",)),
    ("averaged_perceptron_tagger_eng", ("taggers/averaged_perceptron_tagger_eng",)),
    ("wordnet", ("corpora/wordnet", "corpora/wordnet.zip/wordnet/")),
)

for package, resource_options in RESOURCES:
    if not any(_resource_exists(resource) for resource in resource_options):
        if not nltk.download(package, download_dir=NLTK_DATA_DIR, quiet=True):
            raise SystemExit(f"Could not download NLTK resource: {package}")
        if not any(_resource_exists(resource) for resource in resource_options):
            raise SystemExit(f"NLTK resource was not installed: {package}")

print(f"NLTK resources are ready in {NLTK_DATA_DIR}")