"""Internal NLTK data & tokenization helpers."""

import re
import nltk


def ensure_nltk_data(resource_path: str, download_name: str) -> bool:
    """Ensure an NLTK resource is available, downloading it only if missing."""
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        try:
            return bool(nltk.download(download_name, quiet=True))
        except Exception:
            return False


def safe_word_tokenize(text: str) -> list[str]:
    """Tokenize text into words using NLTK if available, with a regex fallback."""
    if ensure_nltk_data("tokenizers/punkt_tab", "punkt_tab") or ensure_nltk_data(
        "tokenizers/punkt", "punkt"
    ):
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            pass
    return re.findall(r"\w+|[^\w\s]", text)


def safe_sent_tokenize(text: str) -> list[str]:
    """Tokenize text into sentences using NLTK if available, with a regex fallback."""
    if ensure_nltk_data("tokenizers/punkt_tab", "punkt_tab") or ensure_nltk_data(
        "tokenizers/punkt", "punkt"
    ):
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            pass
    stripped = text.strip()
    if not stripped:
        return []
    return re.split(r"(?<=[.!?])\s+", stripped)
