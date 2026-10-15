"""
pdf_extractor.py  –  PDF text extraction and topic detection
"""
import re
import pdfplumber
import PyPDF2
from typing import Dict, List


def extract_text(file_path: str) -> str:
    """Extract text from a PDF file path, with PyPDF2 fallback."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass

    if len(text) < 100:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception:
            pass

    return clean_text(text)


def clean_text(text: str) -> str:
    """Clean up extracted PDF text (strips non-ASCII control chars, keeps structure)."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # non-ASCII → space (PDF only)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_transcript(text: str) -> str:
    """Clean transcript text — preserve Unicode (Hindi/other scripts), just normalise whitespace."""
    text = re.sub(r'\[.*?\]', ' ', text)          # remove [Music], [Applause] etc.
    text = re.sub(r'\(.*?\)', ' ', text)          # remove (laughter) etc.
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def detect_topics(text: str, is_transcript: bool = False) -> Dict[str, str]:
    """
    Detect topic sections from headings (short, Title-Case / ALL-CAPS lines).
    For transcripts (continuous speech), splits into equal-sized chunks instead.
    Returns {topic_name: topic_text}.
    """
    if is_transcript:
        return _split_transcript_into_chunks(text)

    lines = text.split('\n')
    heading_re = re.compile(r'^(?:[A-Z][A-Za-z0-9 :,\-]{2,60}|[A-Z ]{4,60})$')

    topics: Dict[str, str] = {}
    current_topic = "General"
    current_chunk: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        is_heading = (
            len(s) < 80
            and not s.endswith('.')
            and len(s.split()) <= 10
            and not s[0].islower()
            and heading_re.match(s)
        )
        if is_heading:
            if current_chunk:
                topics[current_topic] = ' '.join(current_chunk)
            current_topic = s
            current_chunk = []
        else:
            current_chunk.append(s)

    if current_chunk:
        topics[current_topic] = ' '.join(current_chunk)

    # Merge tiny chunks back into General
    merged: Dict[str, str] = {}
    leftover: List[str] = []
    for t, chunk in topics.items():
        if len(chunk) < 150:
            leftover.append(chunk)
        else:
            merged[t] = chunk
    if leftover:
        merged["General"] = merged.get("General", "") + " " + " ".join(leftover)

    return merged if merged else {"General": text}


def _split_transcript_into_chunks(text: str, chunk_words: int = 300) -> Dict[str, str]:
    """
    Split a transcript (no headings) into named chunks of ~chunk_words words.
    Uses the most frequent ASCII noun in each chunk as the topic label.
    """
    words = text.split()
    if len(words) < chunk_words:
        label = _label_chunk(text) or "General"
        return {label: text}

    chunks: Dict[str, str] = {}
    seen_labels: Dict[str, int] = {}
    idx = 0
    part = 1
    while idx < len(words):
        chunk_text = ' '.join(words[idx: idx + chunk_words])
        idx += chunk_words
        label = _label_chunk(chunk_text) or f"Part {part}"
        # Deduplicate labels
        if label in seen_labels:
            seen_labels[label] += 1
            label = f"{label} {seen_labels[label]}"
        else:
            seen_labels[label] = 1
        chunks[label] = chunk_text
        part += 1

    return chunks


def _label_chunk(text: str) -> str:
    """Pick the most frequent meaningful ASCII noun as a chunk label."""
    stopwords_set = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'this', 'that', 'these',
        'those', 'it', 'its', 'of', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'into', 'about', 'and', 'or', 'but',
        'so', 'not', 'we', 'you', 'they', 'he', 'she', 'our', 'your',
        'also', 'just', 'like', 'very', 'here', 'now', 'then', 'than',
        'more', 'some', 'any', 'all', 'can', 'get', 'one', 'two',
        'aur', 'hai', 'hain', 'ka', 'ke', 'ki', 'ko', 'se', 'mein',
        'kya', 'koi', 'bhi', 'toh', 'jo', 'yeh', 'woh', 'nahi',
    }
    freq: Dict[str, int] = {}
    for word in re.findall(r'[A-Za-z]{3,}', text):
        w = word.lower()
        if w not in stopwords_set and word[0].isupper():
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        # fall back to any frequent word
        for word in re.findall(r'[A-Za-z]{4,}', text):
            w = word.lower()
            if w not in stopwords_set:
                freq[w] = freq.get(w, 0) + 1
    if not freq:
        return ""
    best = max(freq, key=lambda k: freq[k])
    return best.title()
