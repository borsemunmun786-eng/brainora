"""
question_generator.py  –  MCQ generation using NLTK (no heavy models needed)
Handles English, Hindi/mixed-language, and keyword-only text robustly.
"""
import re
import random
import os
from typing import Dict, List, Tuple

import nltk
NLTK_DATA_DIR = os.environ.get("NLTK_DATA") or os.path.join(os.path.dirname(__file__), ".nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


def _resource_exists(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False


_NLTK_RESOURCES = (
    ("tokenizers/punkt",),
    ("tokenizers/punkt_tab",),
    ("corpora/stopwords",),
    ("taggers/averaged_perceptron_tagger",),
    ("taggers/averaged_perceptron_tagger_eng",),
    ("corpora/wordnet", "corpora/wordnet.zip/wordnet/"),
)
_NLTK_PACKAGE_MAP = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/wordnet": "wordnet",
}
for _resource_options in _NLTK_RESOURCES:
    if not any(_resource_exists(resource) for resource in _resource_options):
        _pkg = _NLTK_PACKAGE_MAP.get(_resource_options[0], _resource_options[0].split("/")[-1])
        nltk.download(_pkg, download_dir=NLTK_DATA_DIR, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag

STOP_WORDS = set(stopwords.words('english'))

# ── Language detection ────────────────────────────────────────────────────────

def _english_ratio(text: str) -> float:
    """Return fraction of ASCII-alphabetic characters (0.0 – 1.0)."""
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    ascii_alpha = sum(1 for c in chars if ord(c) < 128)
    return ascii_alpha / len(chars)


def _is_english_sentence(s: str) -> bool:
    return _english_ratio(s) >= 0.70


# ── Sentence quality ──────────────────────────────────────────────────────────

def _is_good_sentence(s: str) -> bool:
    """Accept informational English sentences only."""
    s = s.strip()
    if not _is_english_sentence(s):
        return False
    if len(s) < 30 or len(s) > 400:
        return False
    words = [w for w in word_tokenize(s) if w.isalpha()]
    if len(words) < 6:
        return False
    tagged = pos_tag(words)
    has_verb = any(tag.startswith('VB') for _, tag in tagged)
    if not has_verb:
        return False
    non_alpha = sum(1 for c in s if not c.isalpha() and c != ' ')
    if non_alpha / max(len(s), 1) > 0.40:
        return False
    return True


# ── Key phrase extraction ─────────────────────────────────────────────────────

def _extract_key_phrase(sentence: str) -> Tuple[str, str]:
    """Return (answer_phrase, fill-in-blank question).
    Answer must be a SHORT English word/phrase (1-3 tokens, all ASCII alpha)."""
    tokens = [t for t in word_tokenize(sentence) if t.isalpha()]
    tagged = pos_tag(tokens)

    candidates: List[Tuple[str, str]] = []
    i = 0
    while i < len(tagged):
        word, tag = tagged[i]
        # Only accept clean ASCII alpha tokens
        if not word.isascii():
            i += 1
            continue
        if tag in ('NNP', 'NNPS'):
            # Collect consecutive proper-noun tokens (max 3)
            phrase = [word]
            j = i + 1
            while j < len(tagged) and tagged[j][1] in ('NNP', 'NNPS') and tagged[j][0].isascii() and j - i < 3:
                phrase.append(tagged[j][0])
                j += 1
            candidates.append((' '.join(phrase), 'proper'))
            i = j
        elif tag in ('NN', 'NNS') and word.lower() not in STOP_WORDS and len(word) > 2:
            candidates.append((word, 'common'))
            i += 1
        else:
            i += 1

    if not candidates:
        return "", sentence

    proper = [p for p, t in candidates if t == 'proper']
    common = [p for p, t in candidates if t == 'common']
    key = proper[0] if proper else (common[-1] if common else "")

    # Reject if key is too long (likely a sentence fragment) or non-ASCII
    if not key or len(key) < 2 or len(key.split()) > 3 or not key.isascii():
        return "", sentence

    question = re.sub(re.escape(key), "______", sentence, count=1)
    if '______' not in question:
        return "", sentence
    return key, question


# ── Distractors ───────────────────────────────────────────────────────────────

def _wordnet_distractors(word: str, n: int = 3) -> List[str]:
    result = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower() and name.replace(' ', '').isalpha() and name.isascii():
                result.add(name)
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower() and name.replace(' ', '').isalpha() and name.isascii():
                    result.add(name)
        if len(result) >= n * 2:
            break
    return list(result)[:n]


def _corpus_distractors(answer: str, keywords: List[str], n: int = 3) -> List[str]:
    pool = [k for k in keywords
            if k.lower() != answer.lower()
            and len(k) > 2
            and k.isascii()
            and k.replace(' ', '').isalpha()
            and len(k.split()) <= 3]
    random.shuffle(pool)
    return pool[:n]


def _make_distractors(answer: str, keywords: List[str], n: int = 3) -> List[str]:
    # For proper nouns / tech terms (starts uppercase, no spaces), skip WordNet
    # because it returns unrelated meanings (e.g. "Python" → "Zen", "Amazon")
    is_proper = answer[0].isupper() if answer else False
    wn = [] if is_proper else _wordnet_distractors(answer, n)
    corp = _corpus_distractors(answer, keywords, n * 2)  # pull more from corpus
    combined = list({d.lower(): d for d in (wn + corp)}.values())
    combined = [d for d in combined if d.lower() != answer.lower()]
    random.shuffle(combined)
    pads = ["None of the above", "All of the above", "Cannot be determined"]
    for p in pads:
        if len(combined) >= n:
            break
        if p not in combined:
            combined.append(p)
    return combined[:n]


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 60) -> List[str]:
    """Extract top English noun keywords by frequency."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    freq: Dict[str, int] = {}
    original: Dict[str, str] = {}
    for word, tag in tagged:
        if (tag in ('NN', 'NNS', 'NNP', 'NNPS')
                and word.lower() not in STOP_WORDS
                and len(word) > 2
                and word.isalpha()
                and word.isascii()):           # ← English only
            key = word.lower()
            freq[key] = freq.get(key, 0) + 1
            if key not in original:
                original[key] = word
    sorted_keys = sorted(freq, key=lambda k: freq[k], reverse=True)[:top_n]
    return [original[k] for k in sorted_keys]


# ── Main generators ───────────────────────────────────────────────────────────

def generate_mcqs(text: str, topic: str = "General", num_questions: int = 5) -> List[Dict]:
    """Generate MCQ dicts from a text block. Works for English and mixed-language text."""
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    # Only process English sentences
    good = [s for s in sentences if _is_good_sentence(s)]
    keywords = extract_keywords(text)
    random.shuffle(good)

    questions: List[Dict] = []
    seen_answers: set = set()

    for sentence in good:
        if len(questions) >= num_questions:
            break
        answer, q_text = _extract_key_phrase(sentence)
        if not answer or '______' not in q_text:
            continue
        # Deduplicate by answer
        if answer.lower() in seen_answers:
            continue
        distractors = _make_distractors(answer, keywords, n=3)
        if len(distractors) < 3:
            continue

        options_list = [answer] + distractors[:3]
        random.shuffle(options_list)
        labels = ['A', 'B', 'C', 'D']
        options = {l: v for l, v in zip(labels, options_list)}
        correct = next(k for k, v in options.items() if v == answer)

        questions.append({
            'question': q_text,
            'options':  options,
            'answer':   correct,
            'answer_text': answer,
            'topic':    topic,
            'source':   sentence,
        })
        seen_answers.add(answer.lower())

    # ── Fallback: keyword MCQs when sentence-level can't fill quota ──
    if len(questions) < num_questions and len(keywords) >= 4:
        questions.extend(_keyword_fallback_questions(
            keywords, topic, text, num_questions - len(questions), seen_answers
        ))

    return questions


def _keyword_fallback_questions(
    keywords: List[str], topic: str, text: str, n: int, seen: set
) -> List[Dict]:
    """Diverse keyword-based MCQs: each question asks about a DIFFERENT keyword."""
    questions: List[Dict] = []
    # Use only unseen keywords
    pool = [k for k in keywords if k.lower() not in seen]
    random.shuffle(pool)

    words = text.split()
    chunk_size = 80
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    # Build varied question templates
    templates = [
        lambda kw, t: f"Which of the following is a key term in {t}?",
        lambda kw, t: f"In the context of {t}, which term refers to a core concept?",
        lambda kw, t: f"Which of the following concepts is related to {t}?",
        lambda kw, t: f"Identify the correct term discussed under {t}:",
        lambda kw, t: f"Which of these is directly associated with {t}?",
    ]
    tmpl_idx = 0

    i = 0
    while len(questions) < n and i < len(pool):
        answer = pool[i]
        i += 1
        if answer.lower() in seen:
            continue
        distractors = _make_distractors(answer, pool, n=3)
        if len(distractors) < 3:
            continue

        options_list = [answer] + distractors[:3]
        random.shuffle(options_list)
        labels = ['A', 'B', 'C', 'D']
        options = {l: v for l, v in zip(labels, options_list)}
        correct = next(k for k, v in options.items() if v == answer)

        source = next((c for c in chunks if answer.lower() in c.lower()),
                      chunks[0] if chunks else "")

        q_text = templates[tmpl_idx % len(templates)](answer, topic)
        tmpl_idx += 1

        questions.append({
            'question': q_text,
            'options':  options,
            'answer':   correct,
            'answer_text': answer,
            'topic':    topic,
            'source':   source,
        })
        seen.add(answer.lower())

    return questions


def generate_quiz_from_topics(
    topics: Dict[str, str],
    questions_per_topic: int = 3,
    max_total: int = 20
) -> List[Dict]:
    """Generate quiz from {topic: text} dict."""
    all_qs: List[Dict] = []
    for topic, text in topics.items():
        if len(text) < 50:
            continue
        qs = generate_mcqs(text, topic=topic, num_questions=questions_per_topic)
        all_qs.extend(qs)
    random.shuffle(all_qs)
    return all_qs[:max_total]
