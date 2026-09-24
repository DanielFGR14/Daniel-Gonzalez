"""Entrenamiento local: aprende el estilo a partir de TUS respuestas, sin gastar API de OpenAI.

1. ``extract_pairs`` saca pares (comentario del fan -> tu respuesta) de los hilos de tu canal.
2. ``build_profile`` resume cómo escribes: longitud, vocabulario, emojis, saludos y despedidas,
   y guarda un conjunto variado de ejemplos reales.
3. ``ExampleIndex`` busca tus respuestas pasadas a comentarios parecidos, para dárselas al
   modelo como ejemplos en cada llamada.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from datetime import datetime
from typing import Iterable

from .youtube import Comment, Thread

HANDLE_RE = re.compile(r"^\s*(@[\w.\-]+)")
LEADING_MENTIONS_RE = re.compile(r"^(\s*@[\w.\-]+[\s,:]*)+")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # banderas
    "\U0001F300-\U0001FAFF"  # símbolos y pictogramas
    "☀-➿"  # símbolos varios y dingbats
    "⭐⭕❤✨"
    "]"
)

STOPWORDS = set(
    """
    a al algo ante antes aquí así aun aunque cada como con contra cual cuando de del desde donde dos
    e el ella ellas ellos en entre era es esa ese eso esta este esto estos esas esos está están fue
    ha han hay la las le les lo los me mi mis mucho muy más ni no nos o os otra otro para pero poco
    por porque que qué se ser si sí sin sobre su sus también te ti tu tus un una uno unos unas ya y yo
    the and to of in is it you that for on with this are be at as your have was so but not my me i
    we our they them its it's i'm im an or if do just all can will from about what there their
    """.split()
)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 2]


def strip_mentions(text: str) -> str:
    return LEADING_MENTIONS_RE.sub("", text).strip()


# -- 1. Pares de entrenamiento ------------------------------------------------------


def _parent_of(thread: Thread, index: int) -> Comment:
    """A quién respondía la respuesta ``index``: al @mencionado o, si no, al comentario principal."""
    reply = thread.replies[index]
    match = HANDLE_RE.match(reply.text)
    if match:
        handle = match.group(1).lower()
        for previous in reversed(thread.replies[:index]):
            if previous.author_name.lower() == handle:
                return previous
    return thread.top


def extract_pairs(
    threads: Iterable[Thread],
    owner_id: str,
    exclude_reply_ids: Iterable[str] = (),
    before: datetime | None = None,
) -> list[dict]:
    """Solo usa respuestas escritas por el dueño del canal (y nunca las que publicó el agente)."""
    excluded = set(exclude_reply_ids)
    pairs = []
    for thread in threads:
        for i, reply in enumerate(thread.replies):
            if reply.author_channel_id != owner_id or reply.id in excluded:
                continue
            if before and reply.published_at >= before:
                continue
            parent = _parent_of(thread, i)
            if parent.author_channel_id == owner_id:
                continue
            answer = strip_mentions(reply.text)
            question = strip_mentions(parent.text)
            if not answer or not question:
                continue
            pairs.append(
                {
                    "comment": question,
                    "reply": answer,
                    "reply_id": reply.id,
                    "video_id": thread.video_id,
                    "published_at": reply.published_at.isoformat(),
                }
            )
    return pairs


# -- 2. Perfil de estilo -------------------------------------------------------


def _rate(items: list[str], predicate) -> float:
    return round(sum(1 for i in items if predicate(i)) / len(items), 2) if items else 0.0


def _percentile(values: list[int], pct: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))]


def _edge_phrases(replies: list[str], first: bool, top: int) -> list[str]:
    counts: Counter[str] = Counter()
    for reply in replies:
        tokens = words(reply)
        if len(tokens) < 2:
            continue
        counts[" ".join(tokens[:2] if first else tokens[-2:])] += 1
    return [phrase for phrase, n in counts.most_common(top) if n >= 2]


def select_canonical(pairs: list[dict], count: int) -> list[dict]:
    """Ejemplos variados: sin duplicados y repartidos a lo largo del tiempo."""
    seen: set[str] = set()
    unique = []
    for pair in sorted(pairs, key=lambda p: p["published_at"], reverse=True):
        key = " ".join(words(pair["reply"]))
        if key and key not in seen and len(pair["reply"]) <= 400 and len(pair["comment"]) <= 400:
            seen.add(key)
            unique.append(pair)
    if len(unique) <= count:
        return unique
    step = len(unique) / count
    return [unique[int(i * step)] for i in range(count)]


def build_profile(pairs: list[dict], canonical_count: int = 20) -> dict:
    replies = [p["reply"] for p in pairs]
    word_counts = [len(words(r)) for r in replies]
    vocabulary = Counter(w for r in replies for w in content_words(r))
    emojis = Counter(e for r in replies for e in EMOJI_RE.findall(r))
    return {
        "examples_total": len(pairs),
        "median_words": int(statistics.median(word_counts)) if word_counts else 0,
        "p10_words": _percentile(word_counts, 0.1) if word_counts else 0,
        "p90_words": _percentile(word_counts, 0.9) if word_counts else 0,
        "p90_chars": _percentile([len(r) for r in replies], 0.9) if replies else 0,
        "emoji_rate": _rate(replies, lambda r: bool(EMOJI_RE.search(r))),
        "exclamation_rate": _rate(replies, lambda r: "!" in r),
        "question_rate": _rate(replies, lambda r: "?" in r),
        "capitalized_rate": _rate(replies, lambda r: r[:1].isupper()),
        "top_words": [w for w, _ in vocabulary.most_common(60)],
        "top_emojis": [e for e, _ in emojis.most_common(12)],
        # Para medir el estilo de cada respuesta nueva.
        "vocabulary": [w for w, _ in vocabulary.most_common(5000)],
        "emojis": [e for e, _ in emojis.most_common(50)],
        "openers": _edge_phrases(replies, first=True, top=10),
        "closers": _edge_phrases(replies, first=False, top=10),
        "canonical_examples": [
            {"comment": p["comment"], "reply": p["reply"]} for p in select_canonical(pairs, canonical_count)
        ],
    }


# -- 3. Búsqueda de ejemplos parecidos (TF-IDF local) ------------------------------------


class ExampleIndex:
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs
        docs = [Counter(content_words(p["comment"])) for p in pairs]
        doc_freq = Counter(term for doc in docs for term in doc)
        total = len(docs)
        self.idf = {term: math.log((1 + total) / (1 + n)) + 1 for term, n in doc_freq.items()}
        self.vectors = [self._vector(doc) for doc in docs]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vec = {term: n * self.idf[term] for term, n in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(self, text: str, k: int = 3, skip_replies: Iterable[str] = ()) -> list[dict]:
        query = self._vector(Counter(content_words(text)))
        if not query or k <= 0:
            return []
        skip = set(skip_replies)
        scored = []
        for pair, vec in zip(self.pairs, self.vectors):
            score = sum(weight * vec.get(term, 0.0) for term, weight in query.items())
            if score > 0 and pair["reply"] not in skip:
                scored.append((score, pair))
        scored.sort(key=lambda item: item[0], reverse=True)
        results, seen = [], set()
        for _, pair in scored:
            if pair["reply"] in seen:
                continue
            seen.add(pair["reply"])
            results.append({"comment": pair["comment"], "reply": pair["reply"]})
            if len(results) == k:
                break
        return results
