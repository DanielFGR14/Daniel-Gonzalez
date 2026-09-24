"""Control de calidad: que cada respuesta suene a ti pero NO sea una copia.

Para cada comentario el modelo propone varias opciones y aquí se elige la mejor:

- **Novedad (obligatoria):** se compara con todas tus respuestas pasadas, con las que el agente
  publicó en los últimos días y con las ya elegidas en esta misma ejecución. Si se parece
  demasiado a alguna (``max_similarity``), se descarta. Esto también evita comentarios
  repetitivos, que YouTube trata como spam.
- **Estilo:** puntaje de 0 a 1 que mide si usa tu vocabulario, tu largo típico, tus emojis y qué
  tan cerca está de tu respuesta real más parecida (sin llegar a copiarla).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .style import EMOJI_RE, content_words, words

SPACES_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin signos ni emojis: 'Gracias!! 🙏' y 'gracias' quedan iguales."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return SPACES_RE.sub(" ", text).strip()


def trigrams(text: str) -> frozenset[str]:
    padded = f" {normalize(text)} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def similarity(a: str, b: str) -> float:
    """Coeficiente de Dice sobre trigramas de caracteres: 1.0 = idénticas, 0.0 = nada en común."""
    ga, gb = trigrams(a), trigrams(b)
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


class NoveltyChecker:
    def __init__(self, texts: Iterable[str] = ()):
        self._known: dict[frozenset[str], str] = {}
        for text in texts:
            self.add(text)

    def add(self, text: str) -> None:
        grams = trigrams(text)
        if grams:
            self._known.setdefault(grams, text)

    def closest(self, text: str) -> tuple[float, str]:
        """La respuesta ya existente más parecida y su similitud."""
        grams = trigrams(text)
        if not grams:
            return 0.0, ""
        if grams in self._known:
            return 1.0, self._known[grams]
        best, best_text, n = 0.0, "", len(grams)
        for other, other_text in self._known.items():
            m = len(other)
            # Cota superior barata: si ni en el mejor caso supera al mejor, se salta.
            if 2 * min(n, m) / (n + m) <= best:
                continue
            score = 2 * len(grams & other) / (n + m)
            if score > best:
                best, best_text = score, other_text
        return best, best_text


def _length_fit(n_words: int, profile: dict) -> float:
    # Margen amplio: nadie escribe siempre igual de largo.
    low = max(1, profile.get("p10_words", 1)) / 2
    p90 = max(1, profile.get("p90_words", 1))
    high = max(p90 * 1.5, p90 + 3)
    if n_words < low:
        return n_words / low
    if n_words > high:
        return max(0.0, 1 - (n_words - high) / high)
    return 1.0


def _emoji_fit(reply: str, profile: dict) -> float:
    used = EMOJI_RE.findall(reply)
    rate = profile.get("emoji_rate", 0.0)
    if not used:
        return 1 - rate / 2
    if rate < 0.05:
        return 0.0
    own = set(profile.get("emojis", profile.get("top_emojis", [])))
    return sum(1 for e in used if e in own) / len(used)


def _vocab_fit(reply: str, profile: dict, context: str) -> float:
    reply_words = content_words(reply)
    if not reply_words:
        return 1.0
    known = set(profile.get("vocabulary", profile.get("top_words", []))) | set(content_words(context))
    return sum(1 for w in reply_words if w in known) / len(reply_words)


@dataclass
class Evaluation:
    text: str
    style: float
    closest_similarity: float
    closest_text: str
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return not self.reason


def evaluate(
    reply: str,
    profile: dict,
    novelty: NoveltyChecker,
    context: str = "",
    max_similarity: float = 0.75,
    min_style: float = 0.5,
) -> Evaluation:
    sim, closest = novelty.closest(reply)
    closeness = min(sim, max_similarity) / max_similarity if max_similarity else 0.0
    style = (
        0.45 * _vocab_fit(reply, profile, context)
        + 0.20 * _length_fit(len(words(reply)), profile)
        + 0.15 * _emoji_fit(reply, profile)
        + 0.20 * closeness
    )
    result = Evaluation(reply, round(style, 2), round(sim, 2), closest)
    if sim >= max_similarity:
        result.reason = f"demasiado parecida a una respuesta existente ({sim:.2f})"
    elif style < min_style:
        result.reason = f"no suena a ti (estilo {style:.2f})"
    return result


def choose(
    options: list[str],
    profile: dict,
    novelty: NoveltyChecker,
    context: str = "",
    max_similarity: float = 0.75,
    min_style: float = 0.5,
) -> Evaluation | None:
    """La opción aceptada con mejor estilo; si ninguna pasa, la mejor rechazada (para el reporte)."""
    evaluations = [evaluate(o, profile, novelty, context, max_similarity, min_style) for o in options if o]
    if not evaluations:
        return None
    evaluations.sort(key=lambda e: (e.accepted, e.style), reverse=True)
    return evaluations[0]
