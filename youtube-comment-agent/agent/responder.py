"""Genera las respuestas con la API de OpenAI usando el mínimo de llamadas y tokens.

- Varios comentarios por llamada (``batch_size``), así las instrucciones se envían una vez por lote.
- Las instrucciones (tu estilo + ejemplos) son idénticas en cada llamada, así que OpenAI
  las cachea y cobra ~10% del precio a partir de la segunda.
- ``reasoning.effort = none`` y ``service_tier = flex`` (mitad de precio) por defecto.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .style import URL_RE, strip_mentions

log = logging.getLogger(__name__)

HARD_MAX_CHARS = 1000
TOKENS_PER_REPLY = 200

REPLIES_SCHEMA = {
    "type": "object",
    "properties": {
        "replies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "skip": {"type": "boolean"},
                    "reply": {"type": "string"},
                },
                "required": ["id", "skip", "reply"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["replies"],
    "additionalProperties": False,
}


def build_instructions(profile: dict) -> str:
    def join(values: list[str]) -> str:
        return ", ".join(values) if values else "(ninguno destacado)"

    examples = "\n\n".join(
        f"Comentario: {ex['comment']}\nMi respuesta: {ex['reply']}" for ex in profile["canonical_examples"]
    )
    emoji_note = (
        f"Usas emojis en el {int(profile['emoji_rate'] * 100)}% de tus respuestas; los más comunes: "
        f"{join(profile['top_emojis'])}."
        if profile["emoji_rate"] > 0
        else "Casi nunca usas emojis: no los pongas."
    )
    return f"""Eres quien administra este canal de YouTube y respondes, en primera persona, a los comentarios de tu comunidad.
Tu único objetivo es sonar EXACTAMENTE como en tus respuestas reales de abajo: mismo tono, largo, vocabulario, puntuación y forma de saludar y despedirte. Si una palabra o expresión no encaja con cómo escribes en los ejemplos, no la uses.

CÓMO ESCRIBES (sacado de {profile['examples_total']} respuestas tuyas reales):
- Largo típico: {profile['median_words']} palabras (casi nunca más de {profile['p90_words']}).
- Empiezas con mayúscula en el {int(profile['capitalized_rate'] * 100)}% de los casos; signos de exclamación en el {int(profile['exclamation_rate'] * 100)}%; preguntas de vuelta en el {int(profile['question_rate'] * 100)}%.
- {emoji_note}
- Palabras que más usas: {join(profile['top_words'])}.
- Inicios frecuentes: {join(profile['openers'])}.
- Cierres frecuentes: {join(profile['closers'])}.

REGLAS:
1. Responde en el mismo idioma del comentario, con tu estilo.
2. No inventes datos, enlaces, fechas, precios, promesas ni anuncios de videos futuros.
3. Si el comentario es de un miembro del canal ("tier": "miembro"), agradécele con cercanía, sin exagerar ni salirte de tu estilo.
4. Pon "skip": true y "reply": "" si el comentario es spam, promoción, ofensivo, trata un tema delicado (salud, dinero, temas legales, una crisis personal), pregunta algo que solo tú podrías saber, o no tienes claro cómo lo responderías. Es mejor no responder que responder mal.
5. Sin comillas, sin @menciones y sin hashtags. Una sola respuesta por comentario.
6. Cada comentario trae "ejemplos_parecidos": respuestas tuyas reales a comentarios similares. Úsalas como guía principal de tono y palabras.

Devuelve un objeto JSON con "replies": una entrada por cada comentario recibido, con su mismo "id".

TUS RESPUESTAS REALES:

{examples}
"""


def clean_reply(text: str, profile: dict) -> str | None:
    reply = strip_mentions(text.strip().strip('"“”').strip())
    reply = re.sub(r"\n{3,}", "\n\n", reply)
    if not reply:
        return None
    if URL_RE.search(reply) and profile.get("link_rate", 0) == 0:
        return None
    limit = min(HARD_MAX_CHARS, max(300, profile.get("p90_chars", 0) * 3))
    if len(reply) > limit:
        return None
    return reply


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage) -> None:
        self.calls += 1
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        details = getattr(usage, "input_tokens_details", None)
        self.cached_tokens += getattr(details, "cached_tokens", 0) or 0


@dataclass
class Responder:
    client: object
    model: str
    reasoning_effort: str = "none"
    service_tier: str = "flex"
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_env(cls, model: str, reasoning_effort: str, service_tier: str) -> "Responder":
        from openai import OpenAI

        # flex puede tardar más en responder; le damos margen.
        return cls(OpenAI(timeout=900.0, max_retries=3), model, reasoning_effort, service_tier)

    def _request(self, instructions: str, payload: str, n_items: int):
        kwargs = {
            "model": self.model,
            "instructions": instructions,
            "input": payload,
            "text": {
                "format": {"type": "json_schema", "name": "replies", "schema": REPLIES_SCHEMA, "strict": True}
            },
            "max_output_tokens": TOKENS_PER_REPLY * n_items + 300,
            "prompt_cache_key": "youtube-comment-agent",
            "store": False,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.reasoning_effort != "none":
                kwargs["max_output_tokens"] += 2000
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        try:
            return self.client.responses.create(**kwargs)
        except Exception as err:  # noqa: BLE001 - reintento sin flex ante cualquier fallo
            if "service_tier" not in kwargs:
                raise
            log.warning("Falló con service_tier=%s (%s); reintento con el tier normal.", self.service_tier, err)
            kwargs.pop("service_tier")
            return self.client.responses.create(**kwargs)

    def generate(self, instructions: str, items: list[dict], profile: dict, batch_size: int = 10) -> dict[str, str]:
        """Devuelve {id_del_comentario: respuesta}. Los comentarios omitidos no aparecen."""
        results: dict[str, str] = {}
        for start in range(0, len(items), max(1, batch_size)):
            batch = items[start : start + batch_size]
            payload = json.dumps({"comentarios": batch}, ensure_ascii=False)
            response = self._request(instructions, payload, len(batch))
            self.usage.add(getattr(response, "usage", None))
            try:
                data = json.loads(response.output_text)
            except (json.JSONDecodeError, TypeError):
                log.warning("Respuesta no válida del modelo para un lote de %d comentarios; lo salto.", len(batch))
                continue
            wanted = {item["id"] for item in batch}
            for entry in data.get("replies", []):
                if entry.get("id") not in wanted or entry.get("skip"):
                    continue
                reply = clean_reply(entry.get("reply", ""), profile)
                if reply:
                    results[entry["id"]] = reply
        return results
