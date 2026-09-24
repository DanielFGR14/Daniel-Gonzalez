"""Modo aprobación: los borradores se publican como un issue de GitHub que tú revisas.

Marcas las respuestas que quieres publicar (puedes editar su texto) y cierras el issue; eso
dispara la publicación. Luego el contenido del issue se borra, porque las políticas de YouTube
no permiten guardar sus datos indefinidamente.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

LABEL = "respuestas-por-aprobar"
BOT_LOGIN = "github-actions[bot]"
MARKER = "<!-- yt-agent-borradores -->"
ZWSP = "​"

THREAD_RE = re.compile(r"^<!-- thread:([\w-]+) -->[ \t]*$", re.M)
CHECK_RE = re.compile(r"^- \[([ xX])\] Publicar esta", re.M)
ALL_RE = re.compile(r"^- \[[xX]\] Aprobar TODAS", re.M)
REPLY_RE = re.compile(r"\*\*Respuesta:\*\*[ \t]*\n(.*?)\n?<!-- fin -->", re.S)


@dataclass
class Approval:
    thread_id: str
    checked: bool
    reply: str


def _safe(text: str) -> str:
    """Evita que el texto de YouTube rompa el issue o mencione a usuarios de GitHub."""
    text = text.replace("<!--", "").replace("-->", "").replace("<", "&lt;")
    return text.replace("@", "@" + ZWSP)


def render_issue(queue: list[tuple[str, dict]], owner_login: str, date: str, expires_days: int) -> str:
    lines = [
        MARKER,
        f"@{owner_login} tienes **{len(queue)} respuestas** listas para aprobar ({date}).",
        "",
        "**Cómo aprobar:** marca *Publicar esta* en las que quieras (o *Aprobar TODAS*) y luego "
        "**cierra el issue**. Se publicarán con pausas. Para cambiar una respuesta, edita el issue y "
        "cambia solo el texto entre **Respuesta:** y el final de esa sección. Si cierras el issue como "
        f"*no planeado* no se publica nada. Estos borradores caducan en {expires_days} días.",
        "",
        "- [ ] Aprobar TODAS",
        "",
    ]
    for number, (thread_id, entry) in enumerate(queue, start=1):
        comment = "\n".join("> " + line for line in _safe(entry["comment"][:600]).splitlines()) or ">"
        lines += [
            "---",
            "",
            f"### {number}. [{entry['tier']}] `{entry['author']}` · estilo {entry.get('style')} · "
            f"parecido {entry.get('closest_similarity')}",
            f"<!-- thread:{thread_id} -->",
            comment,
            "",
            "- [ ] Publicar esta",
            "",
            "**Respuesta:**",
            _safe(entry["reply"]),
            "<!-- fin -->",
            "",
        ]
    return "\n".join(lines)


def parse_issue(body: str) -> tuple[bool, list[Approval]]:
    """Lee qué marcaste y el texto final de cada respuesta."""
    body = (body or "").replace("\r\n", "\n")
    markers = list(THREAD_RE.finditer(body))
    # "Aprobar TODAS" solo cuenta en el encabezado, antes del primer borrador.
    approve_all = bool(ALL_RE.search(body[: markers[0].start()] if markers else body))
    approvals = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        section = body[marker.end() : end]
        check = CHECK_RE.search(section)
        reply = REPLY_RE.search(section)
        approvals.append(
            Approval(
                thread_id=marker.group(1),
                checked=bool(check and check.group(1).lower() == "x"),
                reply=(reply.group(1) if reply else "").replace(ZWSP, "").replace("&lt;", "<").strip(),
            )
        )
    return approve_all, approvals


def result_body(summary: dict) -> str:
    """Lo que queda en el issue después de publicar: solo totales, sin textos de YouTube."""
    status = summary.get("status", {})
    lines = [
        MARKER,
        f"**Resultado ({summary.get('date', '')}):** "
        + (", ".join(f"{k}: {v}" for k, v in status.items()) or "nada que publicar"),
        "",
        "El texto de los comentarios y respuestas se borró de aquí (las políticas de YouTube no permiten "
        "guardarlo indefinidamente). El detalle queda 7 días en el artefacto *reporte* de la ejecución.",
    ]
    if summary.get("hidden_by_youtube"):
        lines += ["", "**Atención:** YouTube ocultó una respuesta (posible spam) y el agente se detuvo."]
    return "\n".join(lines)


EXPIRED_BODY = (
    f"{MARKER}\nEstos borradores caducaron sin aprobación y no se publicó nada. "
    "Su contenido se borró por las políticas de datos de YouTube."
)


class GitHubIssues:
    """Lo mínimo de la API de GitHub para manejar los issues de aprobación."""

    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo

    @classmethod
    def from_env(cls) -> "GitHubIssues":
        return cls(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])

    def _call(self, method: str, path: str, data: dict | None = None):
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}{path}",
            method=method,
            data=json.dumps(data).encode() if data is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            content = resp.read()
        return json.loads(content) if content else None

    def ensure_label(self) -> None:
        try:
            self._call("GET", f"/labels/{LABEL}")
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            self._call("POST", "/labels", {"name": LABEL, "color": "d93f0b", "description": "Borradores del agente de YouTube"})

    def create(self, title: str, body: str) -> int:
        return self._call("POST", "/issues", {"title": title, "body": body, "labels": [LABEL]})["number"]

    def get(self, number: int) -> dict:
        return self._call("GET", f"/issues/{number}")

    def update(self, number: int, **fields) -> None:
        self._call("PATCH", f"/issues/{number}", fields)

    def open_drafts(self) -> list[dict]:
        issues = self._call("GET", f"/issues?state=open&labels={LABEL}&per_page=100") or []
        return [i for i in issues if i.get("user", {}).get("login") == BOT_LOGIN and "pull_request" not in i]


def is_agent_issue(issue: dict) -> bool:
    labels = {label.get("name") for label in issue.get("labels", [])}
    return (
        LABEL in labels
        and issue.get("user", {}).get("login") == BOT_LOGIN
        and MARKER in (issue.get("body") or "")
    )
