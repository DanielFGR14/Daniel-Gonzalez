"""Uso:

    python -m agent train              # (re)aprende tu estilo con tus respuestas reales
    python -m agent draft              # prepara borradores y abre un issue para que los apruebes
    python -m agent publish --issue N  # publica lo que aprobaste en el issue N (al cerrarlo)
    python -m agent run                # modo automático, sin aprobación (o simulación si DRY_RUN=true)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from google.auth.exceptions import RefreshError

from .approval import EXPIRED_BODY, GitHubIssues, is_agent_issue, parse_issue, render_issue, result_body
from .config import Config
from .pipeline import draft, publish_approved, run, set_draft_issue, train, wipe_data
from .youtube import QuotaExceeded, YouTube

EXIT_TOKEN_INVALID = 3
EXIT_HIDDEN_BY_YOUTUBE = 4

log = logging.getLogger("agent")


def _required(command: str) -> list[str]:
    names = ["YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"]
    if command in ("draft", "run"):
        names.append("OPENAI_API_KEY")
    if command in ("draft", "publish"):
        names += ["GITHUB_TOKEN", "GITHUB_REPOSITORY"]
    return names


def _expire_old_issues(gh: GitHubIssues, days: int) -> None:
    limit = datetime.now(timezone.utc) - timedelta(days=days)
    for issue in gh.open_drafts():
        created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        if created < limit:
            gh.update(issue["number"], body=EXPIRED_BODY, state="closed", state_reason="not_planned")
            log.info("Issue #%s caducó sin aprobación; lo cerré y borré su contenido.", issue["number"])


def _cmd_draft(cfg: Config, yt) -> dict:
    from .responder import Responder

    gh = GitHubIssues.from_env()
    _expire_old_issues(gh, cfg.lookback_days)
    responder = Responder.from_env(cfg.openai_model, cfg.reasoning_effort, cfg.service_tier, cfg.variants)
    summary, queue = draft(cfg, yt, responder)
    if queue:
        owner = os.environ.get("GITHUB_REPOSITORY_OWNER") or gh.repo.split("/")[0]
        gh.ensure_label()
        number = gh.create(
            f"Respuestas por aprobar — {summary['date']} ({len(queue)})",
            render_issue(queue, owner, summary["date"], cfg.lookback_days),
        )
        set_draft_issue(cfg, [thread_id for thread_id, _ in queue], number)
        summary["issue"] = number
    return summary


def _cmd_publish(cfg: Config, yt, number: int) -> dict:
    gh = GitHubIssues.from_env()
    issue = gh.get(number)
    if not is_agent_issue(issue):
        raise SystemExit(f"El issue #{number} no es un issue de borradores del agente; no publico nada.")
    summary = None
    try:
        approve_all, approvals = parse_issue(issue.get("body") or "")
        summary = publish_approved(
            cfg, yt, approvals, approve_all, reject_all=issue.get("state_reason") == "not_planned"
        )
        return summary
    finally:
        # Pase lo que pase, el issue no se queda con textos de YouTube.
        gh.update(number, body=result_body(summary or {"status": {"error": "revisa el log de la ejecución"}}))


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("command", choices=["train", "draft", "publish", "run"])
    parser.add_argument("--issue", type=int, help="número del issue aprobado (para publish)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    missing = [name for name in _required(args.command) if not os.environ.get(name)]
    if missing:
        print(f"Faltan estos secretos: {', '.join(missing)}. Mira el README.", file=sys.stderr)
        return 2
    if args.command == "publish" and not args.issue:
        parser.error("publish necesita --issue N")

    cfg = Config.from_env()
    yt = YouTube.from_env(cfg.quota_budget)

    try:
        if args.command == "train":
            profile = train(cfg, yt)
            print(f"Estilo aprendido de {profile['examples_total']} respuestas tuyas.")
            return 0
        if args.command == "draft":
            summary = _cmd_draft(cfg, yt)
        elif args.command == "publish":
            summary = _cmd_publish(cfg, yt, args.issue)
        else:
            from .responder import Responder

            responder = Responder.from_env(cfg.openai_model, cfg.reasoning_effort, cfg.service_tier, cfg.variants)
            summary = run(cfg, yt, responder)
    except QuotaExceeded as err:
        print(f"Cuota de YouTube agotada por hoy: {err}", file=sys.stderr)
        return 1
    except RefreshError as err:
        # Las políticas de YouTube piden borrar los datos si el token se revoca o ya no se puede renovar.
        wipe_data(cfg.data_dir)
        print(
            f"El token de YouTube ya no sirve ({err}). Borré los datos guardados del canal. "
            "Genera un YT_REFRESH_TOKEN nuevo con get_refresh_token.py (ver README).",
            file=sys.stderr,
        )
        return EXIT_TOKEN_INVALID

    # En los logs solo van totales; el detalle (comentarios y respuestas) queda en reports/.
    totals = {k: v for k, v in summary.items() if k != "entries"}
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("## Agente de comentarios\n\n```json\n" + json.dumps(totals, ensure_ascii=False, indent=2) + "\n```\n")
    if summary.get("hidden_by_youtube"):
        # Falla a propósito para que GitHub te avise por correo.
        print("YouTube ocultó una respuesta como posible spam; el agente se detuvo. Revisa el reporte.", file=sys.stderr)
        return EXIT_HIDDEN_BY_YOUTUBE
    return 0


if __name__ == "__main__":
    sys.exit(main())
