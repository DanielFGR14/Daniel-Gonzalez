"""Uso:

    python -m agent train   # (re)aprende tu estilo con tus respuestas reales
    python -m agent run     # responde los comentarios nuevos (o los simula si DRY_RUN=true)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from google.auth.exceptions import RefreshError

from .config import Config
from .pipeline import run, train, wipe_data
from .youtube import QuotaExceeded, YouTube

EXIT_TOKEN_INVALID = 3
EXIT_HIDDEN_BY_YOUTUBE = 4


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("command", choices=["train", "run"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    required = ["YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"]
    if args.command == "run":
        required.append("OPENAI_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print(f"Faltan estos secretos: {', '.join(missing)}. Mira el README.", file=sys.stderr)
        return 2

    cfg = Config.from_env()
    yt = YouTube.from_env(cfg.quota_budget)

    try:
        if args.command == "train":
            profile = train(cfg, yt)
            print(f"Estilo aprendido de {profile['examples_total']} respuestas tuyas.")
            return 0

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
