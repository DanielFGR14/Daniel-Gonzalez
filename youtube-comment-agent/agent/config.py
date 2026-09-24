"""Configuración del agente, leída de variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _str(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _int(name: str, default: int) -> int:
    value = _str(name)
    return int(value) if value else default


def _bool(name: str, default: bool) -> bool:
    value = _str(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "si", "sí", "on"}


def _date(name: str) -> datetime | None:
    value = _str(name)
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Config:
    # OpenAI
    openai_model: str = "gpt-6-luna"
    reasoning_effort: str = "none"
    service_tier: str = "flex"
    batch_size: int = 10

    # Qué responder
    dry_run: bool = True
    lookback_days: int = 3
    max_member_replies: int = 50
    max_subscriber_replies: int = 10
    max_replies_per_author: int = 2
    include_non_subscribers: bool = False

    # Entrenamiento
    examples_per_comment: int = 3
    canonical_examples: int = 20
    train_before: datetime | None = None

    # YouTube: cuota diaria gratuita es 10.000 unidades; dejamos margen.
    quota_budget: int = 9000

    data_dir: Path = Path("data")
    members_file: Path = Path("members.txt")

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_model=_str("OPENAI_MODEL", cls.openai_model),
            reasoning_effort=_str("OPENAI_REASONING_EFFORT", cls.reasoning_effort),
            service_tier=_str("OPENAI_SERVICE_TIER", cls.service_tier),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            dry_run=_bool("DRY_RUN", cls.dry_run),
            lookback_days=_int("LOOKBACK_DAYS", cls.lookback_days),
            max_member_replies=_int("MAX_MEMBER_REPLIES", cls.max_member_replies),
            max_subscriber_replies=_int("MAX_SUBSCRIBER_REPLIES", cls.max_subscriber_replies),
            max_replies_per_author=_int("MAX_REPLIES_PER_AUTHOR", cls.max_replies_per_author),
            include_non_subscribers=_bool("INCLUDE_NON_SUBSCRIBERS", cls.include_non_subscribers),
            examples_per_comment=_int("EXAMPLES_PER_COMMENT", cls.examples_per_comment),
            canonical_examples=_int("CANONICAL_EXAMPLES", cls.canonical_examples),
            train_before=_date("TRAIN_BEFORE"),
            quota_budget=_int("YT_QUOTA_BUDGET", cls.quota_budget),
            data_dir=Path(_str("DATA_DIR", str(cls.data_dir))),
            members_file=Path(_str("MEMBERS_FILE", str(cls.members_file))),
        )
