"""Los dos comandos del agente: ``train`` (aprender tu estilo) y ``run`` (responder)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .responder import build_instructions
from .style import ExampleIndex, build_profile, extract_pairs
from .youtube import WRITE_COST, QuotaExceeded, Thread

log = logging.getLogger(__name__)

MEMBER = "miembro"
SUBSCRIBER = "suscriptor"
OTHER = "otro"


# -- archivos de estado -----------------------------------------------------------


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class Store:
    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.pairs_path = data_dir / "training_pairs.json"
        self.profile_path = data_dir / "style_profile.json"
        self.posted_path = data_dir / "posted.json"
        self.state_path = data_dir / "state.json"

    def posted(self) -> dict:
        return _load_json(self.posted_path, {})

    def state(self) -> dict:
        return _load_json(self.state_path, {})

    def has_training(self) -> bool:
        return self.pairs_path.exists() and self.profile_path.exists()


def load_member_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line}


# -- entrenamiento ---------------------------------------------------------------


def train(cfg: Config, yt, owner_id: str | None = None) -> dict:
    store = Store(cfg.data_dir)
    owner_id = owner_id or yt.my_channel_id()
    exclude = {entry["reply_id"] for entry in store.posted().values() if entry.get("reply_id")}

    # Nunca aprender de respuestas publicadas por el propio agente.
    before = cfg.train_before
    live_since = store.state().get("live_since")
    if live_since:
        live_dt = datetime.fromisoformat(live_since)
        before = min(before, live_dt) if before else live_dt

    threads: list[Thread] = []
    scanned = 0
    try:
        for thread in yt.iter_threads(owner_id):
            scanned += 1
            if thread.replies_truncated:
                thread = yt.complete_replies(thread)
            if any(r.author_channel_id == owner_id for r in thread.replies):
                threads.append(thread)
    except QuotaExceeded:
        log.warning("Se acabó el presupuesto de cuota de YouTube; entreno con lo leído hasta ahora.")

    pairs = extract_pairs(threads, owner_id, exclude_reply_ids=exclude, before=before)
    if not pairs:
        raise SystemExit(
            f"Revisé {scanned} hilos y no encontré respuestas tuyas a comentarios de otras personas. "
            "No hay con qué entrenar."
        )
    profile = build_profile(pairs, cfg.canonical_examples)
    _save_json(store.pairs_path, pairs)
    _save_json(store.profile_path, profile)
    log.info(
        "Entrenado con %d respuestas tuyas (de %d hilos revisados). Cuota de YouTube usada: %d.",
        len(pairs), scanned, yt.units_used,
    )
    return profile


# -- selección de comentarios ------------------------------------------------------


@dataclass
class Candidate:
    thread: Thread
    tier: str


def select_candidates(
    threads: list[Thread],
    owner_id: str,
    members: set[str],
    subscribers: set[str],
    handled: set[str],
    since: datetime,
    max_members: int,
    max_subscribers: int,
    max_per_author: int,
    include_non_subscribers: bool = False,
) -> list[Candidate]:
    """Primero todos los miembros (hasta el tope); luego unos pocos suscriptores."""
    by_tier: dict[str, list[Thread]] = {MEMBER: [], SUBSCRIBER: [], OTHER: []}
    for thread in threads:
        author = thread.top.author_channel_id
        if (
            not author
            or author == owner_id
            or not thread.can_reply
            or thread.id in handled
            or thread.top.published_at < since
            or not thread.top.text.strip()
            or any(r.author_channel_id == owner_id for r in thread.replies)
        ):
            continue
        tier = MEMBER if author in members else SUBSCRIBER if author in subscribers else OTHER
        by_tier[tier].append(thread)

    # Miembros: del más viejo al más nuevo, para que ninguno se quede sin respuesta.
    by_tier[MEMBER].sort(key=lambda t: t.top.published_at)
    # Suscriptores: los comentarios con más "me gusta" primero.
    for tier in (SUBSCRIBER, OTHER):
        by_tier[tier].sort(key=lambda t: (t.top.like_count, t.top.published_at), reverse=True)

    per_author: Counter[str] = Counter()
    chosen: list[Candidate] = []

    def take(pool: list[Thread], limit: int, label: str) -> None:
        taken = 0
        for thread in pool:
            if taken >= limit:
                return
            author = thread.top.author_channel_id
            if per_author[author] >= max_per_author:
                continue
            per_author[author] += 1
            chosen.append(Candidate(thread, label))
            taken += 1

    take(by_tier[MEMBER], max_members, MEMBER)
    before = len(chosen)
    take(by_tier[SUBSCRIBER], max_subscribers, SUBSCRIBER)
    if include_non_subscribers:
        take(by_tier[OTHER], max_subscribers - (len(chosen) - before), OTHER)
    return chosen


# -- ejecución diaria --------------------------------------------------------------


def run(cfg: Config, yt, responder, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = Store(cfg.data_dir)
    owner_id = yt.my_channel_id()

    if not store.has_training():
        log.info("No hay entrenamiento guardado; entrenando primero.")
        train(cfg, yt, owner_id)
    pairs = _load_json(store.pairs_path, [])
    profile = _load_json(store.profile_path, {})

    members = yt.member_ids()
    if members is None:
        members = load_member_file(cfg.members_file)
        log.info("Uso la lista de miembros de %s (%d canales).", cfg.members_file, len(members))
    subscribers = yt.public_subscriber_ids() if cfg.max_subscriber_replies > 0 else set()

    since = now - timedelta(days=cfg.lookback_days)
    threads = []
    for thread in yt.iter_threads(owner_id, since=since):
        if thread.top.published_at < since or thread.top.author_channel_id == owner_id:
            continue
        if thread.replies_truncated and not any(r.author_channel_id == owner_id for r in thread.replies):
            thread = yt.complete_replies(thread)
        threads.append(thread)

    posted = store.posted()
    candidates = select_candidates(
        threads, owner_id, members, subscribers, set(posted), since,
        cfg.max_member_replies, cfg.max_subscriber_replies, cfg.max_replies_per_author,
        cfg.include_non_subscribers,
    )
    summary = {
        "date": now.date().isoformat(),
        "dry_run": cfg.dry_run,
        "threads_checked": len(threads),
        "candidates": Counter(c.tier for c in candidates),
        "posted": 0,
        "skipped_by_model": 0,
        "entries": [],
    }
    if not candidates:
        log.info("No hay comentarios nuevos que responder.")
        return _finish(store, summary, yt, responder)

    titles = yt.video_titles(c.thread.video_id for c in candidates)
    index = ExampleIndex(pairs)
    canonical_replies = [ex["reply"] for ex in profile["canonical_examples"]]
    items = [
        {
            "id": c.thread.id,
            "tier": c.tier,
            "video": titles.get(c.thread.video_id, ""),
            "comentario": c.thread.top.text[:1500],
            "ejemplos_parecidos": index.search(
                c.thread.top.text, cfg.examples_per_comment, skip_replies=canonical_replies
            ),
        }
        for c in candidates
    ]
    replies = responder.generate(build_instructions(profile), items, profile, cfg.batch_size)

    state = store.state()
    for cand in candidates:
        reply = replies.get(cand.thread.id)
        entry = {"tier": cand.tier, "author": cand.thread.top.author_name, "comment": cand.thread.top.text, "reply": reply}
        summary["entries"].append(entry)
        if not reply:
            summary["skipped_by_model"] += 1
            continue
        if cfg.dry_run:
            continue
        if not yt.can_spend(WRITE_COST):
            log.warning("Sin cuota de YouTube para publicar más respuestas hoy.")
            entry["reply"] = None
            break
        if "live_since" not in state:
            state["live_since"] = now.isoformat()
            _save_json(store.state_path, state)
        reply_id = yt.reply(cand.thread.id, reply)
        posted[cand.thread.id] = {"reply_id": reply_id, "posted_at": now.isoformat()}
        _save_json(store.posted_path, posted)
        summary["posted"] += 1
    return _finish(store, summary, yt, responder)


def _finish(store: Store, summary: dict, yt, responder) -> dict:
    usage = responder.usage
    summary["candidates"] = dict(summary["candidates"])
    summary["youtube_quota_units"] = yt.units_used
    summary["openai"] = {
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_tokens,
        "output_tokens": usage.output_tokens,
    }
    _save_json(store.dir / "reports" / f"{summary['date']}.json", summary)
    _write_markdown_report(store.dir / "reports" / f"{summary['date']}.md", summary)
    return summary


def _write_markdown_report(path: Path, summary: dict) -> None:
    mode = "SIMULACIÓN (no se publicó nada)" if summary["dry_run"] else "EN VIVO"
    lines = [
        f"# Respuestas del {summary['date']} — {mode}",
        "",
        f"- Hilos revisados: {summary['threads_checked']}",
        f"- Seleccionados: {summary['candidates']}",
        f"- Publicadas: {summary['posted']}",
        f"- Omitidas por el modelo: {summary['skipped_by_model']}",
        f"- Cuota YouTube: {summary['youtube_quota_units']} unidades",
        f"- OpenAI: {summary['openai']}",
        "",
    ]
    for entry in summary["entries"]:
        lines += [
            f"## [{entry['tier']}] {entry['author']}",
            "",
            "> " + entry["comment"].replace("\n", "\n> "),
            "",
            entry["reply"] or "_(sin respuesta)_",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
