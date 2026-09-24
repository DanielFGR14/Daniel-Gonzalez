"""Los dos comandos del agente: ``train`` (aprender tu estilo) y ``run`` (responder)."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError

from .config import Config
from .quality import NoveltyChecker, choose
from .responder import build_instructions
from .style import ExampleIndex, build_profile, extract_pairs
from .youtube import WRITE_COST, QuotaExceeded, Thread, error_reason

log = logging.getLogger(__name__)

MEMBER = "miembro"
SUBSCRIBER = "suscriptor"
OTHER = "otro"

LOCAL_TZ = ZoneInfo("America/Bogota")

# Si YouTube responde con alguno de estos errores, se deja de publicar por hoy: son problemas
# de la cuenta o de límites, no de un comentario en particular.
STOP_REASONS = {
    "quotaExceeded", "rateLimitExceeded", "userRateLimitExceeded", "dailyLimitExceeded",
    "forbidden", "insufficientPermissions", "ineligibleAccount", "authError", "401",
}

# Un comentario cuyas opciones fueron descartadas se reintenta, como mucho, estas veces en total.
MAX_ATTEMPTS = 2


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
        self.attempts_path = data_dir / "attempts.json"
        self.state_path = data_dir / "state.json"

    def posted(self) -> dict:
        return _load_json(self.posted_path, {})

    def attempts(self) -> dict:
        return _load_json(self.attempts_path, {})

    def state(self) -> dict:
        return _load_json(self.state_path, {})

    def profile(self) -> dict:
        return _load_json(self.profile_path, {})

    def has_training(self) -> bool:
        return self.pairs_path.exists() and self.profile_path.exists()


def load_member_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    # Lo que va después de " #" es comentario (sin tocar los @handles).
    lines = (line.split(" #", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def manual_members(cfg: Config) -> set[str]:
    """IDs de canal tal cual y @handles en minúscula, de MEMBER_CHANNEL_IDS y del archivo."""
    entries = set(cfg.member_list) | load_member_file(cfg.members_file)
    return {e.lower() if e.startswith("@") else e for e in entries}


def wipe_data(data_dir: Path) -> None:
    """Borra todo lo guardado de YouTube (lo exigen sus políticas si el token deja de servir).

    Se conserva state.json (solo guarda la fecha en que el agente empezó a publicar, para no
    aprender nunca de sus propias respuestas) y se deja una marca para que la caché de GitHub
    guarde esta versión vacía; las copias viejas se borran solas a los 7 días sin uso.
    """
    if data_dir.exists():
        for path in data_dir.rglob("*"):
            if path.is_file() and path.name != "state.json":
                path.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".wiped").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def prune_posted(posted: dict, now: datetime, retention_days: int, key: str = "posted_at") -> dict:
    """Borra del registro lo que tenga más de ``retention_days`` días."""
    limit = now - timedelta(days=retention_days)
    return {k: v for k, v in posted.items() if datetime.fromisoformat(v[key]) >= limit}


def _attempt(attempts: dict, thread_id: str, now: datetime, final: bool = False) -> None:
    previous = attempts.get(thread_id, {})
    attempts[thread_id] = {
        "count": previous.get("count", 0) + 1,
        "last": now.isoformat(),
        "final": final or previous.get("final", False),
    }


def needs_training(store: Store, now: datetime, retrain_days: int) -> bool:
    if not store.has_training():
        return True
    trained_at = store.profile().get("trained_at")
    return not trained_at or now - datetime.fromisoformat(trained_at) > timedelta(days=retrain_days)


# -- entrenamiento ---------------------------------------------------------------


def train(cfg: Config, yt, owner_id: str | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
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
    profile["trained_at"] = now.isoformat()
    # Se sobrescribe todo: así lo borrado en YouTube también desaparece de aquí.
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
        is_member = author in members or thread.top.author_name.lower() in members
        tier = MEMBER if is_member else SUBSCRIBER if author in subscribers else OTHER
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


def run(
    cfg: Config,
    yt,
    responder,
    now: datetime | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
    rng: random.Random | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    rng = rng or random.Random()
    store = Store(cfg.data_dir)
    owner_id = yt.my_channel_id()

    posted = prune_posted(store.posted(), now, cfg.retention_days)
    _save_json(store.posted_path, posted)
    attempts = prune_posted(store.attempts(), now, cfg.retention_days, key="last")
    _save_json(store.attempts_path, attempts)
    # No volver a gastar tokens en lo ya publicado, en lo que el modelo decidió no responder
    # ni en lo que ya se intentó MAX_ATTEMPTS veces.
    handled = set(posted) | {
        tid for tid, a in attempts.items() if a.get("final") or a.get("count", 0) >= MAX_ATTEMPTS
    }

    if needs_training(store, now, cfg.retrain_days):
        log.info("Entrenamiento ausente o con más de %d días; entrenando de nuevo.", cfg.retrain_days)
        train(cfg, yt, owner_id, now)
    pairs = _load_json(store.pairs_path, [])
    profile = store.profile()

    if not cfg.dry_run and not cfg.train_before:
        log.warning("Modo en vivo sin TRAIN_BEFORE: defínela para blindar el entrenamiento (ver README).")

    members = manual_members(cfg)
    api_members = yt.member_ids()
    if api_members is not None:
        members |= api_members
    elif not members:
        log.warning(
            "No tengo lista de miembros: YouTube no dio acceso a members.list y MEMBER_CHANNEL_IDS "
            "está vacía. Todos los comentarios se tratarán como de no miembros (ver README)."
        )
    subscribers = yt.public_subscriber_ids() if cfg.max_subscriber_replies > 0 else set()

    since = now - timedelta(days=cfg.lookback_days)
    threads = []
    for thread in yt.iter_threads(owner_id, since=since):
        if thread.top.published_at < since or thread.top.author_channel_id == owner_id:
            continue
        if thread.replies_truncated and not any(r.author_channel_id == owner_id for r in thread.replies):
            thread = yt.complete_replies(thread)
        threads.append(thread)

    candidates = select_candidates(
        threads, owner_id, members, subscribers, handled, since,
        cfg.max_member_replies, cfg.max_subscriber_replies, cfg.max_replies_per_author,
        cfg.include_non_subscribers,
    )
    summary = {
        # Fecha de Colombia: a las 9 p. m. allá, en UTC ya es el día siguiente.
        "date": now.astimezone(LOCAL_TZ).date().isoformat(),
        "dry_run": cfg.dry_run,
        "threads_checked": len(threads),
        "candidates": dict(Counter(c.tier for c in candidates)),
        "status": Counter(),
        "hidden_by_youtube": False,
        "entries": [],
    }
    if not candidates:
        log.info("No hay comentarios nuevos que responder.")
        return _finish(cfg, summary, yt, responder)

    # 1. Generar opciones (pocas llamadas, en lotes).
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
    options = responder.generate(build_instructions(profile, cfg.variants), items, profile, cfg.batch_size)

    # 2. Elegir la mejor opción de cada uno: con tu estilo, pero distinta de todo lo ya escrito.
    novelty = NoveltyChecker([p["reply"] for p in pairs] + [v.get("text", "") for v in posted.values()])
    queue = []
    for cand in candidates:
        entry = {"tier": cand.tier, "author": cand.thread.top.author_name, "comment": cand.thread.top.text, "reply": None}
        summary["entries"].append(entry)
        tid = cand.thread.id
        if tid not in options:
            entry["status"] = "sin respuesta del modelo (se reintenta mañana)"
            continue
        if options[tid] is None:
            entry["status"] = "omitido por el modelo"
            _attempt(attempts, tid, now, final=True)
            continue
        if not options[tid]:
            entry["status"] = "descartada: opciones con enlaces, hashtags o demasiado largas"
            _attempt(attempts, tid, now)
            continue
        context = f"{cand.thread.top.text} {titles.get(cand.thread.video_id, '')}"
        best = choose(options[tid], profile, novelty, context, cfg.max_similarity, cfg.min_style_score)
        entry.update(style=best.style, closest_similarity=best.closest_similarity)
        if not best.accepted:
            entry.update(status=f"descartada: {best.reason}", rejected_option=best.text)
            _attempt(attempts, tid, now)
            continue
        novelty.add(best.text)  # las siguientes de esta noche tampoco podrán parecerse a esta
        entry["reply"] = best.text
        queue.append((cand, entry))
    _save_json(store.attempts_path, attempts)

    # 3. Publicar despacio, con pausas aleatorias entre respuestas.
    state = store.state()
    started = clock()
    posted_any = False
    unverified = None  # la última publicada, hasta confirmar que YouTube la muestra

    def stop(from_position: int, reason: str) -> None:
        for _, pending in queue[from_position:]:
            pending["status"] = f"pendiente: {reason}"

    for position, (cand, entry) in enumerate(queue):
        if cfg.dry_run:
            entry["status"] = "simulada (no publicada)"
            continue
        if posted_any:
            delay = rng.uniform(cfg.min_delay_seconds, cfg.max_delay_seconds)
            if clock() - started + delay > cfg.max_run_minutes * 60:
                stop(position, "se acabó el tiempo de hoy")
                break
            sleep(delay)
            # Si YouTube ocultó la respuesta anterior (su señal de posible spam), se para por hoy.
            if unverified and not _confirm_visible(yt, unverified, sleep, cfg.min_delay_seconds):
                summary["hidden_by_youtube"] = True
                stop(position, "YouTube ocultó la respuesta anterior; se detuvo por hoy")
                unverified = None
                break
            unverified = None
        if not yt.can_spend(WRITE_COST + 6):
            stop(position, "sin cuota de YouTube hoy")
            break

        try:
            # Justo antes de publicar: ¿sigue existiendo y nadie (tú) lo respondió mientras tanto?
            current = yt.fetch_thread(cand.thread.id)
            if current is None:
                entry["status"] = "el comentario ya no existe"
                continue
            if any(r.author_channel_id == owner_id for r in current.replies):
                entry["status"] = "ya lo respondiste tú"
                continue
            if not current.can_reply:
                entry["status"] = "ya no admite respuestas"
                continue

            if "live_since" not in state:
                state["live_since"] = now.isoformat()
                _save_json(store.state_path, state)
            reply_id = yt.reply(cand.thread.id, entry["reply"])
        except HttpError as err:
            reason = error_reason(err) or str(err.status_code)
            entry["status"] = f"error de YouTube: {reason}"
            if reason in STOP_REASONS:
                log.warning("YouTube respondió %s; dejo de publicar por hoy.", reason)
                stop(position + 1, f"YouTube respondió {reason}")
                break
            continue
        posted[cand.thread.id] = {"reply_id": reply_id, "posted_at": now.isoformat(), "text": entry["reply"]}
        _save_json(store.posted_path, posted)
        entry["status"] = "publicada"
        posted_any = True
        unverified = (cand.thread.id, reply_id, entry)

    if unverified:  # verificar también la última
        sleep(cfg.min_delay_seconds)
        if not _confirm_visible(yt, unverified, sleep, cfg.min_delay_seconds):
            summary["hidden_by_youtube"] = True
    return _finish(cfg, summary, yt, responder)


def _confirm_visible(yt, published: tuple, sleep, wait: float) -> bool:
    """True si la respuesta aparece en el hilo (con un reintento, por si YouTube tarda en mostrarla)."""
    thread_id, reply_id, entry = published
    try:
        for attempt in range(2):
            if yt.is_reply_visible(thread_id, reply_id):
                return True
            if attempt == 0:
                sleep(wait)
    except (HttpError, QuotaExceeded):
        return True  # no se pudo comprobar; eso no es una señal de spam
    entry["status"] = "publicada, pero YouTube la ocultó (posible spam)"
    log.warning("YouTube no muestra la respuesta %s: posible filtro de spam.", reply_id)
    return False


def _finish(cfg: Config, summary: dict, yt, responder) -> dict:
    usage = responder.usage
    summary["status"] = dict(Counter(e.get("status", "sin procesar") for e in summary["entries"]))
    summary["youtube_quota_units"] = yt.units_used
    summary["openai"] = {
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_tokens,
        "output_tokens": usage.output_tokens,
    }
    # Los reportes van fuera de data/: no se guardan en la caché, solo como artefacto temporal.
    _save_json(cfg.reports_dir / f"{summary['date']}.json", summary)
    _write_markdown_report(cfg.reports_dir / f"{summary['date']}.md", summary)
    return summary


def _write_markdown_report(path: Path, summary: dict) -> None:
    mode = "SIMULACIÓN (no se publicó nada)" if summary["dry_run"] else "EN VIVO"
    lines = [
        f"# Respuestas del {summary['date']} — {mode}",
        "",
    ]
    if summary.get("hidden_by_youtube"):
        lines += [
            "> **Atención:** YouTube ocultó una respuesta (posible filtro de spam) y el agente se detuvo. "
            "Considera bajar MAX_MEMBER_REPLIES o subir las pausas.",
            "",
        ]
    lines += [
        f"- Hilos revisados: {summary['threads_checked']}",
        f"- Seleccionados: {summary['candidates']}",
        f"- Resultado: {summary['status']}",
        f"- Cuota YouTube: {summary['youtube_quota_units']} unidades",
        f"- OpenAI: {summary['openai']}",
        "",
        "_estilo_ = qué tanto suena a ti (0 a 1). _parecido_ = similitud con tu respuesta real más "
        "cercana (1 = copia exacta; por encima del límite se descarta).",
        "",
    ]
    for entry in summary["entries"]:
        scores = ""
        if "style" in entry:
            scores = f" · estilo {entry['style']} · parecido {entry['closest_similarity']}"
        if entry["reply"]:
            body = entry["reply"]
        elif entry.get("rejected_option"):
            body = f"_(opción descartada: {entry['rejected_option']})_"
        else:
            body = "_(sin respuesta)_"
        lines += [
            f"## [{entry['tier']}] {entry['author']} — {entry.get('status', '')}{scores}",
            "",
            "> " + entry["comment"].replace("\n", "\n> "),
            "",
            body,
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
