"""Acceso a la YouTube Data API v3 (lectura de comentarios, miembros y respuestas)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

TOKEN_URI = "https://oauth2.googleapis.com/token"

# Coste en unidades de cuota de cada tipo de llamada (documentación de YouTube).
READ_COST = 1
WRITE_COST = 50


class QuotaExceeded(RuntimeError):
    pass


def error_reason(err: Exception) -> str:
    """El 'reason' de un HttpError de Google (p. ej. quotaExceeded, commentsDisabled)."""
    details = getattr(err, "error_details", None)
    if isinstance(details, list) and details and isinstance(details[0], dict):
        return details[0].get("reason", "") or ""
    return getattr(err, "reason", "") or ""


@dataclass
class Comment:
    id: str
    author_channel_id: str
    author_name: str
    text: str
    published_at: datetime
    like_count: int = 0


@dataclass
class Thread:
    id: str
    video_id: str
    top: Comment
    replies: list[Comment] = field(default_factory=list)
    total_reply_count: int = 0
    can_reply: bool = True

    @property
    def replies_truncated(self) -> bool:
        return self.total_reply_count > len(self.replies)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_comment(item: dict) -> Comment:
    sn = item["snippet"]
    return Comment(
        id=item["id"],
        author_channel_id=(sn.get("authorChannelId") or {}).get("value", ""),
        author_name=sn.get("authorDisplayName", ""),
        # textOriginal solo llega para los comentarios del usuario autenticado.
        text=sn.get("textOriginal") or sn.get("textDisplay", ""),
        published_at=_parse_time(sn["publishedAt"]),
        like_count=sn.get("likeCount", 0),
    )


def parse_thread(item: dict) -> Thread:
    sn = item["snippet"]
    replies = [parse_comment(c) for c in (item.get("replies") or {}).get("comments", [])]
    replies.sort(key=lambda c: c.published_at)
    return Thread(
        id=item["id"],
        video_id=sn.get("videoId", ""),
        top=parse_comment(sn["topLevelComment"]),
        replies=replies,
        total_reply_count=sn.get("totalReplyCount", 0),
        can_reply=sn.get("canReply", True),
    )


class YouTube:
    def __init__(self, service, quota_budget: int = 9000):
        self.api = service
        self.quota_budget = quota_budget
        self.units_used = 0

    @classmethod
    def from_env(cls, quota_budget: int = 9000) -> "YouTube":
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(
            token=None,
            refresh_token=os.environ["YT_REFRESH_TOKEN"],
            client_id=os.environ["YT_CLIENT_ID"],
            client_secret=os.environ["YT_CLIENT_SECRET"],
            token_uri=TOKEN_URI,
        )
        service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        return cls(service, quota_budget)

    # -- utilidades internas -------------------------------------------------

    def can_spend(self, units: int) -> bool:
        return self.units_used + units <= self.quota_budget

    def _execute(self, request, cost: int = READ_COST) -> dict:
        if not self.can_spend(cost):
            raise QuotaExceeded(f"presupuesto de cuota agotado ({self.units_used}/{self.quota_budget})")
        self.units_used += cost
        return request.execute(num_retries=3)

    # -- lecturas ---------------------------------------------------------------

    def my_channel_id(self) -> str:
        resp = self._execute(self.api.channels().list(part="id", mine=True))
        items = resp.get("items") or []
        if not items:
            raise RuntimeError("La cuenta autorizada no tiene canal de YouTube.")
        return items[0]["id"]

    def iter_threads(self, channel_id: str, since: datetime | None = None) -> Iterator[Thread]:
        """Recorre los hilos de comentarios de todos los videos del canal, del más nuevo al más viejo.

        Con ``since`` deja de paginar cuando una página entera es más vieja que esa fecha.
        """
        page_token = None
        while True:
            resp = self._execute(
                self.api.commentThreads().list(
                    part="snippet,replies",
                    allThreadsRelatedToChannelId=channel_id,
                    maxResults=100,
                    order="time",
                    textFormat="plainText",
                    pageToken=page_token,
                )
            )
            threads = [parse_thread(item) for item in resp.get("items", [])]
            yield from threads
            page_token = resp.get("nextPageToken")
            if not page_token:
                return
            if since and threads and all(t.top.published_at < since for t in threads):
                return

    def fetch_thread(self, thread_id: str) -> Thread | None:
        """El hilo tal como está ahora (None si lo borraron)."""
        resp = self._execute(
            self.api.commentThreads().list(part="snippet,replies", id=thread_id, textFormat="plainText")
        )
        items = resp.get("items") or []
        if not items:
            return None
        thread = parse_thread(items[0])
        return self.complete_replies(thread) if thread.replies_truncated else thread

    def complete_replies(self, thread: Thread) -> Thread:
        """commentThreads.list solo trae algunas respuestas; esto trae todas."""
        replies: list[Comment] = []
        page_token = None
        while True:
            resp = self._execute(
                self.api.comments().list(
                    part="snippet",
                    parentId=thread.id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=page_token,
                )
            )
            replies.extend(parse_comment(item) for item in resp.get("items", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        replies.sort(key=lambda c: c.published_at)
        thread.replies = replies
        thread.total_reply_count = max(thread.total_reply_count, len(replies))
        return thread

    def member_ids(self) -> set[str] | None:
        """IDs de canal de los miembros actuales. None si la API no lo permite."""
        from googleapiclient.errors import HttpError

        ids: set[str] = set()
        page_token = None
        try:
            while True:
                resp = self._execute(
                    self.api.members().list(
                        part="snippet", mode="all_current", maxResults=1000, pageToken=page_token
                    )
                )
                for item in resp.get("items", []):
                    member_id = item["snippet"].get("memberDetails", {}).get("channelId")
                    if member_id:
                        ids.add(member_id)
                page_token = resp.get("nextPageToken")
                if not page_token:
                    return ids
        except HttpError as err:
            log.warning("No pude leer la lista de miembros (%s).", err.status_code)
            return None

    def public_subscriber_ids(self) -> set[str]:
        """Suscriptores que tienen sus suscripciones públicas (la API no ve a los demás)."""
        from googleapiclient.errors import HttpError

        ids: set[str] = set()
        page_token = None
        try:
            while True:
                resp = self._execute(
                    self.api.subscriptions().list(
                        part="subscriberSnippet", mySubscribers=True, maxResults=50, pageToken=page_token
                    )
                )
                for item in resp.get("items", []):
                    sub_id = item.get("subscriberSnippet", {}).get("channelId")
                    if sub_id:
                        ids.add(sub_id)
                page_token = resp.get("nextPageToken")
                if not page_token:
                    return ids
        except HttpError as err:
            log.warning("No pude leer la lista de suscriptores (%s).", err.status_code)
            return ids

    def video_titles(self, video_ids: Iterable[str]) -> dict[str, str]:
        ids = sorted({v for v in video_ids if v})
        titles: dict[str, str] = {}
        for start in range(0, len(ids), 50):
            chunk = ids[start : start + 50]
            resp = self._execute(self.api.videos().list(part="snippet", id=",".join(chunk), maxResults=50))
            for item in resp.get("items", []):
                titles[item["id"]] = item["snippet"].get("title", "")
        return titles

    # -- escritura -------------------------------------------------------------

    def reply(self, parent_id: str, text: str) -> str:
        resp = self._execute(
            self.api.comments().insert(
                part="snippet", body={"snippet": {"parentId": parent_id, "textOriginal": text}}
            ),
            cost=WRITE_COST,
        )
        return resp["id"]
