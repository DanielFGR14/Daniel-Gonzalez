import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent.config import Config
from agent.pipeline import MEMBER, OTHER, SUBSCRIBER, run, select_candidates, train
from agent.responder import Responder, build_instructions, clean_reply
from agent.style import ExampleIndex, build_profile, extract_pairs
from agent.youtube import Comment, Thread, parse_thread

OWNER = "UCowner"
NOW = datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)


def c(cid, author, text, hours_ago=1, name=None, likes=0):
    return Comment(cid, author, name or f"@{author}", text, NOW - timedelta(hours=hours_ago), likes)


def thread(tid, top, replies=(), total=None, video="v1"):
    replies = list(replies)
    return Thread(tid, video, top, replies, len(replies) if total is None else total, True)


# -- entrenamiento -----------------------------------------------------------------


def test_extract_pairs_uses_only_owner_replies_and_finds_parent():
    t = thread(
        "t1",
        c("a", "UCfan", "Me encantó el video"),
        [
            c("b", OWNER, "Gracias parce!!"),
            c("c", "UCother", "Yo también lo amé", name="@otra"),
            c("d", OWNER, "@otra Qué bueno, gracias por estar aquí"),
            c("e", "UCfan", "Respuesta de un fan que no es del dueño"),
        ],
    )
    pairs = extract_pairs([t], OWNER)
    assert [(p["comment"], p["reply"]) for p in pairs] == [
        ("Me encantó el video", "Gracias parce!!"),
        ("Yo también lo amé", "Qué bueno, gracias por estar aquí"),
    ]


def test_extract_pairs_skips_agent_replies_and_owner_threads():
    own_top = thread("t1", c("a", OWNER, "Comentario fijado"), [c("b", OWNER, "Edit: link abajo")])
    fan = thread("t2", c("c", "UCfan", "Hola"), [c("d", OWNER, "Hola!", hours_ago=1)])
    assert extract_pairs([own_top, fan], OWNER, exclude_reply_ids={"d"}) == []
    assert extract_pairs([fan], OWNER, before=NOW - timedelta(hours=5)) == []


def sample_pairs():
    data = [
        ("Qué música usas?", "Es de mi autoría, gracias por preguntar 🙏"),
        ("Me ayudó a dormir", "Qué bueno!! Descansa mucho 🙏"),
        ("Gracias por el video", "Gracias a ti por verlo 🙏"),
        ("Me ayudó muchísimo a dormir", "Me alegra mucho!! Descansa 🙏"),
    ]
    return [
        {"comment": q, "reply": r, "reply_id": str(i), "video_id": "v", "published_at": f"2026-01-0{i + 1}"}
        for i, (q, r) in enumerate(data)
    ]


def test_build_profile_captures_style():
    profile = build_profile(sample_pairs(), canonical_count=2)
    assert profile["examples_total"] == 4
    assert profile["emoji_rate"] == 1.0
    assert profile["top_emojis"] == ["🙏"]
    assert "gracias" in profile["top_words"]
    assert len(profile["canonical_examples"]) == 2
    assert "Descansa" in build_instructions(profile) or "gracias" in build_instructions(profile)


def test_example_index_finds_similar_comment():
    index = ExampleIndex(sample_pairs())
    hits = index.search("este audio me ayudó a dormir toda la noche", k=2)
    assert len(hits) == 2
    assert all("dormir" in h["comment"] for h in hits)
    assert index.search("", k=2) == []


# -- selección -----------------------------------------------------------------------


def test_select_members_first_then_few_subscribers():
    threads = [
        thread("m1", c("1", "UCmem", "comentario miembro", hours_ago=5)),
        thread("m2", c("2", "UCmem", "otro de miembro", hours_ago=4)),
        thread("m3", c("3", "UCmem", "tercero de miembro", hours_ago=3)),
        thread("s1", c("4", "UCsub1", "suscriptor poco like", likes=1)),
        thread("s2", c("5", "UCsub2", "suscriptor muy popular", likes=50)),
        thread("o1", c("6", "UCrandom", "no suscrito", likes=99)),
        thread("done", c("7", "UCmem2", "ya respondido"), [c("8", OWNER, "gracias")]),
        thread("old", c("9", "UCmem2", "muy viejo", hours_ago=24 * 10)),
        thread("mine", c("10", OWNER, "mi propio comentario")),
        thread("handled", c("11", "UCmem2", "ya lo respondió el agente")),
    ]
    picked = select_candidates(
        threads, OWNER, members={"UCmem", "UCmem2"}, subscribers={"UCsub1", "UCsub2"},
        handled={"handled"}, since=NOW - timedelta(days=3),
        max_members=50, max_subscribers=1, max_per_author=2,
    )
    assert [(p.thread.id, p.tier) for p in picked] == [("m1", MEMBER), ("m2", MEMBER), ("s2", SUBSCRIBER)]

    with_others = select_candidates(
        threads, OWNER, {"UCmem"}, set(), set(), NOW - timedelta(days=3), 0, 2, 2, include_non_subscribers=True
    )
    assert [p.tier for p in with_others] == [OTHER, OTHER]


# -- generación --------------------------------------------------------------------


class FakeOpenAI:
    def __init__(self, fail_flex=False):
        self.calls = []
        self.fail_flex = fail_flex
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_flex and kwargs.get("service_tier") == "flex":
            raise RuntimeError("flex no disponible")
        items = json.loads(kwargs["input"])["comentarios"]
        replies = [
            {"id": it["id"], "skip": "spam" in it["comentario"], "reply": f"Gracias {it['id']} 🙏"} for it in items
        ]
        usage = SimpleNamespace(input_tokens=100, output_tokens=20, input_tokens_details=SimpleNamespace(cached_tokens=80))
        return SimpleNamespace(output_text=json.dumps({"replies": replies}), usage=usage)


def test_responder_batches_and_respects_skip():
    fake = FakeOpenAI(fail_flex=True)
    responder = Responder(fake, "gpt-6-luna")
    profile = build_profile(sample_pairs())
    items = [{"id": f"t{i}", "comentario": "spam aquí" if i == 2 else "hola"} for i in range(5)]
    out = responder.generate("instr", items, profile, batch_size=3)
    assert set(out) == {"t0", "t1", "t3", "t4"}
    # 2 lotes, cada uno falla con flex y se reintenta sin él.
    assert len(fake.calls) == 4
    assert fake.calls[0]["reasoning"] == {"effort": "none"}
    assert "service_tier" not in fake.calls[1]
    assert responder.usage.calls == 2 and responder.usage.cached_tokens == 160


def test_clean_reply_guards():
    profile = {"link_rate": 0, "p90_chars": 40}
    assert clean_reply('"@fan Gracias!"', profile) == "Gracias!"
    assert clean_reply("mira https://x.com", profile) is None
    assert clean_reply("x" * 400, profile) is None
    assert clean_reply("   ", profile) is None


# -- flujo completo con YouTube simulado ------------------------------------------------


class FakeYouTube:
    def __init__(self, threads, members):
        self.threads = threads
        self.members = members
        self.units_used = 0
        self.posted = []

    def my_channel_id(self):
        return OWNER

    def iter_threads(self, channel_id, since=None):
        yield from self.threads

    def complete_replies(self, t):
        return t

    def member_ids(self):
        return self.members

    def public_subscriber_ids(self):
        return set()

    def video_titles(self, ids):
        return {i: "Sonidos para dormir" for i in ids}

    def can_spend(self, units):
        return True

    def reply(self, parent_id, text):
        self.posted.append((parent_id, text))
        return f"r-{parent_id}"


def test_run_trains_then_replies_and_never_repeats(tmp_path):
    history = [
        thread(f"h{i}", c(f"q{i}", f"UCf{i}", "me ayudó a dormir", hours_ago=500), [c(f"a{i}", OWNER, "Descansa 🙏", hours_ago=499)])
        for i in range(3)
    ]
    new = [thread("n1", c("n1c", "UCmem", "gracias por este audio")), thread("n2", c("n2c", "UCmem", "spam spam"))]
    yt = FakeYouTube(history + new, members={"UCmem"})
    cfg = replace(Config(), data_dir=tmp_path, dry_run=False)

    summary = run(cfg, yt, Responder(FakeOpenAI(), "gpt-6-luna"), now=NOW)
    assert summary["posted"] == 1 and summary["skipped_by_model"] == 1
    assert yt.posted == [("n1", "Gracias n1 🙏")]
    assert (tmp_path / "reports" / "2026-09-25.md").exists()

    # Segunda ejecución: no vuelve a responder el mismo hilo.
    again = run(cfg, yt, Responder(FakeOpenAI(), "gpt-6-luna"), now=NOW)
    assert again["posted"] == 0 and len(yt.posted) == 1

    # Reentrenar no aprende de lo que publicó el agente.
    yt.threads[3].replies.append(c("r-n1", OWNER, "Gracias n1 🙏"))
    profile = train(cfg, yt)
    assert profile["examples_total"] == 3


def test_dry_run_posts_nothing(tmp_path):
    history = [thread("h", c("q", "UCf", "hola"), [c("a", OWNER, "Hola!!")])]
    yt = FakeYouTube(history + [thread("n", c("nc", "UCmem", "hola"))], members={"UCmem"})
    summary = run(replace(Config(), data_dir=tmp_path), yt, Responder(FakeOpenAI(), "m"), now=NOW)
    assert summary["dry_run"] and yt.posted == [] and summary["entries"][0]["reply"]


def test_parse_thread_from_api_payload():
    item = {
        "id": "T",
        "snippet": {
            "videoId": "V",
            "totalReplyCount": 7,
            "canReply": True,
            "topLevelComment": {
                "id": "T",
                "snippet": {
                    "authorChannelId": {"value": "UCfan"},
                    "authorDisplayName": "@fan",
                    "textDisplay": "hola",
                    "publishedAt": "2026-09-24T10:00:00Z",
                    "likeCount": 3,
                },
            },
        },
        "replies": {"comments": [{"id": "R", "snippet": {"authorChannelId": {"value": OWNER}, "textOriginal": "hey", "textDisplay": "hey", "publishedAt": "2026-09-24T11:00:00Z"}}]},
    }
    t = parse_thread(item)
    assert t.top.like_count == 3 and t.replies[0].author_channel_id == OWNER and t.replies_truncated
