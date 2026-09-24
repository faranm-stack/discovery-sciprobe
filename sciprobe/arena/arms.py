"""Arms = the systems under test.

The fair comparison is ``RetrieveThenAnswer``: one shared generator (same model,
same prompt) fed by different retrievers. ``NullRetriever`` gives the
no-retrieval baseline, which measures what the model already "knows" - with a
private (e.g. Wiley) corpus, a strong baseline means the questions leak.

End-to-end systems that produce their own answers (a Bookshelf GraphRAG global
search, the Wiley agent, any Discovery agent) plug in via ``CallableArm`` or
``DiscoveryAgentArm``; they are compared on answers but are NOT a controlled
retrieval-only A/B, and the report says so.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any, Callable, Protocol

from .model import ArmAnswer, Context, Question
from .text import content_tokens, sentences, tokens

HttpFn = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]


def _urllib_http(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def _auth_headers(api_key: str | None, token_provider: Callable[[], str] | None, key_header: str) -> dict[str, str]:
    if api_key:
        return {key_header: api_key}
    if token_provider:
        return {"Authorization": f"Bearer {token_provider()}"}
    raise ValueError("provide api_key or token_provider")


# ------------------------------------------------------------------ protocols

class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[Context]: ...


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[Context]) -> str: ...


class Arm(Protocol):
    name: str
    controlled: bool  # True when only the retriever differs from other controlled arms

    def answer(self, q: Question) -> ArmAnswer: ...


# ------------------------------------------------------------------ retrievers

class NullRetriever:
    name = "none"

    def retrieve(self, query: str, k: int) -> list[Context]:
        return []


class TfidfRetriever:
    """Pure-Python TF-IDF cosine retriever over an in-memory corpus.

    A dependency-free stand-in for "straight retrieval" in offline tests and a
    sanity baseline next to real vector search.
    """

    def __init__(self, docs: dict[str, str], name: str = "tfidf"):
        self.name = name
        self.docs = docs
        self._tf = {d: Counter(content_tokens(t)) for d, t in docs.items()}
        df = Counter(t for tf in self._tf.values() for t in tf)
        n = len(docs)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._norm = {d: math.sqrt(sum((c * self._idf[t]) ** 2 for t, c in tf.items())) or 1.0 for d, tf in self._tf.items()}

    def retrieve(self, query: str, k: int) -> list[Context]:
        q = Counter(content_tokens(query))
        qn = math.sqrt(sum((c * self._idf.get(t, 0)) ** 2 for t, c in q.items())) or 1.0
        scored = []
        for d, tf in self._tf.items():
            dot = sum(c * self._idf.get(t, 0) * tf.get(t, 0) * self._idf.get(t, 0) for t, c in q.items())
            if dot > 0:
                scored.append((dot / (qn * self._norm[d]), d))
        scored.sort(reverse=True)
        return [Context(doc_id=d, text=self.docs[d], score=round(s, 4)) for s, d in scored[:k]]


class AzureAISearchRetriever:
    """Azure AI Search (keyword, vector via integrated vectorizer, hybrid, semantic).

    REST: POST {endpoint}/indexes/{index}/docs/search?api-version=...
    Vector search uses ``vectorQueries: [{kind: "text", ...}]`` which requires a
    vectorizer configured on the index; omit ``vector_fields`` for keyword only.
    """

    def __init__(self, endpoint: str, index: str, api_key: str | None = None,
                 token_provider: Callable[[], str] | None = None, api_version: str = "2024-07-01",
                 id_field: str = "id", text_field: str = "content", vector_fields: list[str] | None = None,
                 semantic_config: str | None = None, keyword: bool = True, name: str = "azure-ai-search",
                 http: HttpFn | None = None):
        self.name = name
        self.url = f"{endpoint.rstrip('/')}/indexes/{urllib.parse.quote(index)}/docs/search?api-version={api_version}"
        self.api_key, self.token_provider = api_key, token_provider
        self.id_field, self.text_field = id_field, text_field
        self.vector_fields, self.semantic_config, self.keyword = vector_fields, semantic_config, keyword
        self.http = http or _urllib_http

    def retrieve(self, query: str, k: int) -> list[Context]:
        body: dict[str, Any] = {"top": k, "select": f"{self.id_field},{self.text_field}"}
        body["search"] = query if self.keyword else "*"
        if self.vector_fields:
            body["vectorQueries"] = [{"kind": "text", "text": query, "fields": ",".join(self.vector_fields), "k": k}]
        if self.semantic_config:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = self.semantic_config
        headers = {"Content-Type": "application/json", **_auth_headers(self.api_key, self.token_provider, "api-key")}
        status, data = self.http("POST", self.url, headers, json.dumps(body).encode())
        if status >= 400:
            raise RuntimeError(f"Azure AI Search HTTP {status}: {str(data)[:300]}")
        return [Context(doc_id=str(r.get(self.id_field)), text=str(r.get(self.text_field) or ""),
                        score=r.get("@search.rerankerScore", r.get("@search.score")))
                for r in (data or {}).get("value", [])]


class HttpRetriever:
    """Generic JSON-over-HTTP retriever (e.g. a Bookshelf search endpoint once its
    contract is fixed): you supply the request builder and response parser."""

    def __init__(self, name: str, url: str, build_body: Callable[[str, int], dict],
                 parse: Callable[[Any], list[Context]], headers: Callable[[], dict[str, str]] | None = None,
                 http: HttpFn | None = None):
        self.name, self.url, self.build_body, self.parse = name, url, build_body, parse
        self.headers = headers or (lambda: {})
        self.http = http or _urllib_http

    def retrieve(self, query: str, k: int) -> list[Context]:
        status, data = self.http("POST", self.url, {"Content-Type": "application/json", **self.headers()},
                                 json.dumps(self.build_body(query, k)).encode())
        if status >= 400:
            raise RuntimeError(f"{self.name} HTTP {status}: {str(data)[:300]}")
        return self.parse(data)


# ------------------------------------------------------------------ generators

ANSWER_SYSTEM_PROMPT = (
    "You are a careful scientific assistant. Answer the question using ONLY the numbered sources provided. "
    "Cite every factual sentence with the source id in square brackets, e.g. [doc-id]. "
    "If no sources are provided, answer from your own knowledge and write NO_SOURCES at the end. "
    "If the sources do not contain the answer, say so explicitly rather than guessing."
)


def format_sources(contexts: list[Context], max_chars: int = 2500) -> str:
    return "\n\n".join(f"[{c.doc_id}]\n{c.text[:max_chars]}" for c in contexts) or "(no sources)"


class AzureOpenAIGenerator:
    """Chat Completions on Azure OpenAI. Temperature 0 for reproducibility."""

    def __init__(self, endpoint: str, deployment: str, api_key: str | None = None,
                 token_provider: Callable[[], str] | None = None, api_version: str = "2024-10-21",
                 system_prompt: str = ANSWER_SYSTEM_PROMPT, max_tokens: int = 1200, http: HttpFn | None = None):
        self.name = f"aoai:{deployment}"
        self.url = (f"{endpoint.rstrip('/')}/openai/deployments/{urllib.parse.quote(deployment)}"
                    f"/chat/completions?api-version={api_version}")
        self.api_key, self.token_provider = api_key, token_provider
        self.system_prompt, self.max_tokens = system_prompt, max_tokens
        self.http = http or _urllib_http

    def chat(self, system: str, user: str) -> str:
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0, "max_tokens": self.max_tokens}
        headers = {"Content-Type": "application/json", **_auth_headers(self.api_key, self.token_provider, "api-key")}
        status, data = self.http("POST", self.url, headers, json.dumps(body).encode())
        if status >= 400:
            raise RuntimeError(f"Azure OpenAI HTTP {status}: {str(data)[:300]}")
        return data["choices"][0]["message"]["content"] or ""

    def generate(self, question: str, contexts: list[Context]) -> str:
        return self.chat(self.system_prompt, f"Sources:\n{format_sources(contexts)}\n\nQuestion: {question}")


class ExtractiveGenerator:
    """Offline generator: stitches the source sentences most relevant to the
    question, each cited. Deterministic; used for tests and dry runs."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[Context]) -> str:
        if not contexts:
            return "I have no sources for this question. NO_SOURCES"
        q = set(content_tokens(question))
        cands = []
        for c in contexts:
            for s in sentences(c.text):
                overlap = len(q & set(content_tokens(s)))
                if overlap:
                    cands.append((overlap, -len(cands), s, c.doc_id))
        cands.sort(reverse=True)
        picked = cands[: self.max_sentences]
        if not picked:
            return "The sources do not contain the answer."
        return " ".join(f"{s.rstrip('.')} [{d}]." for _, _, s, d in picked)


# ------------------------------------------------------------------ arms

class RetrieveThenAnswer:
    controlled = True

    def __init__(self, name: str, retriever: Retriever, generator: Generator, k: int = 8):
        self.name, self.retriever, self.generator, self.k = name, retriever, generator, k

    def answer(self, q: Question) -> ArmAnswer:
        t0 = time.perf_counter()
        try:
            ctx = self.retriever.retrieve(q.text, self.k)
            text = self.generator.generate(q.text, ctx)
            return ArmAnswer(self.name, q.id, text, ctx, (time.perf_counter() - t0) * 1e3,
                             meta={"retriever": self.retriever.name, "generator": self.generator.name, "k": self.k})
        except Exception as e:  # noqa: BLE001
            return ArmAnswer(self.name, q.id, "", [], (time.perf_counter() - t0) * 1e3, error=f"{type(e).__name__}: {e}")


class CallableArm:
    """Wrap any end-to-end system: fn(question) -> str | ArmAnswer | (text, contexts)."""

    controlled = False

    def __init__(self, name: str, fn: Callable[[Question], Any]):
        self.name, self.fn = name, fn

    def answer(self, q: Question) -> ArmAnswer:
        t0 = time.perf_counter()
        try:
            out = self.fn(q)
        except Exception as e:  # noqa: BLE001
            return ArmAnswer(self.name, q.id, "", [], (time.perf_counter() - t0) * 1e3, error=f"{type(e).__name__}: {e}")
        ms = (time.perf_counter() - t0) * 1e3
        if isinstance(out, ArmAnswer):
            out.arm, out.latency_ms = self.name, out.latency_ms or ms
            return out
        if isinstance(out, tuple):
            return ArmAnswer(self.name, q.id, str(out[0]), list(out[1]), ms)
        return ArmAnswer(self.name, q.id, str(out), [], ms)


class DiscoveryAgentArm:
    """A deployed Discovery agent (e.g. one with a Bookshelf attached) as an arm,
    via sciprobe's DiscoveryAdapter (fresh conversation per question; call
    ``adapter.cleanup()`` afterwards)."""

    controlled = False

    def __init__(self, name: str, adapter, agent: str):
        self.name, self.adapter, self.agent = name, adapter, agent

    def answer(self, q: Question) -> ArmAnswer:
        rep = self.adapter.ask(self.agent, q.text)
        return ArmAnswer(self.name, q.id, rep.text, [], rep.latency_ms, rep.error, meta=dict(rep.meta))


class ReplayArm:
    """Pre-computed answers (e.g. from an existing pipeline or BenchmarkQED files)."""

    def __init__(self, name: str, answers: list[ArmAnswer], controlled: bool = False):
        self.name, self.controlled = name, controlled
        self._by_q = {a.question_id: a for a in answers}

    def answer(self, q: Question) -> ArmAnswer:
        a = self._by_q.get(q.id)
        return a if a else ArmAnswer(self.name, q.id, "", [], error="no recorded answer")


__all__ = [
    "ANSWER_SYSTEM_PROMPT", "AzureAISearchRetriever", "AzureOpenAIGenerator", "CallableArm", "DiscoveryAgentArm",
    "ExtractiveGenerator", "HttpRetriever", "NullRetriever", "ReplayArm", "RetrieveThenAnswer", "TfidfRetriever",
    "format_sources", "tokens",
]
