"""Shared HTTP for public scientific APIs: OS trust store (corporate TLS
inspection), polite User-Agent, per-host throttling, retries, in-memory cache,
and a swappable transport for tests."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

try:  # use the Windows/macOS certificate store when available (never disables verification)
    import truststore  # type: ignore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

USER_AGENT = "sciprobe/0.1 (+https://github.com/microsoft/discovery; scientific agent evaluation)"
Transport = Callable[[str, dict[str, str], bytes | None, str], tuple[int, Any]]


class NotFound(Exception):
    pass


def _urllib(url: str, headers: dict[str, str], body: bytes | None, method: str) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
            return r.status, (json.loads(raw) if raw and "json" in ctype else raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


_transport: Transport = _urllib
_cache: dict[str, Any] = {}
_last_hit: dict[str, float] = {}
_lock = threading.Lock()
MIN_INTERVAL_S = 0.34  # ~3 req/s/host: within NCBI's unauthenticated limit


def set_transport(fn: Transport | None) -> None:
    """Swap the HTTP transport (tests) and clear the cache. ``None`` restores urllib."""
    global _transport
    _transport = fn or _urllib
    _cache.clear()


def get_json(url: str, params: dict[str, Any] | None = None, retries: int = 2, use_cache: bool = True) -> Any:
    if params:
        url = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params, doseq=True)}"
    if use_cache and url in _cache:
        return _cache[url]
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(retries + 1):
        with _lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_hit.get(host, 0.0))
            _last_hit[host] = time.monotonic() + max(0.0, wait)
        if wait > 0:
            time.sleep(wait)
        status, data = _transport(url, {"User-Agent": USER_AGENT, "Accept": "application/json"}, None, "GET")
        if status == 404 or (status == 400 and "pubchem" in host):
            raise NotFound(url)
        if status in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(1.5 * (attempt + 1))
            continue
        if status >= 400:
            raise RuntimeError(f"HTTP {status} for {url}: {str(data)[:200]}")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        if use_cache:
            _cache[url] = data
        return data
    raise RuntimeError(f"gave up on {url}")


def get_text(url: str) -> str:
    status, data = _transport(url, {"User-Agent": USER_AGENT}, None, "GET")
    if status == 404:
        raise NotFound(url)
    if status >= 400:
        raise RuntimeError(f"HTTP {status} for {url}")
    return data if isinstance(data, str) else json.dumps(data)
