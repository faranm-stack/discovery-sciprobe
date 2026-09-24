"""Direct REST adapter for Microsoft Discovery agents.

Mirrors the Discovery Toolbox ``discovery_askAgent`` REST path:
  PUT  {ws}/projects/{p}/investigations/{inv}?api-version=...
  POST {ws}/conversations?api-version=...
  POST {ws}/conversations/{conv}/openai/v1/responses   {input, agent_reference}
  GET  {ws}/conversations/{conv}/openai/v1/responses/{id}   (poll)

Every prompt gets a fresh conversation so equivalent-input variants can't see
each other's answers (which would defeat invariance testing). Each run uses its
own investigation (``sciprobe-<utc>-<rand>`` by default) and ``cleanup()``
deletes only the conversations/investigation this adapter created, so shared
projects are left as they were found.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

from ..model import AgentReply

DISCOVERY_SCOPE = "https://discovery.azure.com/.default"
ARM_SCOPE = "https://management.azure.com/.default"
DEFAULT_API_VERSION = "2026-06-01"
_TERMINAL = {"completed", "succeeded", "failed", "cancelled", "canceled", "incomplete", "error", "timeout", "timedout"}
_PAUSED = {"requires_action", "paused", "awaiting_input", "input_required"}

HttpFn = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]


class DiscoveryError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _urllib_http(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


class TokenProvider:
    """env var -> azure-identity -> az CLI, cached until shortly before expiry."""

    def __init__(self, scope: str, env_var: str):
        self.scope, self.env_var = scope, env_var
        self._token: str | None = None
        self._exp = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            if self._token and time.time() < self._exp - 120:
                return self._token
            env = os.environ.get(self.env_var)
            if env:
                self._token, self._exp = env, time.time() + 3000
                return env
            errors = []
            try:
                from azure.identity import DefaultAzureCredential  # type: ignore

                tok = DefaultAzureCredential().get_token(self.scope)
                self._token, self._exp = tok.token, float(tok.expires_on)
                return tok.token
            except Exception as e:  # noqa: BLE001
                errors.append(f"azure-identity: {e}")
            az = shutil.which("az") or r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
            if az and os.path.exists(az):
                proc = subprocess.run([az, "account", "get-access-token", "--scope", self.scope, "-o", "json"],
                                      capture_output=True, text=True, timeout=60)
                if proc.returncode == 0:
                    data = json.loads(proc.stdout)
                    self._token = data["accessToken"]
                    self._exp = float(data.get("expires_on") or time.time() + 3000)
                    return self._token
                errors.append(f"az: {proc.stderr.strip()[:200]}")
            raise RuntimeError(f"no token for {self.scope}; set {self.env_var} or sign in. ({'; '.join(errors)})")


def extract_response_text(resp: dict) -> str:
    if resp.get("output_text"):
        return str(resp["output_text"])
    parts: list[str] = []
    for item in resp.get("output") or []:
        if isinstance(item.get("text"), str):
            parts.append(item["text"])
        for block in item.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(p for p in parts if p).strip()


class DiscoveryAdapter:
    name = "discovery"

    def __init__(
        self,
        project: str,
        workspace_uri: str | None = None,
        workspace_arm_id: str | None = None,
        investigation: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        poll_timeout_s: float = 900.0,
        poll_interval_s: float = 3.0,
        token_provider: Callable[[], str] | None = None,
        arm_token_provider: Callable[[], str] | None = None,
        http: HttpFn | None = None,
    ):
        if not workspace_uri and not workspace_arm_id:
            raise ValueError("provide workspace_uri or workspace_arm_id")
        self.project = project
        self._ws_uri = workspace_uri.rstrip("/") if workspace_uri else None
        self.workspace_arm_id = workspace_arm_id
        self.investigation = investigation or (
            f"sciprobe-{time.strftime('%Y%m%dt%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:4]}"
        )
        self.api_version = api_version
        self.poll_timeout_s = poll_timeout_s
        self.poll_interval_s = poll_interval_s
        self.token = token_provider or TokenProvider(DISCOVERY_SCOPE, "SCIPROBE_DISCOVERY_TOKEN")
        self.arm_token = arm_token_provider or TokenProvider(ARM_SCOPE, "SCIPROBE_ARM_TOKEN")
        self.http = http or _urllib_http
        self._inv_ready = False
        self._created_investigation = False
        self._created_conversations: list[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------ plumbing
    def _call(self, method: str, url: str, body: Any = None, token: str | None = None) -> Any:
        headers = {"Authorization": f"Bearer {token or self.token()}", "Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        status, payload = self.http(method, url, headers, data)
        if status >= 400:
            raise DiscoveryError(status, json.dumps(payload)[:500] if not isinstance(payload, str) else payload[:500])
        return payload

    def _dp(self, path: str) -> str:
        return f"{self.ws_uri}{path}?{urllib.parse.urlencode({'api-version': self.api_version})}"

    @property
    def ws_uri(self) -> str:
        if self._ws_uri is None:
            url = f"https://management.azure.com{self.workspace_arm_id}?api-version={self.api_version}"
            data = self._call("GET", url, token=self.arm_token()) or {}
            props = data.get("properties") or {}
            uri = props.get("workspaceApiUri") or props.get("discoveryApiUri") or props.get("dataPlaneUri")
            if not uri and data.get("name"):
                uri = f"https://{data['name']}.workspace.discovery.azure.com"
            if not uri:
                raise RuntimeError("could not resolve workspace data-plane URI")
            self._ws_uri = uri.rstrip("/")
        return self._ws_uri

    def _inv_path(self) -> str:
        return f"/projects/{urllib.parse.quote(self.project)}/investigations/{urllib.parse.quote(self.investigation)}"

    def _ensure_investigation(self) -> None:
        with self._lock:
            if self._inv_ready:
                return
            # GET first: PUT is an upsert, so only a 404 proves we are the creator.
            try:
                self._call("GET", self._dp(self._inv_path()))
                existed = True
            except DiscoveryError as e:
                if e.status != 404:
                    raise
                existed = False
            if not existed:
                try:
                    self._call("PUT", self._dp(self._inv_path()), {"displayName": self.investigation})
                    self._created_investigation = True
                except DiscoveryError as e:
                    if e.status != 409:
                        raise
            self._inv_ready = True

    def cleanup(self, mode: str = "all") -> dict[str, Any]:
        """Delete what this adapter created. Never touches pre-existing resources.

        mode: "none" | "conversations" | "all" (conversations + the investigation,
        only if this adapter created it).
        """
        summary: dict[str, Any] = {"mode": mode, "deleted_conversations": 0, "failed": [], "investigation_deleted": False}
        if mode == "none":
            return summary
        with self._lock:
            convs, self._created_conversations = self._created_conversations, []
        for name in convs:
            try:
                self._call("DELETE", self._dp(f"/conversations/{urllib.parse.quote(name)}"))
                summary["deleted_conversations"] += 1
            except DiscoveryError as e:
                if e.status == 404:
                    summary["deleted_conversations"] += 1
                else:
                    summary["failed"].append(f"conversation {name}: {e}")
            except Exception as e:  # noqa: BLE001
                summary["failed"].append(f"conversation {name}: {e}")
        if mode == "all" and self._created_investigation:
            try:
                self._call("DELETE", self._dp(self._inv_path()))
                summary["investigation_deleted"] = True
                self._created_investigation = False
                self._inv_ready = False
            except DiscoveryError as e:
                if e.status == 404:
                    summary["investigation_deleted"] = True
                else:
                    summary["failed"].append(f"investigation {self.investigation}: {e}")
        return summary

    # ------------------------------------------------------------ public
    def ask(self, agent: str, prompt: str) -> AgentReply:
        t0 = time.perf_counter()
        try:
            self._ensure_investigation()
            conv = self._call("POST", self._dp("/conversations"), {
                "projectName": self.project,
                "investigationName": f"projects/{self.project}/investigations/{self.investigation}",
                "displayName": f"sciprobe-{agent}-{uuid.uuid4().hex[:8]}",
            }) or {}
            conv_name = conv.get("name")
            if not conv_name:
                raise RuntimeError(f"conversation create returned no name: {conv}")
            with self._lock:
                self._created_conversations.append(conv_name)
            base = f"{self.ws_uri}/conversations/{urllib.parse.quote(conv_name)}/openai/v1/responses"
            resp = self._call("POST", base, {"input": prompt, "agent_reference": {"type": "agent_reference", "name": agent}}) or {}
            deadline = time.monotonic() + self.poll_timeout_s
            while True:
                status = str(resp.get("status") or "").lower()
                if resp.get("error") or status in _TERMINAL:
                    break
                if status in _PAUSED:
                    return AgentReply(text=extract_response_text(resp), error="response paused awaiting human approval",
                                      latency_ms=(time.perf_counter() - t0) * 1e3, meta={"conversation": conv_name})
                if time.monotonic() > deadline:
                    return AgentReply(text=extract_response_text(resp), error=f"timed out (last status {status!r})",
                                      latency_ms=(time.perf_counter() - t0) * 1e3, meta={"conversation": conv_name})
                time.sleep(self.poll_interval_s)
                resp = self._call("GET", f"{base}/{urllib.parse.quote(str(resp['id']))}") or {}
            err = resp.get("error")
            text = extract_response_text(resp)
            return AgentReply(
                text=text,
                error=(json.dumps(err) if err else None) if not text else None,
                latency_ms=(time.perf_counter() - t0) * 1e3,
                meta={"conversation": conv_name, "response_id": resp.get("id"), "status": resp.get("status"),
                      "output_types": [i.get("type") for i in (resp.get("output") or []) if isinstance(i, dict)]},
            )
        except Exception as e:  # noqa: BLE001
            return AgentReply(text="", error=f"{type(e).__name__}: {e}", latency_ms=(time.perf_counter() - t0) * 1e3)
