import json

from sciprobe import packs, report, runner
from sciprobe.adapters import (
    DiscoveryAdapter,
    FaultInjectingAdapter,
    RDKitReferenceAgent,
    RecordingAdapter,
    ReplayAdapter,
)
from sciprobe.cli import main
from sciprobe.model import Status


def _chem():
    return packs.load(["cheminformatics", "controls"])


def test_all_packs_load_and_ids_unique():
    tasks, probes = packs.load()
    ids = [t.id for t in tasks] + [p.id for p in probes]
    assert len(ids) == len(set(ids))
    assert len(tasks) > 80 and len(probes) >= 10


def test_reference_agent_is_perfect():
    tasks, probes = _chem()
    res = runner.run(tasks, probes, RDKitReferenceAgent(), agents=["mol-toolkit"])
    bad = [v for v in res.verdicts if not v.status.ok]
    assert not bad, bad[:3]
    assert all(r.status == Status.PASS for r in res.invariance)
    (card,) = report.scorecards(res)
    assert card.sciscore == 100.0


def test_fault_injection_is_caught():
    tasks, probes = _chem()
    faulty = FaultInjectingAdapter(RDKitReferenceAgent(), {
        "exact_mass_for_mw": 1.0, "hallucinate_invalid": 1.0, "input_order_noise": 1.0}, seed=1)
    res = runner.run(tasks, probes, faulty, agents=["mol-toolkit"])
    by_id = {v.task_id: v for v in res.verdicts}
    assert by_id["chem.mw.aspirin"].diagnosis == "monoisotopic exact mass reported instead of average MW"
    assert by_id["ctrl.smiles.unclosed_ring"].status == Status.HALLUCINATED
    inv = {r.probe_id: r for r in res.invariance}
    assert inv["chem.inv.tpsa.smiles.naproxen"].status == Status.FAIL
    # A *consistent* bias (exact mass on every variant) is invisible to invariance
    # probes - that is exactly what the oracle layer is for.
    assert inv["chem.inv.mw.smiles.oseltamivir"].status == Status.PASS
    (card,) = report.scorecards(res)
    assert card.sciscore < 70


def test_record_then_replay_is_identical(tmp_path):
    tasks, probes = _chem()
    rec = tmp_path / "rec.jsonl"
    live = runner.run(tasks, probes, RecordingAdapter(RDKitReferenceAgent(), rec), agents=["mol-toolkit"])
    replay = runner.run(tasks, probes, ReplayAdapter([rec]), agents=["mol-toolkit"])
    strip = lambda r: [(v.task_id, v.status, v.got) for v in r.verdicts]  # noqa: E731
    assert strip(live) == strip(replay)


def test_replay_reads_toolbox_logs_and_serves_repeats_in_order(tmp_path):
    log = tmp_path / "ws-inv-conv.jsonl"
    lines = [
        {"kind": "header", "agentName": "pdb-search"},
        {"kind": "message", "message": {"role": "user", "text": "Q  one"}},
        {"kind": "message", "message": {"role": "agent", "text": "A1", "agentName": "pdb-search", "latencyMs": 5}},
        {"kind": "message", "message": {"role": "user", "text": "Q one"}},
        {"kind": "message", "message": {"role": "agent", "text": "A2", "agentName": "pdb-search"}},
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    rp = ReplayAdapter([tmp_path])
    assert rp.ask("PDB-SEARCH", "Q one").text == "A1"
    assert rp.ask("pdb-search", "Q   one").text == "A2"
    assert rp.ask("pdb-search", "Q one").error
    assert rp.ask("other", "Q one").error


def test_plan_and_fail_under_cli(tmp_path, capsys):
    plan = tmp_path / "plan.jsonl"
    assert main(["plan", "--packs", "controls", "--agents", "psi4", "--out", str(plan)]) == 0
    rows = [json.loads(l) for l in plan.read_text().splitlines()]
    assert rows and all(r["agent"] == "psi4" for r in rows)
    rc = main(["run", "--adapter", "faulty", "--packs", "cheminformatics,controls", "--agents", "mol-toolkit",
               "--out", str(tmp_path / "o"), "--fail-under", "90", "--quiet"])
    assert rc == 1
    assert (tmp_path / "o" / "scorecard.html").exists()


def test_agent_map_renames_target():
    tasks, probes = packs.load(["controls"])
    jobs = runner.plan(tasks, probes, agents=["psi4"], agent_map={"psi4": "my-psi4"})
    assert jobs and all(j.target == "my-psi4" and j.agent == "psi4" for j in jobs)


class FakeDiscovery:
    """Scripted Discovery data plane."""

    def __init__(self, statuses=("in_progress", "completed"), inv_exists=False):
        self.calls = []
        self.statuses = list(statuses)
        self.inv_exists = inv_exists
        self._conv_n = 0

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, json.loads(body) if body else None))
        path = url.split("?")[0]
        if "/investigations/" in path:
            if method == "GET":
                return (200, {"name": "x"}) if self.inv_exists else (404, {"error": "NotFound"})
            if method == "PUT":
                self.inv_exists = True
                return 201, {"name": "x"}
            if method == "DELETE":
                return 202, None
        if path.endswith("/conversations") and method == "POST":
            self._conv_n += 1
            return 200, {"name": f"conv-{self._conv_n}"}
        if "/conversations/" in path and method == "DELETE" and "/responses" not in path:
            return 204, None
        if path.endswith("/responses") and method == "POST":
            return 200, {"id": "resp_1", "status": self.statuses.pop(0)}
        if "/responses/resp_1" in path:
            st = self.statuses.pop(0)
            out = [{"type": "message", "content": [{"type": "output_text", "text": 'ANSWER: {"value": 1.8, "unit": "Å"}'}]}]
            return 200, {"id": "resp_1", "status": st, "output": out if st == "completed" else []}
        return 404, {"error": "unexpected"}


def _paths(fake):
    return [(m, u.split("?")[0].replace("https://ws.example", "")) for m, u, _, _ in fake.calls]


def test_discovery_adapter_request_flow():
    fake = FakeDiscovery()
    ad = DiscoveryAdapter("proj", workspace_uri="https://ws.example/", investigation="inv1",
                          token_provider=lambda: "tok", http=fake, poll_interval_s=0)
    rep = ad.ask("pdb-search", "hello")
    assert rep.error is None and "1.8" in rep.text
    assert _paths(fake) == [
        ("GET", "/projects/proj/investigations/inv1"),
        ("PUT", "/projects/proj/investigations/inv1"),
        ("POST", "/conversations"),
        ("POST", "/conversations/conv-1/openai/v1/responses"),
        ("GET", "/conversations/conv-1/openai/v1/responses/resp_1"),
    ]
    assert "api-version=2026-06-01" in fake.calls[0][1]
    assert fake.calls[2][3]["investigationName"] == "projects/proj/investigations/inv1"
    assert fake.calls[3][3] == {"input": "hello", "agent_reference": {"type": "agent_reference", "name": "pdb-search"}}
    assert all(h["Authorization"] == "Bearer tok" for _, _, h, _ in fake.calls)
    fake.statuses = ["completed"]
    ad.ask("pdb-search", "again")
    assert sum(1 for m, _ in _paths(fake) if m == "PUT") == 1


def test_default_investigation_is_unique_per_run():
    a = DiscoveryAdapter("p", workspace_uri="https://ws", token_provider=lambda: "t", http=FakeDiscovery())
    b = DiscoveryAdapter("p", workspace_uri="https://ws", token_provider=lambda: "t", http=FakeDiscovery())
    assert a.investigation.startswith("sciprobe-") and a.investigation != b.investigation


def test_cleanup_deletes_only_what_it_created():
    fake = FakeDiscovery(statuses=["completed", "completed"])
    ad = DiscoveryAdapter("proj", workspace_uri="https://ws.example", investigation="inv1",
                          token_provider=lambda: "t", http=fake, poll_interval_s=0)
    ad.ask("a", "q1")
    ad.ask("a", "q2")
    summary = ad.cleanup("all")
    assert summary == {"mode": "all", "deleted_conversations": 2, "failed": [], "investigation_deleted": True}
    deletes = [p for m, p in _paths(fake) if m == "DELETE"]
    assert deletes == ["/conversations/conv-1", "/conversations/conv-2", "/projects/proj/investigations/inv1"]
    assert ad.cleanup("all")["deleted_conversations"] == 0  # idempotent


def test_cleanup_never_deletes_preexisting_investigation():
    fake = FakeDiscovery(statuses=["completed"], inv_exists=True)
    ad = DiscoveryAdapter("proj", workspace_uri="https://ws.example", investigation="shared",
                          token_provider=lambda: "t", http=fake, poll_interval_s=0)
    ad.ask("a", "q")
    summary = ad.cleanup("all")
    assert summary["deleted_conversations"] == 1 and summary["investigation_deleted"] is False
    assert not any(m in ("PUT",) for m, _ in _paths(fake))
    assert [p for m, p in _paths(fake) if m == "DELETE"] == ["/conversations/conv-1"]


def test_cleanup_modes():
    fake = FakeDiscovery(statuses=["completed"])
    ad = DiscoveryAdapter("proj", workspace_uri="https://ws.example", investigation="inv1",
                          token_provider=lambda: "t", http=fake, poll_interval_s=0)
    ad.ask("a", "q")
    assert ad.cleanup("none")["deleted_conversations"] == 0
    s = ad.cleanup("conversations")
    assert s["deleted_conversations"] == 1 and s["investigation_deleted"] is False


def test_recording_adapter_forwards_cleanup(tmp_path):
    fake = FakeDiscovery(statuses=["completed"])
    ad = DiscoveryAdapter("proj", workspace_uri="https://ws.example", investigation="inv1",
                          token_provider=lambda: "t", http=fake, poll_interval_s=0)
    rec = RecordingAdapter(ad, tmp_path / "r.jsonl")
    rec.ask("a", "q")
    assert rec.cleanup("all")["deleted_conversations"] == 1
    assert RecordingAdapter(RDKitReferenceAgent(), tmp_path / "x.jsonl").cleanup("all") is None


def test_discovery_adapter_paused_and_errors():
    fake = FakeDiscovery(statuses=["requires_action"])
    ad = DiscoveryAdapter("p", workspace_uri="https://ws", token_provider=lambda: "t", http=fake, poll_interval_s=0)
    assert "paused" in ad.ask("a", "q").error

    def boom(method, url, headers, body):
        return 403, {"error": {"code": "Forbidden"}}

    ad2 = DiscoveryAdapter("p", workspace_uri="https://ws", token_provider=lambda: "t", http=boom)
    assert "403" in ad2.ask("a", "q").error
