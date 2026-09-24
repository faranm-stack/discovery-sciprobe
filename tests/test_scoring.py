import pytest

from sciprobe.model import AgentReply, Confusion, Kind, OracleTask, Reference, Status
from sciprobe.scoring import parse_formula, score_task


def task(ref, **kw):
    return OracleTask(id="t", domain="d", agents=["a"], question="q", reference=ref, **kw)


def reply(value, unit="", extra=""):
    import json

    return AgentReply(text=f"{extra}\nANSWER: {json.dumps({'value': value, 'unit': unit})}")


E_REF = Reference(Kind.NUMERIC, -76.02663, "hartree", abs_tol=5e-4)


def test_numeric_pass_same_unit():
    v = score_task(task(E_REF), reply(-76.0266, "Eh"), "psi4")
    assert v.status == Status.PASS and v.score == 1.0


def test_numeric_pass_after_unit_conversion():
    v = score_task(task(E_REF), reply(-76.02663 * 2625.4996394799, "kJ/mol"), "psi4")
    assert v.status == Status.PASS


def test_unit_mislabel_is_diagnosed():
    # agent printed a kcal/mol number but labelled it hartree
    v = score_task(task(E_REF), reply(-76.02663 * 627.5094740631, "hartree"), "psi4")
    assert v.status == Status.FAIL and "kcal/mol" in v.diagnosis


def test_right_number_wrong_label():
    v = score_task(task(E_REF), reply(-76.02663, "eV"), "psi4")
    assert v.status == Status.FAIL and "wrong unit label" in v.diagnosis


def test_wrong_dimension():
    v = score_task(task(E_REF), reply(-76.0, "Å"), "psi4")
    assert v.status == Status.FAIL and "wrong dimension" in v.diagnosis


def test_missing_unit_assumed():
    v = score_task(task(E_REF), reply(-76.0266, ""), "psi4")
    assert v.status == Status.PASS and "assumed" in v.diagnosis


def test_confusion_label():
    t = task(Reference(Kind.NUMERIC, 180.159, "g/mol", abs_tol=0.05),
             confusions=[Confusion(180.042, "exact mass instead of MW", "g/mol")])
    v = score_task(t, reply(180.0423, "g/mol"), "mol")
    assert v.status == Status.FAIL and v.diagnosis == "exact mass instead of MW"


def test_partial_credit_decays():
    ref = Reference(Kind.NUMERIC, 10.0, "V", abs_tol=0.1)
    near = score_task(task(ref), reply(10.2, "V"), "x").score
    far = score_task(task(ref), reply(12.0, "V"), "x").score
    assert 0 < near < 1 and far == 0


def test_range():
    ref = Reference(Kind.RANGE, unit="", lo=1100, hi=1800)
    assert score_task(task(ref), reply(1430), "g").status == Status.PASS
    assert score_task(task(ref), reply(50), "g").status == Status.FAIL


def test_exact_int_and_string():
    assert score_task(task(Reference(Kind.EXACT, 76)), reply("76"), "p").status == Status.PASS
    assert score_task(task(Reference(Kind.EXACT, 76)), reply(76.0), "p").status == Status.PASS
    assert score_task(task(Reference(Kind.EXACT, "CHEMBL25")), reply("chembl25"), "c").status == Status.PASS
    assert score_task(task(Reference(Kind.EXACT, 2244)), reply(338), "c").status == Status.FAIL


def test_formula():
    assert parse_formula("C9H8O4") == parse_formula("H8C9O4")
    assert score_task(task(Reference(Kind.FORMULA, "C9H8O4")), reply("C₉H₈O₄"), "m").status == Status.PASS
    assert score_task(task(Reference(Kind.FORMULA, "C9H8O4")), reply("C9H8O3"), "m").status == Status.FAIL


def test_negative_controls():
    t = task(None, expect_refusal=True)
    ok = score_task(t, AgentReply(text='ANSWER: {"value": null, "reason": "invalid SMILES"}'), "m")
    bad = score_task(t, reply(123.4, "g/mol"), "m")
    assert ok.status == Status.CORRECT_REFUSAL and ok.status.ok
    assert bad.status == Status.HALLUCINATED and not bad.status.ok


def test_refusal_when_answer_expected():
    v = score_task(task(E_REF), AgentReply(text='ANSWER: {"value": null, "reason": "no tool"}'), "x")
    assert v.status == Status.REFUSED


def test_transport_error():
    v = score_task(task(E_REF), AgentReply(text="", error="HTTP 500"), "x")
    assert v.status == Status.ERROR and v.diagnosis == "HTTP 500"


@pytest.mark.parametrize("text", ["I think it's around -76", "no numbers here at all"])
def test_no_answer_or_heuristic(text):
    v = score_task(task(E_REF), AgentReply(text=text), "x")
    assert v.status in (Status.NO_ANSWER, Status.FAIL)
