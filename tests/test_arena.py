import json

import pytest

from sciprobe.arena import demo, stats
from sciprobe.arena.arms import AzureAISearchRetriever, CallableArm, TfidfRetriever
from sciprobe.arena.citations import analyze, cited_ids
from sciprobe.arena.judge import LLMAssertionChecker, LLMPairwiseJudge, LexicalAssertionChecker, counterbalanced
from sciprobe.arena.model import ArmAnswer, Context, Question
from sciprobe.arena.questions import export_autoe, load_answers, load_questions
from sciprobe.arena.report import to_markdown
from sciprobe.arena.run import run_arena


def test_stats_match_hand_calculations():
    assert stats.sign_test(6, 4) == pytest.approx(0.7539, abs=1e-4)  # Ash's 6-4 per-topic split
    assert stats.binom_two_sided(66, 110) == pytest.approx(0.045, abs=0.002)
    assert stats.min_wins_for_significance(10) == 9 and stats.min_wins_for_significance(100) == 61
    assert stats.questions_needed(0.6) in range(190, 200)
    lo, hi = stats.wilson_ci(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    lo, hi = stats.bootstrap_mean_ci([1.0] * 10)
    assert lo == hi == 1.0


def test_citations_metrics():
    a = ArmAnswer("x", "q", "ZRK7 is inhibited by QX-114 [d1]. It binds [2]. Made-up fact [d9]. Uncited claim here.",
                  [Context("d1", "Inhibitor QX-114 blocks ZRK7 potently"), Context("d2", "binds the hinge"), Context("d3", "other")])
    assert cited_ids(a.text, ["d1", "d2", "d3"]) == ["d1", "d2", "d9"]
    r = analyze(a, corpus_ids={"d1", "d2", "d3"})
    assert r.n_cited == 3 and r.n_retrieved == 3
    assert r.precision == pytest.approx(2 / 3) and r.utilization == pytest.approx(2 / 3)
    assert r.out_of_corpus == ["d9"] and r.unretrieved == ["d9"]
    assert r.uncited_sentence_fraction is not None and r.uncited_sentence_fraction > 0
    assert cited_ids("see https://doi.org/10.1000/ABC.1.") == ["10.1000/abc.1"]


def test_benchmarkqed_io_roundtrip(tmp_path):
    qf = tmp_path / "data_local_assertions.json"
    qf.write_text(json.dumps([{"question_id": "a1", "question_text": "How many chairs?",
                               "assertions": "[{'statement': 'The response should report 15 chairs.', 'score': 10}]"}]))
    af = tmp_path / "vector" / "data_local_answers.json"
    af.parent.mkdir()
    af.write_text(json.dumps([{"question_id": "a1", "question": "How many chairs?", "answer": "15 chairs."}]))
    qs = load_questions([qf])
    assert qs[0].qclass == "data_local" and qs[0].assertions == ["The response should report 15 chairs."]
    ans = load_answers(af.parent, "vector")
    assert ans[0].text == "15 chairs."
    out = export_autoe(qs, {"vector": ans, "graph": ans}, tmp_path / "autoe", base="vector")
    exported = json.loads((out / "input" / "graph" / "data_local_answers.json").read_text())
    assert exported == [{"question_id": "a1", "question_text": "How many chairs?", "answer": "15 chairs."}]
    settings = (out / "settings.yaml").read_text()
    assert "name: vector" in settings and "- name: graph" in settings and "- data_local" in settings


def test_lexical_checker():
    c = LexicalAssertionChecker()
    assert c.check("q", "QX-114 has an IC50 of 12 nM against ZRK7.", "The IC50 of QX-114 against ZRK7 is 12 nM.")
    assert not c.check("q", "The IC50 is 12 nM against ZRK7.", "The IC50 of QX-114 against ZRK7 is 12 nM.")  # entity missing
    assert not c.check("q", "QX-114 has an IC50 of 15 nM against ZRK7.", "The IC50 of QX-114 against ZRK7 is 12 nM.")


class PositionBiasedJudge:
    name = "biased"

    def judge(self, question, first, second, criterion):
        return "1"  # always prefers whatever is shown first


def test_counterbalancing_neutralises_position_bias():
    v = counterbalanced(PositionBiasedJudge(), "q", "a", "b", "relevance")
    assert v.winner == "tie" and not v.consistent


def test_llm_judges_parse_replies():
    j = LLMPairwiseJudge(lambda s, u: "Answer 2 is more specific.\nWINNER: 2")
    assert j.judge("q", "x", "y", "relevance") == "2"
    assert LLMPairwiseJudge(lambda s, u: "no verdict").judge("q", "x", "y", "relevance") == "tie"
    assert LLMAssertionChecker(lambda s, u: "ok\nVERDICT: YES").check("q", "a", "x")
    assert not LLMAssertionChecker(lambda s, u: "VERDICT: NO").check("q", "a", "x")


def test_demo_arena_end_to_end():
    res = run_arena(demo.QUESTIONS, demo.arms(), assertion_checker=LexicalAssertionChecker(),
                    baseline="no_retrieval", corpus_ids=set(demo.CORPUS))
    by = {s["arm"]: s for s in res.arm_summary}
    assert by["no_retrieval"]["assertion_pass"] == 0
    assert by["vector_k4"]["assertion_pass"] > 0.6 and by["vector_k4"]["out_of_corpus_citations"] == 0
    assert by["vector_k4"]["assertion_pass_by_class"]["connection"] < 1  # multi-hop is where vector RAG struggles
    lift = {r["arm"]: r for r in res.lift}
    assert lift["vector_k4"]["better"] == 6 and lift["vector_k4"]["p_sign"] == pytest.approx(0.03125)
    main = [r for r in res.pair_summary if r["qclass"] == "all"]
    assert all("p_holm" in r for r in main)
    assert any("question-level wins" in w for w in res.warnings)
    md = to_markdown(res)
    assert "Lift over the no-retrieval baseline" in md and "Validity notes" in md


def test_mixed_arms_warn_and_fabrication_flagged():
    fab = CallableArm("fabricator", lambda q: ("QX-114 is great [10.9999/fake.999].", [Context("10.9999/kin.001", "x")]))
    res = run_arena(demo.QUESTIONS[:2], demo.arms()[:1] + [fab], assertion_checker=LexicalAssertionChecker(),
                    corpus_ids=set(demo.CORPUS))
    assert any("end-to-end systems" in w for w in res.warnings)
    assert any("fabricator cited" in w for w in res.warnings)


def test_azure_ai_search_request_shape():
    seen = {}

    def http(method, url, headers, body):
        seen.update(method=method, url=url, headers=headers, body=json.loads(body))
        return 200, {"value": [{"id": "d1", "content": "text", "@search.score": 3.2, "@search.rerankerScore": 2.9}]}

    r = AzureAISearchRetriever("https://s.search.windows.net/", "kinase", api_key="k", vector_fields=["vec"],
                               semantic_config="sem", http=http)
    ctx = r.retrieve("CDK12 inhibitors", 5)
    assert seen["url"] == "https://s.search.windows.net/indexes/kinase/docs/search?api-version=2024-07-01"
    assert seen["headers"]["api-key"] == "k"
    assert seen["body"]["vectorQueries"] == [{"kind": "text", "text": "CDK12 inhibitors", "fields": "vec", "k": 5}]
    assert seen["body"]["queryType"] == "semantic" and seen["body"]["top"] == 5
    assert ctx[0].doc_id == "d1" and ctx[0].score == 2.9


def test_tfidf_retrieves_the_right_doc():
    r = TfidfRetriever(demo.CORPUS)
    assert r.retrieve("kinobead off-targets of QX-114", 1)[0].doc_id == "10.9999/kin.004"
