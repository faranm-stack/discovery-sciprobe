"""Offline demo: a tiny FICTIONAL kinase corpus (so no model can know it from
pre-training - it behaves like a private-corpus control), questions of several
classes including a two-document "connection" question, and three controlled
arms that share one generator and differ only in retrieval.
"""
from __future__ import annotations

from .arms import ExtractiveGenerator, NullRetriever, RetrieveThenAnswer, TfidfRetriever
from .model import Question

CORPUS: dict[str, str] = {
    "10.9999/kin.001": "ZRK7 is a serine/threonine kinase expressed in cardiac fibroblasts. "
                       "Inhibitor QX-114 blocks ZRK7 with an IC50 of 12 nM in biochemical assays.",
    "10.9999/kin.002": "Phosphorylation of the adaptor protein LUMA-3 at Ser212 is carried out by ZRK7. "
                       "Loss of LUMA-3 phosphorylation abolishes downstream fibrotic signalling.",
    "10.9999/kin.003": "In murine models, LUMA-3 phosphorylation drives resistance to the anti-fibrotic drug pelvanib. "
                       "Resistant animals show a threefold increase in collagen deposition.",
    "10.9999/kin.004": "Kinobead profiling of QX-114 across 280 kinases identified only two off-targets, TAK9 and MRK2, "
                       "each inhibited above 70 percent at 1 micromolar.",
    "10.9999/kin.005": "Macrocyclisation of the QX series improved selectivity for ZRK7 but reduced oral bioavailability "
                       "from 45 percent to 18 percent in rats.",
    "10.9999/kin.006": "A deuterated analogue, QX-114-d3, extended plasma half-life from 2.1 to 5.4 hours in dogs "
                       "without changing ZRK7 potency.",
}

QUESTIONS: list[Question] = [
    Question("q1", "What is the IC50 of QX-114 against ZRK7?", "control",
             ["The IC50 of QX-114 against ZRK7 is 12 nM."], required_docs=["10.9999/kin.001"]),
    Question("q2", "Which off-targets did kinobead profiling identify for QX-114?", "data_local",
             ["Kinobead profiling identified TAK9 as an off-target of QX-114.",
              "Kinobead profiling identified MRK2 as an off-target of QX-114."], required_docs=["10.9999/kin.004"]),
    Question("q3", "How did deuteration change the half-life of QX-114?", "data_local",
             ["The deuterated analogue QX-114-d3 extended plasma half-life from 2.1 to 5.4 hours."],
             required_docs=["10.9999/kin.006"]),
    Question("q4", "What trade-off did macrocyclisation of the QX series introduce?", "activity_local",
             ["Macrocyclisation improved ZRK7 selectivity.", "Macrocyclisation reduced oral bioavailability from 45 to 18 percent."],
             required_docs=["10.9999/kin.005"]),
    Question("q5", "Could inhibiting ZRK7 help overcome pelvanib resistance, and through which mechanism?", "connection",
             ["ZRK7 phosphorylates LUMA-3 at Ser212.", "LUMA-3 phosphorylation drives resistance to pelvanib."],
             required_docs=["10.9999/kin.002", "10.9999/kin.003"]),
    Question("q6", "Across the corpus, what are the main strengths and liabilities of the QX-114 programme?", "data_global",
             ["QX-114 inhibits ZRK7 potently with an IC50 of 12 nM.", "QX-114 has off-targets TAK9 and MRK2.",
              "Macrocyclisation reduced oral bioavailability.", "Deuteration extended half-life."]),
]


def arms():
    gen = ExtractiveGenerator(max_sentences=4)
    return [
        RetrieveThenAnswer("vector_k4", TfidfRetriever(CORPUS, "tfidf"), gen, k=4),
        RetrieveThenAnswer("vector_k1", TfidfRetriever(CORPUS, "tfidf"), gen, k=1),
        RetrieveThenAnswer("no_retrieval", NullRetriever(), gen, k=0),
    ]
