"""Retrieval arena: GraphRAG vs vector RAG vs live search (and a no-retrieval
baseline), evaluated the way MSR's BenchmarkQED does it - plus verifiable
assertions, citation forensics and question-level paired statistics.

Design rule (the whole point of an A/B): hold the corpus, the question set and
the answer generator constant; vary ONLY the retrieval method.
"""
from .model import ArmAnswer, Context, Question  # noqa: F401
