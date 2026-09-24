"""sciprobe: ground-truth and invariance evaluation for scientific AI agents.

Two complementary checks:

* **Oracle tasks** – questions with a verifiable answer (analytic physics,
  literature benchmarks, or values computed live with RDKit), graded with
  unit-aware tolerances and a diagnosis of *why* a wrong answer is wrong.
* **Invariance probes** – scientifically equivalent inputs (re-ordered SMILES,
  rotated geometries, enantiomers, unit re-framings, paraphrases, repeats)
  whose answers must agree. No ground truth needed.
"""
__version__ = "0.1.0"

from .model import AgentReply, Confusion, Kind, OracleTask, Reference, Status, Verdict  # noqa: E402
from .invariance import InvarianceProbe, Relation  # noqa: E402

__all__ = [
    "AgentReply",
    "Confusion",
    "InvarianceProbe",
    "Kind",
    "OracleTask",
    "Reference",
    "Relation",
    "Status",
    "Verdict",
    "__version__",
]
