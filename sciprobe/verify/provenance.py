"""Did the agent actually run its tool, or answer from the language model alone?

Structural evidence (Discovery response ``output`` item types) is used when the
adapter captured it; otherwise a textual heuristic reports what the agent
*claims* it did. The label is informational: a correct answer computed by hand
is still correct, but an agent built around a scientific tool that never calls
it is a reliability risk (it will not scale past what the model can do mentally).
"""
from __future__ import annotations

import re

from ..model import AgentReply

_TOOL_TYPES_EXCLUDED = {"message", "reasoning", "output_text", "text"}
_CLAIM = re.compile(
    r"\b(?:ran|run(?:ning)?|executed|execut(?:e|ing)|script|psi4_utils|computed with|calculated with|using (?:rdkit|psi4|"
    r"xyce|openmm|gromacs|openfoam|autodock|vina|the (?:tool|predictor|model))|simulation (?:output|results?)|"
    r"tool (?:call|output|result)|saved results?|results? (?:file|saved))\b",
    re.IGNORECASE,
)


def provenance(reply: AgentReply) -> str:
    """'tool-called' | 'no-tool-call' | 'claims-tool-use' | 'no-evidence' | 'n/a'."""
    if reply.error or not reply.text:
        return "n/a"
    types = reply.meta.get("output_types")
    if types is not None:
        return "tool-called" if any(t not in _TOOL_TYPES_EXCLUDED for t in types if t) else "no-tool-call"
    return "claims-tool-use" if _CLAIM.search(reply.text) else "no-evidence"
