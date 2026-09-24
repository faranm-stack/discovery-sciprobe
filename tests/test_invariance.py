import itertools
import math
import random

import pytest
from rdkit import Chem

from sciprobe.invariance import (
    InvarianceProbe,
    Relation,
    Variant,
    evaluate,
    mirror_smiles,
    parse_xyz,
    permute_atoms,
    randomized_smiles,
    rigid_motion,
    variants,
)
from sciprobe.model import AgentReply, Status
from sciprobe.packs.quantum import WATER_XYZ

OSELTAMIVIR = "CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(C)=O)N)C(=O)OCC"


def _dists(block):
    atoms = parse_xyz(block)
    return sorted(
        round(math.dist(a[1:], b[1:]), 5) for a, b in itertools.combinations(atoms, 2)
    )


def test_randomized_smiles_are_distinct_but_equivalent():
    out = randomized_smiles(OSELTAMIVIR, 5, seed=1)
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(OSELTAMIVIR))
    assert len(out) == 5 and len(set(out)) == 5 and OSELTAMIVIR not in out
    assert all(Chem.MolToSmiles(Chem.MolFromSmiles(s)) == canon for s in out)


def test_randomized_smiles_deterministic():
    assert randomized_smiles(OSELTAMIVIR, 3, seed=7) == randomized_smiles(OSELTAMIVIR, 3, seed=7)


def test_mirror_is_enantiomer():
    s = "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1"
    m = mirror_smiles(s)
    assert m == "C[C@@H](C(=O)O)c1ccc2cc(OC)ccc2c1"
    a, b = Chem.MolFromSmiles(s), Chem.MolFromSmiles(m)
    assert Chem.MolToSmiles(a) != Chem.MolToSmiles(b)
    assert Chem.MolToSmiles(a, isomericSmiles=False) == Chem.MolToSmiles(b, isomericSmiles=False)
    assert mirror_smiles(m) == s


def test_rigid_motion_preserves_geometry():
    rng = random.Random(3)
    moved = rigid_motion(WATER_XYZ, rng)
    assert moved != WATER_XYZ
    assert _dists(moved) == pytest.approx(_dists(WATER_XYZ), abs=1e-5)


def test_permutation_preserves_atoms():
    rng = random.Random(0)
    perm = permute_atoms(WATER_XYZ, rng)
    assert sorted(parse_xyz(perm)) == sorted(parse_xyz(WATER_XYZ))
    assert perm.splitlines() != [l for l in WATER_XYZ.splitlines()]


def test_variants_include_answer_instruction():
    p = InvarianceProbe("p", "d", ["a"], Relation.SMILES_RANDOMIZATION, "MW of {input}?", OSELTAMIVIR, "g/mol", n_variants=4)
    vs = variants(p)
    assert len(vs) == 4 and vs[0].label == "original"
    assert all("ANSWER:" in v.prompt for v in vs)


def test_enantiomer_requires_stereo():
    p = InvarianceProbe("p", "d", ["a"], Relation.ENANTIOMER, "{input}", "CCO", None)
    with pytest.raises(ValueError):
        variants(p)


def _rep(v, unit=""):
    return AgentReply(text=f'ANSWER: {{"value": {v}, "unit": "{unit}"}}')


def _probe(**kw):
    base = dict(id="p", domain="d", agents=["a"], relation=Relation.REPEAT, template="{input}", base_input="x",
                unit="hartree", abs_tol=1e-4)
    base.update(kw)
    return InvarianceProbe(**base)


def test_evaluate_pass_and_fail():
    p = _probe()
    v = [Variant(f"r{i}", "", "x") for i in range(3)]
    ok = evaluate(p, "a", list(zip(v, [_rep(-1.0), _rep(-1.00005), _rep(-1.00001)])))
    bad = evaluate(p, "a", list(zip(v, [_rep(-1.0), _rep(-1.01), _rep(-1.0)])))
    assert ok.status == Status.PASS
    assert bad.status == Status.FAIL and "min at r1" in bad.diagnosis


def test_evaluate_unit_reframe_converts():
    p = _probe(relation=Relation.UNIT_REFRAME, extras={"units": ["hartree", "kcal/mol"]})
    v1, v2 = Variant("h", "", "x", {"unit": "hartree"}), Variant("k", "", "x", {"unit": "kcal/mol"})
    # second reply omits the unit: evaluator must fall back to the unit that was asked for
    res = evaluate(p, "a", [(v1, _rep(-1.11676, "hartree")), (v2, AgentReply(text='ANSWER: {"value": -700.779}'))])
    assert res.status == Status.PASS, res.diagnosis


def test_evaluate_inconsistent_refusal():
    p = _probe()
    v = [Variant("a", "", "x"), Variant("b", "", "y")]
    res = evaluate(p, "a", [(v[0], _rep(-1.0)), (v[1], AgentReply(text='ANSWER: {"value": null, "reason": "bad input"}'))])
    assert res.status == Status.FAIL and "refused" in res.diagnosis


def test_evaluate_needs_two_answers():
    p = _probe()
    v = [Variant("a", "", "x"), Variant("b", "", "y")]
    res = evaluate(p, "a", [(v[0], _rep(-1.0)), (v[1], AgentReply(text="", error="boom"))])
    assert res.status == Status.NO_ANSWER
