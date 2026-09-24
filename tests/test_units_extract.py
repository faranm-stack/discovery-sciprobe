import math

import pytest

from sciprobe import units
from sciprobe.extract import extract


class TestUnits:
    def test_energy_round_trip(self):
        assert units.convert(1.0, "hartree", "kcal/mol") == pytest.approx(627.5095, rel=1e-6)
        assert units.convert(627.5094740631, "kcal/mol", "Eh") == pytest.approx(1.0)
        assert units.convert(1.0, "eV", "kJ/mol") == pytest.approx(96.485, rel=1e-4)

    def test_aliases_and_unicode(self):
        assert units.convert(1.0, "Å²", "nm^2") == pytest.approx(0.01)
        assert units.convert(1.0, "g·mol⁻¹", "Da") == pytest.approx(1.0)
        assert units.convert(1.0, "bohr", "Å") == pytest.approx(0.529177, rel=1e-6)

    def test_case_sensitive_prefixes(self):
        assert units.convert(1.0, "MHz", "Hz") == 1e6
        assert units.convert(1.0, "mV", "V") == 1e-3

    def test_temperature_offset(self):
        assert units.convert(25.0, "°C", "K") == pytest.approx(298.15)

    def test_dimension_mismatch(self):
        with pytest.raises(units.UnitError):
            units.convert(1.0, "hartree", "Å")
        with pytest.raises(units.UnitError):
            units.lookup("furlongs")

    @pytest.mark.parametrize(
        "got,expected,needle",
        [
            (-4.184, 1.0, "kcal <-> kJ"),
            (-1.0, 1.0, "sign flip"),
            (627.5095, 1.0, "hartree <-> kcal/mol"),
            (0.001, 1.0, "SI-prefix"),
            (2 * math.pi, 1.0, "2*pi"),
        ],
    )
    def test_diagnose_ratio(self, got, expected, needle):
        assert needle in units.diagnose_ratio(got, expected)

    def test_diagnose_ratio_none(self):
        assert units.diagnose_ratio(1.37, 1.0) is None


class TestExtract:
    def test_answer_line_wins(self):
        ex = extract('blah 12 g/mol\nANSWER: {"value": 180.16, "unit": "g/mol"}')
        assert (ex.value, ex.unit, ex.method) == (180.16, "g/mol", "answer_line")

    def test_last_answer_line_used(self):
        ex = extract('ANSWER: {"value": 1}\nactually\nANSWER: {"value": 2, "unit": ""}')
        assert ex.value == 2

    def test_string_numbers_and_null(self):
        assert extract('ANSWER: {"value": "1,430", "unit": ""}').value == 1430
        ex = extract('ANSWER: {"value": null, "reason": "invalid SMILES"}')
        assert ex.refused and ex.reason == "invalid SMILES"

    def test_single_quotes_tolerated(self):
        assert extract("ANSWER: {'value': 3, 'unit': ''}").value == 3

    def test_fenced_json(self):
        ex = extract('Here:\n```json\n{"value": -76.0266, "unit": "hartree"}\n```')
        assert ex.value == pytest.approx(-76.0266) and ex.method == "json_block"

    def test_number_with_unit_fallback(self):
        ex = extract("The SCF converged in 12 iterations. Final energy: -1.11676 Eh.", "energy")
        assert ex.value == pytest.approx(-1.11676) and ex.unit.lower() == "eh"

    def test_scientific_notation(self):
        ex = extract("f0 is about 5.03 × 10^3 Hz", "frequency")
        assert ex.value == pytest.approx(5030)

    def test_does_not_read_formula_digits(self):
        ex = extract("Formula C9H8O4, mass 180.16 g/mol", "molar_mass")
        assert ex.value == pytest.approx(180.16)

    def test_refusal_phrase(self):
        ex = extract("Error: RDKit reports this is not a valid SMILES string.")
        assert ex.refused and ex.value is None

    def test_empty(self):
        assert not extract("").found
