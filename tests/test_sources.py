import json
import unittest
from pathlib import Path

from licenciagraph.sources.curate import GENERIC_INDUSTRIAL
from licenciagraph.sources.manual import (
    classify_modality,
    normalize_code,
    parse_columns,
    parse_manual,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
MANUAL_PDF = ROOT / "data" / "sources" / "manual-licenciamento-5a-edicao.pdf"


class HelperTests(unittest.TestCase):
    def test_normalize_code_accepts_layout_variants(self):
        self.assertEqual(normalize_code("AG-02"), "AG-02")
        self.assertEqual(normalize_code("CS – 19 – Saúde"), "CS-19")
        self.assertEqual(normalize_code("AI 01 - Atividades"), "AI-01")
        self.assertEqual(normalize_code("RS-02.1"), "RS-02.1")

    def test_classify_modality(self):
        cases = {
            "LICENÇA PRÉVIA": "LP",
            "RENOVAÇÃO DE LICENÇA DE OPERAÇÃO": "RLO",
            "LICENÇA DE ALTERAÇÃO DE INSTALAÇÃO": "LAI",
            "LICENÇA DE OPERAÇÃO CORRETIVA": "LOC",
            "LICENÇA DE OPERAÇÃO PARA PESQUISA": "LOP",
            "LICENCIAMENTO POR ADESÃO E COMPROMISSO": "LAC",
            "AUTORIZAÇÃO AMBIENTAL": "AA",
        }
        for heading, expected in cases.items():
            self.assertEqual(classify_modality([heading]), expected, heading)

    def test_parse_columns_joins_split_header_cells(self):
        raw = ["LP", "LI", "RLI", "LAI", "LO", "RL", "O", "LA", "O", "LO", "C", "AA"]
        self.assertEqual(
            parse_columns(raw),
            ["LP", "LI", "RLI", "LAI", "LO", "RLO", "LAO", "LOC", "AA"],
        )
        self.assertEqual(parse_columns(["FAUNA", "LP", "LI"]), ["LP", "LI"])


class CuratedManualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = json.loads((CURATED / "manual.json").read_text("utf-8"))

    def test_every_checklist_has_items_and_modality(self):
        for checklist in self.manual["checklists"]:
            self.assertTrue(checklist["items"], checklist["key"])
            self.assertTrue(checklist["modality"], checklist["key"])

    def test_keys_are_unique_and_resolve_references(self):
        keys = [c["key"] for c in self.manual["checklists"]]
        self.assertEqual(len(keys), len(set(keys)))
        known = set(keys)
        for activity in self.manual["activities"]:
            for key in activity["checklists"]:
                self.assertIn(key, known, activity["code"])

    def test_duplicated_manual_codes_are_disambiguated(self):
        keys = {c["key"] for c in self.manual["checklists"]}
        self.assertIn("OC-58@LO", keys)
        self.assertIn("OC-58@LAI", keys)

    def test_item_numbers_are_sequential(self):
        for checklist in self.manual["checklists"]:
            numbers = [i["number"] for i in checklist["items"]]
            first = numbers[0]
            self.assertEqual(
                numbers, list(range(first, first + len(numbers))), checklist["key"]
            )

    def test_generic_industrial_checklists_exist(self):
        keys = {c["code"] for c in self.manual["checklists"]}
        for codes in GENERIC_INDUSTRIAL["by_modality"].values():
            for code in codes:
                self.assertIn(code, keys)

    def test_report_has_no_unreferenced_missing_pages(self):
        self.assertEqual(self.manual["report"]["referenced_without_page"], [])


class CuratedNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.norm = json.loads((CURATED / "na101.json").read_text("utf-8"))

    def test_cost_matrix_matches_known_cells(self):
        letters = self.norm["cost_letters_ufr"]
        self.assertEqual(letters["A"]["LP"], "4")
        self.assertEqual(letters["F"]["LI"], "19")
        self.assertEqual(letters["A"]["REGISTRO"], "ISENTO")

    def test_porte_tables_are_mostly_complete(self):
        tables = self.norm["porte_tables"]
        complete = [t for t in tables if len(t["rows"]) >= 4]
        self.assertGreaterEqual(len(complete) / len(tables), 0.99)

    def test_known_activity_porte_and_letter(self):
        table = next(t for t in self.norm["porte_tables"] if "08.49.100" in t["codes"])
        micro = table["rows"][0]
        self.assertEqual((micro["porte"], micro["letter"]), ("MICRO", "C"))
        self.assertEqual(table["rows"][3]["annotation"], "(EVA)")

    def test_activity_lists_have_expected_sizes(self):
        regimes = self.norm["report"]["listed_by_regime"]
        self.assertEqual(regimes["dispensa"], 89)
        self.assertEqual(regimes["lac"], 67)
        self.assertEqual(regimes["lau"], 41)


@unittest.skipUnless(MANUAL_PDF.exists(), "PDF do manual não disponível")
class ManualPdfTests(unittest.TestCase):
    def test_parser_reproduces_curated_counts(self):
        manual = json.loads((CURATED / "manual.json").read_text("utf-8"))
        parsed = parse_manual(MANUAL_PDF)
        self.assertEqual(len(parsed.checklists), len(manual["checklists"]))
        self.assertEqual(len(parsed.activities), len(manual["activities"]))


if __name__ == "__main__":
    unittest.main()
