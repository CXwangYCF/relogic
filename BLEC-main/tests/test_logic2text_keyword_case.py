import unittest

from BLEC.Logic2text.eval import digit_match, logic_matching


class KeywordCaseTests(unittest.TestCase):
    def test_operator_aliases_ignore_case(self):
        for logic, sentence in [
            ("most_eq { all_rows ; surface ; clay } = true", "Most of the surfaces are clay."),
            ("greater { a ; b } = true", "A is MORE than B."),
            ("nth_argmin { all_rows ; age ; 2 }", "The second row with the LOWEST age."),
        ]:
            with self.subTest(logic=logic):
                self.assertEqual(logic_matching(logic, sentence), [])
                self.assertEqual(logic_matching(logic, sentence.lower()), [])

    def test_missing_operator_still_fails(self):
        self.assertIn("most_eq", logic_matching(
            "most_eq { all_rows ; surface ; clay } = true", "The surface is clay."))

    def test_negation_keyword_ignores_case(self):
        logic = "eq { name ; Alice } = true"
        self.assertIn("not", logic_matching(logic, "Alice is NOT the name."))
        self.assertIn("not", logic_matching(logic, "Not equal to Alice."))

    def test_negation_does_not_match_part_of_entity(self):
        for name in ["Nottingham", "nottingham", "Knot"]:
            self.assertEqual(logic_matching("eq { name ; " + name + " } = true", name), [])

    def test_entity_and_numeric_text_are_not_lowercased(self):
        # Numeric matching keeps its existing behavior, including case handling
        # for spelled-out numbers; only operator aliases were changed.
        self.assertEqual(digit_match("2", "Alice has two entries."), 1)
        self.assertEqual(digit_match("2", "Alice has TWO entries."), 0)
        self.assertEqual(digit_match("12", "ALICE has 12 entries."), 1)
        self.assertEqual(logic_matching("eq { name ; Alice } = true", "ALICE"), [])


if __name__ == "__main__":
    unittest.main()
