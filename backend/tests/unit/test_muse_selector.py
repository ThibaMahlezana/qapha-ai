"""Offline validation tests: no API key or network needed."""
import unittest
from app.providers.muse_agent import validate_selection


class MuseSelectionTests(unittest.TestCase):
    ALLOWED = (2.0, 4.0, 6.0)

    def test_approved_selection(self):
        self.assertEqual(validate_selection({"speed_mps": 4, "reason": "Improve information"}, self.ALLOWED),
                         (4.0, "Improve information"))

    def test_rejects_unapproved_speed(self):
        with self.assertRaises(ValueError):
            validate_selection({"speed_mps": 5, "reason": "test"}, self.ALLOWED)

    def test_rejects_wrong_types(self):
        for speed in (True, "4", None, float("nan")):
            with self.subTest(speed=str(speed)), self.assertRaises(ValueError):
                validate_selection({"speed_mps": speed, "reason": "test"}, self.ALLOWED)

    def test_rejects_unexpected_fields(self):
        with self.assertRaises(ValueError):
            validate_selection({"speed_mps": 4, "reason": "test", "command": "run"}, self.ALLOWED)

    def test_rejects_missing_reason(self):
        with self.assertRaises(ValueError):
            validate_selection({"speed_mps": 4}, self.ALLOWED)


if __name__ == "__main__":
    unittest.main()
