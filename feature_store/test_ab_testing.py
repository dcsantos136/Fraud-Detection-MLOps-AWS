import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_testing import choose_model_for_customer


class AbTestingTests(unittest.TestCase):
    def test_route_is_deterministic(self):
        self.assertEqual(choose_model_for_customer(101), choose_model_for_customer(101))

    def test_route_uses_supported_variants(self):
        self.assertIn(choose_model_for_customer(1), {"model_a", "model_b"})

    def test_force_model_overrides_route(self):
        self.assertEqual(choose_model_for_customer(999, force_model="model_b"), "model_b")


if __name__ == "__main__":
    unittest.main()
