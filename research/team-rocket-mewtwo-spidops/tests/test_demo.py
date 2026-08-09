import unittest

from rocketbot.demo import run_demo


class DemoTests(unittest.TestCase):
    def test_end_to_end_demo(self) -> None:
        result = run_demo()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deck"]["total_cards"], 60)
        self.assertEqual(result["shapes"]["features"], [1, 37])
        self.assertTrue(result["serialization"]["trace_roundtrip"])
        self.assertTrue(result["policy"]["illegal_candidate_masked"])


if __name__ == "__main__":
    unittest.main()
