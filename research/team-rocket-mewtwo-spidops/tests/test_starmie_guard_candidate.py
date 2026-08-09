from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "work" / "starmie_spidops_guard_v1" / "main.py"


def load_candidate():
    spec = importlib.util.spec_from_file_location("starmie_guard_candidate", CANDIDATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StarmieGuardCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.guard = load_candidate()

    def test_detector_uses_public_zones_only(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"active": [], "bench": [], "discard": []},
                    {
                        "active": [],
                        "bench": [],
                        "discard": [],
                        "hand": [{"id": 1030}],
                        "prize": [{"id": 1031}],
                        "deck": [{"id": 1030}],
                    },
                ],
            }
        }
        self.assertFalse(self.guard._observe_public_starmie(observation))
        observation["current"]["players"][1]["active"] = [{"id": 1030}]
        self.assertTrue(self.guard._observe_public_starmie(observation))

    def test_balanced_guard_redirects_to_active_spidops(self) -> None:
        self.guard._STARMIE_PUBLIC = True
        self.guard._GUARD_MODE = "balanced"
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 401, "energies": [1]}],
                        "bench": [
                            {"id": 431, "energies": [5, 5]},
                            {"id": 400, "energies": []},
                            {"id": 414, "energies": []},
                        ],
                        "hand": [{"id": 1}],
                    },
                    {"active": [{"id": 1031}], "bench": [], "discard": []},
                ],
            },
            "select": {
                "context": 0,
                "option": [
                    {"type": 8, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
                    {"type": 8, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
                ],
            },
        }
        self.assertEqual(self.guard._starmie_guard_action(observation, [0]), [1])

    def test_balanced_guard_does_not_interrupt_active_mewtwo_readiness(self) -> None:
        self.guard._STARMIE_PUBLIC = True
        self.guard._GUARD_MODE = "balanced"
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 431, "energies": [5, 5]}],
                        "bench": [
                            {"id": 401, "energies": [1]},
                            {"id": 400, "energies": []},
                            {"id": 414, "energies": []},
                        ],
                        "hand": [{"id": 1}],
                    },
                    {"active": [{"id": 1031}], "bench": [], "discard": []},
                ],
            },
            "select": {
                "context": 0,
                "option": [
                    {"type": 8, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
                    {"type": 8, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
                ],
            },
        }
        self.assertEqual(self.guard._starmie_guard_action(observation, [0]), [0])


if __name__ == "__main__":
    unittest.main()
