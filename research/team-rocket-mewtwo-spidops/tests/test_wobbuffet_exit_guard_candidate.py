from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "work" / "wobbuffet_spidops_exit_guard_v1" / "main.py"


def load_candidate():
    spec = importlib.util.spec_from_file_location("wobbuffet_exit_guard_candidate", CANDIDATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(*, attack_id: int = 610, spidops_energies: list[int] | None = None):
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [{"id": 432, "energies": [1, 5, 5]}],
                    "bench": [
                        {"id": 401, "energies": spidops_energies or []},
                        {"id": 400, "energies": []},
                        {"id": 414, "energies": []},
                    ],
                },
                {
                    "active": [{"id": 700}],
                    "bench": [],
                    "hand": [{"id": 999}],
                    "prize": [{"id": 998}],
                    "deck": [{"id": 997}],
                },
            ],
        },
        "select": {
            "context": 0,
            "option": [
                {"type": 13, "attackId": attack_id},
                {"type": 12},
                {"type": 14},
            ],
        },
    }


class WobbuffetExitGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.guard = load_candidate()

    def test_headbutt_redirects_to_offered_retreat(self) -> None:
        obs = observation(spidops_energies=[1, 1])
        self.assertEqual(self.guard._wobbuffet_exit_guard_action(obs, [0]), [1])

    def test_rocket_mirror_is_preserved(self) -> None:
        obs = observation(attack_id=609, spidops_energies=[1, 1])
        self.assertEqual(self.guard._wobbuffet_exit_guard_action(obs, [0]), [0])

    def test_unready_spidops_preserves_headbutt(self) -> None:
        obs = observation(spidops_energies=[1])
        self.assertEqual(self.guard._wobbuffet_exit_guard_action(obs, [0]), [0])

    def test_team_rocket_energy_without_grass_does_not_ready_spidops(self) -> None:
        obs = observation(spidops_energies=[5, 5])
        self.assertEqual(self.guard._wobbuffet_exit_guard_action(obs, [0]), [0])

    def test_hidden_opponent_zones_do_not_affect_trigger(self) -> None:
        obs = observation(spidops_energies=[1, 1])
        expected = self.guard._wobbuffet_exit_guard_action(obs, [0])
        obs["current"]["players"][1]["hand"] = [{"id": 12345}]
        obs["current"]["players"][1]["prize"] = []
        obs["current"]["players"][1]["deck"] = []
        self.assertEqual(self.guard._wobbuffet_exit_guard_action(obs, [0]), expected)


if __name__ == "__main__":
    unittest.main()
