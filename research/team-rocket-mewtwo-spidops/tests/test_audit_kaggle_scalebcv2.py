from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_kaggle_scalebcv2 as audit  # noqa: E402


FROZEN_DECK = [
    1, 1, 1, 1, 1, 1, 1,
    5, 5, 5,
    15, 15, 15, 15,
    400, 400, 400, 400,
    401, 401, 401, 401,
    431, 431,
    414, 414,
    463, 463,
    432,
    1216, 1216, 1216, 1216,
    1134, 1134, 1134, 1134,
    1152, 1152, 1152, 1152,
    1094, 1094, 1094,
    1218, 1218, 1218,
    1220, 1220,
    1119, 1119,
    1257, 1257,
    1227, 1227,
    1219,
    1217,
    1097,
    1159,
    1175,
]


def pokemon(card_id: int, energy_ids: list[int]) -> dict[str, object]:
    return {
        "id": card_id,
        "energyCards": [{"id": energy_id} for energy_id in energy_ids],
    }


class AuditHelpersTest(unittest.TestCase):
    def test_frozen_deck_integrity_and_counts(self) -> None:
        self.assertEqual(len(FROZEN_DECK), 60)
        self.assertEqual(
            sha256(audit.canonical_deck_bytes(FROZEN_DECK)).hexdigest(),
            audit.FOCAL_DECK_SHA256,
        )
        counts = Counter(FROZEN_DECK)
        self.assertEqual(counts[audit.WOBBUFFET], 1)
        self.assertEqual(counts[audit.MURKROW], 2)
        self.assertEqual(counts[audit.SPIDOPS], 4)
        self.assertEqual(counts[audit.HONCHKROW], 0)

    def test_detect_focal_seat_and_validation_mirror(self) -> None:
        other = list(range(10_000, 10_060))
        self.assertEqual(audit.detect_focal_seat([other, FROZEN_DECK]), (1, False))
        self.assertEqual(audit.detect_focal_seat([FROZEN_DECK, FROZEN_DECK]), (0, True))
        with self.assertRaisesRegex(ValueError, "neither replay deck"):
            audit.detect_focal_seat([other, other])

    def test_ready_attackers_distinguish_spidops_and_wobbuffet(self) -> None:
        player = {
            "active": [pokemon(audit.WOBBUFFET, [15, 1])],
            "bench": [
                pokemon(audit.SPIDOPS, [1, 1]),
                pokemon(audit.SPIDOPS, [1]),
                pokemon(audit.MEWTWO, [5, 15, 1]),
                pokemon(audit.MURKROW, []),
            ],
        }
        self.assertEqual(
            audit.ready_attacker_ids(player),
            [audit.WOBBUFFET, audit.SPIDOPS, audit.MEWTWO],
        )

    def test_team_rocket_energy_supply_respects_colored_costs(self) -> None:
        player = {
            "active": [pokemon(audit.WOBBUFFET, [15])],
            "bench": [
                pokemon(audit.SPIDOPS, [15]),
                pokemon(audit.MEWTWO, [15, 1]),
                pokemon(audit.TAROUNTULA, [15]),
                pokemon(audit.MURKROW, []),
            ],
        }
        self.assertEqual(audit.energy_units(player["active"][0]), 2)
        self.assertEqual(audit.typed_energy_units(player["active"][0], {5}), 2)
        self.assertEqual(
            audit.ready_attacker_ids(player),
            [audit.WOBBUFFET, audit.MEWTWO],
        )

    def test_archaludon_matchup_classifier(self) -> None:
        cases = [
            ([700], {700: "Archaludon"}, "archaludon"),
            ([700, 701], {700: "Archaludon", 701: "Cinderace"}, "archaludon_cinderace"),
            ([700], {700: "Archaludon ex"}, "archaludon"),
        ]
        for deck, names, expected in cases:
            with self.subTest(deck=deck, expected=expected):
                self.assertEqual(audit.opponent_archetype(deck, names), expected)


if __name__ == "__main__":
    unittest.main()
