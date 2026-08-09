import unittest

from rocketbot.deck import CardCategory, DeckConfig, load_default_deck


EXPECTED_COUNTS = {
    "Team Rocket's Mewtwo ex": 2,
    "Team Rocket's Tarountula": 4,
    "Team Rocket's Spidops": 4,
    "Team Rocket's Articuno": 2,
    "Team Rocket's Mimikyu": 2,
    "Lillie's Clefairy ex": 1,
    "Lillie's Determination": 4,
    "Team Rocket's Ariana": 4,
    "Team Rocket's Giovanni": 3,
    "Team Rocket's Proton": 2,
    "Team Rocket's Petrel": 1,
    "Team Rocket's Transceiver": 4,
    "Ultra Ball": 4,
    "Night Stretcher": 3,
    "Bug Catching Set": 2,
    "Lucky Helmet": 2,
    "Prism Tower": 2,
    "Team Rocket's Factory": 1,
    "Energy Switch": 1,
    "Maximum Belt": 1,
    "Team Rocket's Energy": 4,
    "Basic Grass Energy": 6,
    "Basic Psychic Energy": 1,
}


class DeckIntegrityTests(unittest.TestCase):
    def test_exact_starting_deck(self) -> None:
        deck = load_default_deck()
        self.assertEqual(deck.total_cards, 60)
        self.assertEqual(deck.card_counts, EXPECTED_COUNTS)
        self.assertEqual(
            deck.category_counts,
            {
                CardCategory.POKEMON: 15,
                CardCategory.TRAINER: 34,
                CardCategory.ENERGY: 11,
            },
        )

    def test_canonical_roundtrip_and_fingerprint(self) -> None:
        deck = load_default_deck()
        restored = DeckConfig.from_json(deck.to_json())
        self.assertEqual(restored, deck)
        self.assertEqual(restored.fingerprint, deck.fingerprint)
        self.assertEqual(len(deck.fingerprint), 64)


if __name__ == "__main__":
    unittest.main()
