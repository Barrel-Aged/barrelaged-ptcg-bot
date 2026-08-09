from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    PROJECT_ROOT
    / "artifacts"
    / "submissions"
    / "HelbrosMewtwoSpidopsGTWTV1.tar.gz"
)
MODEL = (
    PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json"
)


class SubmissionArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = tarfile.open(ARCHIVE, mode="r:gz")
        cls.members = {
            member.name: member
            for member in cls.archive.getmembers()
            if member.isfile()
        }
        cls.manifest = json.loads(cls._read("submission_manifest.json"))

    @classmethod
    def tearDownClass(cls):
        cls.archive.close()

    @classmethod
    def _bytes(cls, name):
        handle = cls.archive.extractfile(cls.members[name])
        assert handle is not None
        return handle.read()

    @classmethod
    def _read(cls, name):
        return cls._bytes(name).decode("utf-8")

    def test_archive_paths_and_manifest_hashes(self):
        for name in self.members:
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
        for name, expected in self.manifest["files"].items():
            self.assertIn(name, self.members)
            self.assertEqual(sha256(self._bytes(name)).hexdigest(), expected)

    def test_packaged_neural_strategy_trace_and_model_identity(self):
        main = self._read("main.py")
        self.assertIn("IntegratedBeliefPlannerAgent", main)
        self.assertIn("tactical_teacher=TACTICAL_PROVIDER", main)
        self.assertNotIn("Phase6CompetitionAgent", main)
        self.assertFalse(
            self.manifest["tactical_policy"]["teacher_is_strategy_provider"]
        )
        self.assertEqual(
            self.manifest["strategic_status"],
            "integrated_frozen_neural_blueprint_is_final_strategy_provider",
        )
        source_model = json.loads(MODEL.read_text(encoding="utf-8"))
        packaged_model = json.loads(self._read("integrated_planner.json"))
        self.assertEqual(packaged_model["fingerprint"], source_model["fingerprint"])
        self.assertEqual(self.manifest["model"]["fingerprint"], source_model["fingerprint"])

    def test_packaged_deck_is_playable_disclosed_compatibility_list(self):
        deck = [int(value) for value in self._read("deck.csv").splitlines()]
        self.assertEqual(len(deck), 60)
        self.assertTrue(self.manifest["deck_resolution"]["playable"])
        self.assertFalse(self.manifest["deck_resolution"]["exact"])
        self.assertEqual(
            self.manifest["deck_resolution"]["substitutions"],
            [
                {
                    "count": 2,
                    "original_name": "Prism Tower",
                    "replacement_name": "Team Rocket's Watchtower",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
