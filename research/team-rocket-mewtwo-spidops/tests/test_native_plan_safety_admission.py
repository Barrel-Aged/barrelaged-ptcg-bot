from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from rocketbot.native_plan_safety_admission import (
    NativePlanSafetyAdmission,
    NativePlanSafetyAdmissionError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativePlanSafetyAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = PROJECT_ROOT / "artifacts" / "phase6_native_safety_admission"
        cls.raw = json.loads(
            (directory / "native_plan_safety_admission.json").read_text(
                encoding="utf-8"
            )
        )
        cls.admission = NativePlanSafetyAdmission.from_dict(cls.raw)
        cls.evidence = json.loads(
            (directory / "phase6q_evidence.json").read_text(encoding="utf-8")
        )

    def test_current_candidate_is_rejected_before_bounds_or_resolving(self) -> None:
        admission = self.admission
        self.assertFalse(admission.admitted)
        self.assertFalse(admission.counterfactual_bounds_enabled)
        self.assertFalse(admission.safe_resolver_enabled)
        self.assertFalse(admission.exploitation_enabled)
        self.assertFalse(admission.official_strategy_control_enabled)
        passed = {item.check_id for item in admission.checks if item.passed}
        self.assertEqual(
            passed,
            {
                "genuine_native_descendant_source",
                "private_data_redaction",
                "checkpoint_restore_drift",
            },
        )
        failed = {item.check_id for item in admission.failed_checks}
        self.assertIn("regret_p95_variance", failed)
        self.assertIn("public_belief_calibration", failed)
        self.assertIn("independent_counterfactual_calibration", failed)
        self.assertIn("equilibrium_convergence_validation", failed)
        with self.assertRaises(NativePlanSafetyAdmissionError):
            admission.require_admitted()

    def test_admission_roundtrips_and_capability_bypass_fails_closed(self) -> None:
        restored = NativePlanSafetyAdmission.from_dict(
            json.loads(self.admission.to_json())
        )
        self.assertEqual(restored.fingerprint, self.admission.fingerprint)

        tampered = copy.deepcopy(self.raw)
        tampered.pop("fingerprint", None)
        tampered["enabled"]["safe_resolver"] = True
        with self.assertRaises(ValueError):
            NativePlanSafetyAdmission.from_dict(tampered)

    def test_evidence_records_no_safety_or_official_execution(self) -> None:
        self.assertEqual(
            self.evidence["status"],
            "phase6q_native_safety_admission_rejected_as_required",
        )
        self.assertFalse(self.evidence["admission"]["admitted"])
        self.assertFalse(self.evidence["execution"]["counterfactual_bounds_built"])
        self.assertFalse(self.evidence["execution"]["resolver_executed"])
        self.assertFalse(self.evidence["execution"]["exploitation_executed"])
        self.assertFalse(
            self.evidence["execution"]["official_strategy_control_enabled"]
        )
        self.assertEqual(self.evidence["claims"]["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
