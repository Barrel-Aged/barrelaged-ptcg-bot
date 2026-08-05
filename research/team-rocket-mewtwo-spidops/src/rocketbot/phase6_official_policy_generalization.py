"""Phase 6AF implementation-heldout official intent-family calibration.

Phase 6AE identifies six controlled opponent policies after fixing player 0.
That does not show that an intent representation transfers across different
implementations of the same declared strategy.  This module makes policy
implementation a split-level nuisance variable: train, calibration, and
validation each use a distinct player-1 policy implementation while player 0,
candidate decks, intent labels, observer orientation, and model inputs retain
the Phase 6AE contract.

The resulting gate is still a controlled intent-family proxy.  It is not human
latent-intent truth, a full public belief, a counterfactual-value provider, a
safe resolver, an exploitation certificate, or official strategy control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .beliefs import OPPONENT_INTENTS
from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    resolve_official_deck,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    rocket_compatibility_profile,
)
from .phase6_aligned_counterfactual_values import ALIGNED_VALUE_SPLITS
from .phase6_aligned_public_range import (
    PUBLIC_RANGE_TEMPERATURE_GRID,
    PUBLIC_RANGE_UNIFORM_MIX_GRID,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_official_intent_calibration import (
    OFFICIAL_INTENT_ECE_LIMIT,
    OFFICIAL_INTENT_L2_GRID,
    OFFICIAL_INTENT_MINIMUM_TOP1,
    OFFICIAL_INTENT_PARTICLES_PER_CANDIDATE,
    OFFICIAL_INTENT_POLICY_VARIANT_IDS,
    OFFICIAL_JOINT_ECE_LIMIT,
    OFFICIAL_JOINT_MINIMUM_TOP1,
    OfficialIntentRangeModel,
    OfficialIntentSelectionMetric,
    OfficialIntentTruthDataset,
    OfficialIntentTruthExample,
    OfficialJointBeliefCalibration,
    OfficialJointBeliefEvaluationRow,
    Phase6OfficialIntentEvidence,
    _capture_official_intent_root,
    _ece,
    _fit_linear_intent_model,
    _linear_logits,
    _softmax_probabilities,
    official_intent_policy_variant_fingerprint,
    evaluate_official_joint_belief,
)
from .phase6_official_range_calibration import (
    OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
    OfficialPublicRangeModel,
    OfficialRangeCalibration,
    OfficialRangeTruthDataset,
    Phase6OfficialRangeEvidence,
)
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_OFFICIAL_POLICY_GENERALIZATION_SCHEMA_VERSION = 1
PHASE6_OFFICIAL_POLICY_GENERALIZATION_SEED = 6_046
OFFICIAL_POLICY_VARIANT_BY_SPLIT = (
    ("train", 0, "canonical"),
    ("train", 1, "canonical"),
    ("train", 2, "engine_first"),
    ("train", 3, "engine_first"),
    ("calibration", 0, "hand_first"),
    ("calibration", 1, "hand_first"),
    ("calibration", 2, "hand_first"),
    ("calibration", 3, "hand_first"),
    ("validation", 0, "action_first"),
    ("validation", 1, "action_first"),
    ("validation", 2, "action_first"),
    ("validation", 3, "action_first"),
)
OFFICIAL_POLICY_ROOTS_PER_CANDIDATE_INTENT_SPLIT = 4
OFFICIAL_POLICY_PARTICLES_PER_CANDIDATE = OFFICIAL_INTENT_PARTICLES_PER_CANDIDATE
OFFICIAL_POLICY_MINIMUM_SLICE_TOP1 = 0.50


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class OfficialPolicyVariantExample:
    """One official root plus split-heldout policy-implementation provenance."""

    base_example: OfficialIntentTruthExample
    policy_variant_id_supervision_only: str
    policy_variant_fingerprint: str
    policy_variant_identifier_selected_for_model_input: bool = False
    policy_variant_is_human_opponent_truth: bool = False

    def __post_init__(self) -> None:
        expected_variant = {
            (split, ordinal): variant
            for split, ordinal, variant in OFFICIAL_POLICY_VARIANT_BY_SPLIT
        }[(self.base_example.split, self.base_example.collection_ordinal)]
        if self.policy_variant_id_supervision_only != expected_variant:
            raise ValueError("official policy variant escaped its split boundary")
        if self.policy_variant_id_supervision_only not in OFFICIAL_INTENT_POLICY_VARIANT_IDS:
            raise ValueError("official policy variant is unsupported")
        _validate_digest(
            self.policy_variant_fingerprint,
            "policy_variant_fingerprint",
        )
        if self.policy_variant_fingerprint != official_intent_policy_variant_fingerprint(
            self.policy_variant_id_supervision_only
        ):
            raise ValueError("official policy variant fingerprint mismatch")
        if (
            self.policy_variant_identifier_selected_for_model_input
            or self.policy_variant_is_human_opponent_truth
        ):
            raise ValueError("official policy variant promoted an invalid claim")

    @property
    def example_id(self) -> str:
        return _fingerprint(
            {
                "base_example_id": self.base_example.example_id,
                "policy_variant_id": self.policy_variant_id_supervision_only,
                "policy_variant_fingerprint": self.policy_variant_fingerprint,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE6_OFFICIAL_POLICY_GENERALIZATION_SCHEMA_VERSION,
            "example_id": self.example_id,
            "policy_variant_id_supervision_only": (
                self.policy_variant_id_supervision_only
            ),
            "policy_variant_fingerprint": self.policy_variant_fingerprint,
            "base_example": self.base_example.to_dict(),
            "claims": {
                "policy_variant_identifier_selected_for_model_input": (
                    self.policy_variant_identifier_selected_for_model_input
                ),
                "policy_variant_is_human_opponent_truth": (
                    self.policy_variant_is_human_opponent_truth
                ),
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPolicyVariantExample":
        claims = raw["claims"]
        result = cls(
            base_example=OfficialIntentTruthExample.from_dict(raw["base_example"]),
            policy_variant_id_supervision_only=str(
                raw["policy_variant_id_supervision_only"]
            ),
            policy_variant_fingerprint=str(raw["policy_variant_fingerprint"]),
            policy_variant_identifier_selected_for_model_input=bool(
                claims["policy_variant_identifier_selected_for_model_input"]
            ),
            policy_variant_is_human_opponent_truth=bool(
                claims["policy_variant_is_human_opponent_truth"]
            ),
        )
        if raw.get("example_id") != result.example_id:
            raise ValueError("official policy variant example fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialPolicyVariantDataset:
    source_phase6ae_dataset_fingerprint: str
    source_phase6ae_model_fingerprint: str
    source_phase6ae_calibration_fingerprint: str
    source_phase6ae_evidence_fingerprint: str
    source_phase6ad_dataset_fingerprint: str
    source_phase6ad_model_fingerprint: str
    source_phase6ad_calibration_fingerprint: str
    source_phase6ad_evidence_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    policy_variant_by_split: tuple[tuple[str, int, str], ...]
    roots_per_candidate_intent_split: int
    particles_per_candidate: int
    examples: tuple[OfficialPolicyVariantExample, ...]
    policy_variants_split_disjoint: bool = True
    target_policy_fixed_across_intents_and_variants: bool = True
    opponent_only_action_semantics: bool = True
    observer_orientation_validated: bool = True
    official_battle_roots_executed: bool = True
    intent_labels_are_controlled_family_proxies: bool = True
    intent_labels_are_human_latent_truth: bool = False
    policy_variant_identifier_selected_for_model_input: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6ae_dataset_fingerprint",
            "source_phase6ae_model_fingerprint",
            "source_phase6ae_calibration_fingerprint",
            "source_phase6ae_evidence_fingerprint",
            "source_phase6ad_dataset_fingerprint",
            "source_phase6ad_model_fingerprint",
            "source_phase6ad_calibration_fingerprint",
            "source_phase6ad_evidence_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.policy_variant_by_split != OFFICIAL_POLICY_VARIANT_BY_SPLIT:
            raise ValueError("official policy variant split mapping changed")
        variants_by_split = {
            split: {
                variant
                for candidate_split, _, variant in self.policy_variant_by_split
                if candidate_split == split
            }
            for split in ALIGNED_VALUE_SPLITS
        }
        if any(
            variants_by_split[left].intersection(variants_by_split[right])
            for index, left in enumerate(ALIGNED_VALUE_SPLITS)
            for right in ALIGNED_VALUE_SPLITS[index + 1 :]
        ):
            raise ValueError("official policy implementation sets overlap splits")
        if self.intent_ids != tuple(item.value for item in OPPONENT_INTENTS):
            raise ValueError("official policy intent catalog changed")
        if self.roots_per_candidate_intent_split <= 0 or self.particles_per_candidate <= 1:
            raise ValueError("official policy collection sizes are invalid")
        if not self.examples:
            raise ValueError("official policy dataset is empty")
        expected_support = {
            (split, candidate, intent, ordinal)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
            for intent in self.intent_ids
            for ordinal in range(self.roots_per_candidate_intent_split)
        }
        observed_support = {
            (
                item.base_example.split,
                item.base_example.candidate_id_supervision_only,
                item.base_example.intent_id_supervision_only,
                item.base_example.collection_ordinal,
            )
            for item in self.examples
        }
        if observed_support != expected_support or len(self.examples) != len(
            expected_support
        ):
            raise ValueError("official policy collection support is incomplete")
        root_ids = [item.base_example.source_root_id for item in self.examples]
        example_ids = [item.example_id for item in self.examples]
        if len(root_ids) != len(set(root_ids)) or len(example_ids) != len(
            set(example_ids)
        ):
            raise ValueError("official policy collection identities repeat")
        expected_variant_fingerprints = {
            (split, ordinal): official_intent_policy_variant_fingerprint(variant)
            for split, ordinal, variant in self.policy_variant_by_split
        }
        for item in self.examples:
            if item.policy_variant_fingerprint != expected_variant_fingerprints[
                (item.base_example.split, item.base_example.collection_ordinal)
            ]:
                raise ValueError("official policy example uses the wrong split variant")
            if (
                item.base_example.candidate_id_supervision_only not in self.candidate_ids
                or item.base_example.intent_id_supervision_only not in self.intent_ids
                or item.base_example.candidate_particle_summaries[0].particle_count
                != self.particles_per_candidate
            ):
                raise ValueError("official policy example catalog/accounting mismatch")
        split_roots = {
            split: {
                item.base_example.source_root_id
                for item in self.examples
                if item.base_example.split == split
            }
            for split in ALIGNED_VALUE_SPLITS
        }
        if any(
            split_roots[left].intersection(split_roots[right])
            for index, left in enumerate(ALIGNED_VALUE_SPLITS)
            for right in ALIGNED_VALUE_SPLITS[index + 1 :]
        ):
            raise ValueError("official policy split roots overlap")
        if not all(
            (
                self.policy_variants_split_disjoint,
                self.target_policy_fixed_across_intents_and_variants,
                self.opponent_only_action_semantics,
                self.observer_orientation_validated,
                self.official_battle_roots_executed,
                self.intent_labels_are_controlled_family_proxies,
            )
        ):
            raise ValueError("official policy collection claim changed")
        if any(
            (
                self.intent_labels_are_human_latent_truth,
                self.policy_variant_identifier_selected_for_model_input,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official policy collection promoted an invalid claim")

    @property
    def intent_dataset_view(self) -> OfficialIntentTruthDataset:
        return OfficialIntentTruthDataset(
            source_phase6ad_dataset_fingerprint=(
                self.source_phase6ad_dataset_fingerprint
            ),
            source_phase6ad_model_fingerprint=self.source_phase6ad_model_fingerprint,
            source_phase6ad_calibration_fingerprint=(
                self.source_phase6ad_calibration_fingerprint
            ),
            source_phase6ad_evidence_fingerprint=(
                self.source_phase6ad_evidence_fingerprint
            ),
            runtime_fingerprint=self.runtime_fingerprint,
            deck_fingerprint=self.deck_fingerprint,
            candidate_ids=self.candidate_ids,
            candidate_deck_fingerprints=self.candidate_deck_fingerprints,
            intent_ids=self.intent_ids,
            roots_per_candidate_intent_split=(
                self.roots_per_candidate_intent_split
            ),
            particles_per_candidate=self.particles_per_candidate,
            examples=tuple(item.base_example for item in self.examples),
            collection_split_disjoint=True,
            target_policy_fixed_across_intents=True,
            opponent_only_action_semantics=True,
            observer_orientation_validated=True,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def roots_per_split(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                split,
                sum(item.base_example.split == split for item in self.examples),
            )
            for split in ALIGNED_VALUE_SPLITS
        )

    @property
    def live_selection_count(self) -> int:
        return sum(len(item.base_example.prefix_trace.steps) for item in self.examples)

    @property
    def opponent_main_action_count(self) -> int:
        return sum(
            len(item.base_example.prefix_trace.opponent_main_action_types)
            for item in self.examples
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_POLICY_GENERALIZATION_SCHEMA_VERSION,
            "sources": {
                "phase6ae_dataset_fingerprint": (
                    self.source_phase6ae_dataset_fingerprint
                ),
                "phase6ae_model_fingerprint": self.source_phase6ae_model_fingerprint,
                "phase6ae_calibration_fingerprint": (
                    self.source_phase6ae_calibration_fingerprint
                ),
                "phase6ae_evidence_fingerprint": (
                    self.source_phase6ae_evidence_fingerprint
                ),
                "phase6ad_dataset_fingerprint": (
                    self.source_phase6ad_dataset_fingerprint
                ),
                "phase6ad_model_fingerprint": self.source_phase6ad_model_fingerprint,
                "phase6ad_calibration_fingerprint": (
                    self.source_phase6ad_calibration_fingerprint
                ),
                "phase6ad_evidence_fingerprint": (
                    self.source_phase6ad_evidence_fingerprint
                ),
            },
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": [
                list(item) for item in self.candidate_deck_fingerprints
            ],
            "intent_ids": list(self.intent_ids),
            "policy_variant_by_split": [
                list(item) for item in self.policy_variant_by_split
            ],
            "collection": {
                "roots_per_candidate_intent_split": (
                    self.roots_per_candidate_intent_split
                ),
                "particles_per_candidate": self.particles_per_candidate,
                "root_count": len(self.examples),
                "roots_per_split": dict(self.roots_per_split),
                "live_selection_count": self.live_selection_count,
                "opponent_main_action_count": self.opponent_main_action_count,
            },
            "examples": [item.to_dict() for item in self.examples],
            "claims": {
                "policy_variants_split_disjoint": self.policy_variants_split_disjoint,
                "target_policy_fixed_across_intents_and_variants": (
                    self.target_policy_fixed_across_intents_and_variants
                ),
                "opponent_only_action_semantics": self.opponent_only_action_semantics,
                "observer_orientation_validated": self.observer_orientation_validated,
                "official_battle_roots_executed": self.official_battle_roots_executed,
                "intent_labels_are_controlled_family_proxies": (
                    self.intent_labels_are_controlled_family_proxies
                ),
                "intent_labels_are_human_latent_truth": (
                    self.intent_labels_are_human_latent_truth
                ),
                "policy_variant_identifier_selected_for_model_input": (
                    self.policy_variant_identifier_selected_for_model_input
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPolicyVariantDataset":
        source = raw["sources"]
        collection = raw["collection"]
        claims = raw["claims"]
        result = cls(
            source_phase6ae_dataset_fingerprint=str(
                source["phase6ae_dataset_fingerprint"]
            ),
            source_phase6ae_model_fingerprint=str(
                source["phase6ae_model_fingerprint"]
            ),
            source_phase6ae_calibration_fingerprint=str(
                source["phase6ae_calibration_fingerprint"]
            ),
            source_phase6ae_evidence_fingerprint=str(
                source["phase6ae_evidence_fingerprint"]
            ),
            source_phase6ad_dataset_fingerprint=str(
                source["phase6ad_dataset_fingerprint"]
            ),
            source_phase6ad_model_fingerprint=str(
                source["phase6ad_model_fingerprint"]
            ),
            source_phase6ad_calibration_fingerprint=str(
                source["phase6ad_calibration_fingerprint"]
            ),
            source_phase6ad_evidence_fingerprint=str(
                source["phase6ad_evidence_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (str(item[0]), str(item[1]))
                for item in raw["candidate_deck_fingerprints"]
            ),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            policy_variant_by_split=tuple(
                (str(item[0]), int(item[1]), str(item[2]))
                for item in raw["policy_variant_by_split"]
            ),
            roots_per_candidate_intent_split=int(
                collection["roots_per_candidate_intent_split"]
            ),
            particles_per_candidate=int(collection["particles_per_candidate"]),
            examples=tuple(
                OfficialPolicyVariantExample.from_dict(item)
                for item in raw["examples"]
            ),
            policy_variants_split_disjoint=bool(
                claims["policy_variants_split_disjoint"]
            ),
            target_policy_fixed_across_intents_and_variants=bool(
                claims["target_policy_fixed_across_intents_and_variants"]
            ),
            opponent_only_action_semantics=bool(
                claims["opponent_only_action_semantics"]
            ),
            observer_orientation_validated=bool(
                claims["observer_orientation_validated"]
            ),
            official_battle_roots_executed=bool(
                claims["official_battle_roots_executed"]
            ),
            intent_labels_are_controlled_family_proxies=bool(
                claims["intent_labels_are_controlled_family_proxies"]
            ),
            intent_labels_are_human_latent_truth=bool(
                claims["intent_labels_are_human_latent_truth"]
            ),
            policy_variant_identifier_selected_for_model_input=bool(
                claims["policy_variant_identifier_selected_for_model_input"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_collection = {
            "roots_per_candidate_intent_split": (
                result.roots_per_candidate_intent_split
            ),
            "particles_per_candidate": result.particles_per_candidate,
            "root_count": len(result.examples),
            "roots_per_split": dict(result.roots_per_split),
            "live_selection_count": result.live_selection_count,
            "opponent_main_action_count": result.opponent_main_action_count,
        }
        if dict(collection) != expected_collection:
            raise ValueError("official policy collection accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official policy dataset fingerprint mismatch")
        return result


def _policy_prediction_records(
    examples: Sequence[OfficialIntentTruthExample],
    *,
    normalizer: NamedFeatureNormalizer,
    weights: Sequence[Sequence[float]],
    biases: Sequence[float],
    temperature: float,
    uniform_mix: float,
) -> tuple[tuple[float, float, float, bool], ...]:
    intent_ids = tuple(item.value for item in OPPONENT_INTENTS)
    records = []
    for example in examples:
        values = normalizer.transform(example.intent_public_features)
        probabilities = _softmax_probabilities(
            _linear_logits(values, weights, biases),
            temperature=temperature,
            uniform_mix=uniform_mix,
        )
        target = intent_ids.index(example.intent_id_supervision_only)
        predicted = min(
            range(len(probabilities)),
            key=lambda index: (-probabilities[index], intent_ids[index]),
        )
        records.append(
            (
                -math.log(max(probabilities[target], 1e-15)),
                sum(
                    (probability - float(index == target)) ** 2
                    for index, probability in enumerate(probabilities)
                ),
                max(probabilities),
                predicted == target,
            )
        )
    return tuple(records)


def _policy_record_metrics(
    records: Sequence[tuple[float, float, float, bool]],
) -> tuple[float, float, float, float]:
    if not records:
        raise ValueError("official policy robust calibration records are empty")
    return (
        sum(item[0] for item in records) / len(records),
        sum(item[1] for item in records) / len(records),
        sum(item[3] for item in records) / len(records),
        min(
            1.0,
            max(0.0, _ece(tuple((item[2], item[3]) for item in records))),
        ),
    )


def _fit_policy_intent_subset(
    examples: Sequence[OfficialIntentTruthExample],
    *,
    l2_penalty: float,
) -> tuple[
    NamedFeatureNormalizer,
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    float,
]:
    rows_fingerprint = _fingerprint(
        [(item.example_id, item.intent_public_features) for item in examples]
    )
    normalizer = NamedFeatureNormalizer.fit(
        [item.intent_public_features for item in examples],
        fit_rows_fingerprint=rows_fingerprint,
    )
    values = torch.tensor(
        [normalizer.transform(item.intent_public_features) for item in examples],
        dtype=torch.float64,
    )
    intent_ids = tuple(item.value for item in OPPONENT_INTENTS)
    labels = torch.tensor(
        [intent_ids.index(item.intent_id_supervision_only) for item in examples],
        dtype=torch.long,
    )
    weights, biases, loss = _fit_linear_intent_model(
        values,
        labels,
        l2_penalty=l2_penalty,
    )
    return normalizer, weights, biases, loss


def train_policy_robust_intent_range(
    dataset: OfficialPolicyVariantDataset,
) -> OfficialIntentRangeModel:
    """Select confidence against two implementations without validation labels.

    Canonical and engine-first train roots are cross-fitted by leaving one
    complete implementation out.  The hand-first calibration variant is
    evaluated by a model fitted on both train implementations.  Hyperparameters
    maximize the worst implementation's top-1 accuracy, then minimize the worst
    ECE/NLL/Brier.  The action-first validation implementation is never read
    during fitting or selection.
    """

    view = dataset.intent_dataset_view
    train = tuple(item for item in view.examples if item.split == "train")
    calibration = tuple(
        item for item in view.examples if item.split == "calibration"
    )
    if {item.collection_ordinal for item in train} != {0, 1, 2, 3}:
        raise ValueError("official policy robust crossfit requires four ordinals")
    train_wrappers = tuple(
        item for item in dataset.examples if item.base_example.split == "train"
    )
    train_variants = ("canonical", "engine_first")
    if {item.policy_variant_id_supervision_only for item in train_wrappers} != set(
        train_variants
    ):
        raise ValueError("official policy robust train variants changed")
    full_fits = {}
    fold_fits = {}
    for l2_penalty in OFFICIAL_INTENT_L2_GRID:
        full_fits[l2_penalty] = _fit_policy_intent_subset(
            train,
            l2_penalty=l2_penalty,
        )
        fold_fits[l2_penalty] = {
            variant: _fit_policy_intent_subset(
                tuple(
                    item.base_example
                    for item in train_wrappers
                    if item.policy_variant_id_supervision_only != variant
                ),
                l2_penalty=l2_penalty,
            )
            for variant in train_variants
        }

    metrics = []
    for l2_penalty in OFFICIAL_INTENT_L2_GRID:
        full_normalizer, full_weights, full_biases, _ = full_fits[l2_penalty]
        for temperature in PUBLIC_RANGE_TEMPERATURE_GRID:
            for uniform_mix in PUBLIC_RANGE_UNIFORM_MIX_GRID:
                development_metrics = []
                for variant in train_variants:
                    normalizer, weights, biases, _ = fold_fits[l2_penalty][variant]
                    records = _policy_prediction_records(
                        tuple(
                            item.base_example
                            for item in train_wrappers
                            if item.policy_variant_id_supervision_only == variant
                        ),
                        normalizer=normalizer,
                        weights=weights,
                        biases=biases,
                        temperature=temperature,
                        uniform_mix=uniform_mix,
                    )
                    development_metrics.append(_policy_record_metrics(records))
                calibration_records = _policy_prediction_records(
                    calibration,
                    normalizer=full_normalizer,
                    weights=full_weights,
                    biases=full_biases,
                    temperature=temperature,
                    uniform_mix=uniform_mix,
                )
                development_metrics.append(
                    _policy_record_metrics(calibration_records)
                )
                metrics.append(
                    OfficialIntentSelectionMetric(
                        l2_penalty=l2_penalty,
                        temperature=temperature,
                        uniform_mix=uniform_mix,
                        calibration_nll=max(item[0] for item in development_metrics),
                        calibration_brier=max(
                            item[1] for item in development_metrics
                        ),
                        calibration_top1_accuracy=min(
                            item[2] for item in development_metrics
                        ),
                        calibration_ece=max(
                            item[3] for item in development_metrics
                        ),
                    )
                )
    selected = min(
        metrics,
        key=lambda item: (
            -item.calibration_top1_accuracy,
            item.calibration_ece,
            item.calibration_nll,
            item.calibration_brier,
            item.l2_penalty,
            item.temperature,
            item.uniform_mix,
        ),
    )
    normalizer, weights, biases, training_loss = full_fits[selected.l2_penalty]
    train_rows_fingerprint = _fingerprint(
        [(item.example_id, item.intent_public_features) for item in train]
    )
    if normalizer.fit_rows_fingerprint != train_rows_fingerprint:
        raise RuntimeError("official policy final normalizer provenance changed")
    return OfficialIntentRangeModel(
        source_dataset_fingerprint=view.fingerprint,
        source_phase6ad_model_fingerprint=view.source_phase6ad_model_fingerprint,
        runtime_fingerprint=view.runtime_fingerprint,
        deck_fingerprint=view.deck_fingerprint,
        candidate_ids=view.candidate_ids,
        intent_ids=view.intent_ids,
        normalizer=normalizer,
        weights=weights,
        biases=biases,
        selection_metrics=tuple(metrics),
        selected_l2_penalty=selected.l2_penalty,
        selected_temperature=selected.temperature,
        selected_uniform_mix=selected.uniform_mix,
        selected_training_loss=training_loss,
        train_root_ids=tuple(item.source_root_id for item in train),
        calibration_root_ids=tuple(
            item.source_root_id for item in calibration
        ),
        train_rows_fingerprint=train_rows_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class OfficialPolicyHeldoutIntentModel:
    source_dataset_fingerprint: str
    intent_dataset_view_fingerprint: str
    policy_variant_by_split: tuple[tuple[str, int, str], ...]
    intent_model: OfficialIntentRangeModel
    selection_protocol: str = (
        "training_policy_leave_one_variant_out_plus_calibration_variant_minimax"
    )
    training_policy_crossfit_variants: tuple[str, ...] = (
        "canonical",
        "engine_first",
    )
    policy_variant_identifier_selected_for_model_input: bool = False
    validation_policy_variant_selected_during_training: bool = False
    intent_labels_are_human_latent_truth: bool = False
    official_engine_executed: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_dataset_fingerprint, "source_dataset_fingerprint")
        _validate_digest(
            self.intent_dataset_view_fingerprint,
            "intent_dataset_view_fingerprint",
        )
        if self.intent_model.source_dataset_fingerprint != self.intent_dataset_view_fingerprint:
            raise ValueError("official policy model/source view mismatch")
        if self.policy_variant_by_split != OFFICIAL_POLICY_VARIANT_BY_SPLIT:
            raise ValueError("official policy model split variants changed")
        if self.selection_protocol != (
            "training_policy_leave_one_variant_out_plus_calibration_variant_minimax"
        ):
            raise ValueError("official policy robust selection protocol changed")
        if self.training_policy_crossfit_variants != (
            "canonical",
            "engine_first",
        ):
            raise ValueError("official policy crossfit variants changed")
        if any(
            (
                self.policy_variant_identifier_selected_for_model_input,
                self.validation_policy_variant_selected_during_training,
                self.intent_labels_are_human_latent_truth,
                self.official_engine_executed,
            )
        ):
            raise ValueError("official policy model promoted an invalid claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_POLICY_GENERALIZATION_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "intent_dataset_view_fingerprint": self.intent_dataset_view_fingerprint,
            "policy_variant_by_split": [
                list(item) for item in self.policy_variant_by_split
            ],
            "intent_model": self.intent_model.to_dict(),
            "robust_selection": {
                "protocol": self.selection_protocol,
                "training_policy_crossfit_variants": list(
                    self.training_policy_crossfit_variants
                ),
                "metric_semantics": {
                    "top1": "minimum_across_training_policy_loo_and_calibration_variant",
                    "ece": "maximum_across_training_policy_loo_and_calibration_variant",
                    "nll": "maximum_across_training_policy_loo_and_calibration_variant",
                    "brier": "maximum_across_training_policy_loo_and_calibration_variant",
                },
            },
            "claims": {
                "policy_variant_identifier_selected_for_model_input": (
                    self.policy_variant_identifier_selected_for_model_input
                ),
                "validation_policy_variant_selected_during_training": (
                    self.validation_policy_variant_selected_during_training
                ),
                "intent_labels_are_human_latent_truth": (
                    self.intent_labels_are_human_latent_truth
                ),
                "official_engine_executed": self.official_engine_executed,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPolicyHeldoutIntentModel":
        claims = raw["claims"]
        robust = raw["robust_selection"]
        if dict(robust["metric_semantics"]) != {
            "top1": "minimum_across_training_policy_loo_and_calibration_variant",
            "ece": "maximum_across_training_policy_loo_and_calibration_variant",
            "nll": "maximum_across_training_policy_loo_and_calibration_variant",
            "brier": "maximum_across_training_policy_loo_and_calibration_variant",
        }:
            raise ValueError("official policy robust metric semantics changed")
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            intent_dataset_view_fingerprint=str(
                raw["intent_dataset_view_fingerprint"]
            ),
            policy_variant_by_split=tuple(
                (str(item[0]), int(item[1]), str(item[2]))
                for item in raw["policy_variant_by_split"]
            ),
            intent_model=OfficialIntentRangeModel.from_dict(raw["intent_model"]),
            selection_protocol=str(robust["protocol"]),
            training_policy_crossfit_variants=tuple(
                str(value)
                for value in robust["training_policy_crossfit_variants"]
            ),
            policy_variant_identifier_selected_for_model_input=bool(
                claims["policy_variant_identifier_selected_for_model_input"]
            ),
            validation_policy_variant_selected_during_training=bool(
                claims["validation_policy_variant_selected_during_training"]
            ),
            intent_labels_are_human_latent_truth=bool(
                claims["intent_labels_are_human_latent_truth"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official policy model fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialIntentFamilySliceMetric:
    intent_id: str
    roots: int
    nll: float
    top1_accuracy: float
    ece: float

    def __post_init__(self) -> None:
        if self.intent_id not in {item.value for item in OPPONENT_INTENTS}:
            raise ValueError("official intent-family slice is unsupported")
        if self.roots <= 0 or self.nll < 0 or not math.isfinite(self.nll):
            raise ValueError("official intent-family slice accounting is invalid")
        if not 0 <= self.top1_accuracy <= 1 or not 0 <= self.ece <= 1:
            raise ValueError("official intent-family slice metric is invalid")


def _validation_intent_family_slices(
    rows: Sequence[OfficialJointBeliefEvaluationRow],
) -> tuple[OfficialIntentFamilySliceMetric, ...]:
    result = []
    for intent in OPPONENT_INTENTS:
        selected = tuple(
            row
            for row in rows
            if row.split == "validation"
            and row.target_intent_id_supervision_only == intent.value
        )
        metrics = tuple(row.intent_metrics for row in selected)
        result.append(
            OfficialIntentFamilySliceMetric(
                intent_id=intent.value,
                roots=len(selected),
                nll=sum(item[0] for item in metrics) / len(metrics),
                top1_accuracy=sum(item[3] for item in metrics) / len(metrics),
                ece=min(
                    1.0,
                    max(0.0, _ece(tuple((item[2], item[3]) for item in metrics))),
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class OfficialPolicyHeldoutCalibration:
    dataset_fingerprint: str
    model_fingerprint: str
    joint_calibration: OfficialJointBeliefCalibration
    validation_intent_family_slices: tuple[OfficialIntentFamilySliceMetric, ...]
    minimum_slice_top1: float
    source_phase6ae_gate_passed: bool
    policy_variant_split_gate_passed: bool
    heldout_intent_family_slice_gate_passed: bool
    implementation_heldout_joint_range_calibrated: bool
    human_latent_intent_calibrated: bool = False
    full_official_public_belief_calibrated: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_compatible: bool = False
    opponent_exploitation_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(self.model_fingerprint, "model_fingerprint")
        if self.minimum_slice_top1 != OFFICIAL_POLICY_MINIMUM_SLICE_TOP1:
            raise ValueError("official policy slice threshold changed")
        if tuple(item.intent_id for item in self.validation_intent_family_slices) != tuple(
            item.value for item in OPPONENT_INTENTS
        ):
            raise ValueError("official policy intent-family slices are incomplete")
        expected_slice = all(
            item.top1_accuracy >= self.minimum_slice_top1
            for item in self.validation_intent_family_slices
        )
        if self.heldout_intent_family_slice_gate_passed != expected_slice:
            raise ValueError("official policy intent-family slice gate is inconsistent")
        expected = all(
            (
                self.source_phase6ae_gate_passed,
                self.policy_variant_split_gate_passed,
                self.heldout_intent_family_slice_gate_passed,
                self.joint_calibration.in_scope_joint_candidate_intent_resource_range_calibrated,
            )
        )
        if self.implementation_heldout_joint_range_calibrated != expected:
            raise ValueError("official policy heldout gate is inconsistent")
        if any(
            (
                self.human_latent_intent_calibrated,
                self.full_official_public_belief_calibrated,
                self.counterfactual_value_compatible,
                self.safe_resolver_compatible,
                self.opponent_exploitation_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("official policy calibration promoted a downstream claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_POLICY_GENERALIZATION_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "joint_calibration": self.joint_calibration.to_dict(),
            "validation_intent_family_slices": [
                asdict(item) for item in self.validation_intent_family_slices
            ],
            "minimum_slice_top1": self.minimum_slice_top1,
            "gates": {
                "source_phase6ae_gate_passed": self.source_phase6ae_gate_passed,
                "policy_variant_split_gate_passed": (
                    self.policy_variant_split_gate_passed
                ),
                "heldout_intent_family_slice_gate_passed": (
                    self.heldout_intent_family_slice_gate_passed
                ),
                "implementation_heldout_joint_range_calibrated": (
                    self.implementation_heldout_joint_range_calibrated
                ),
                "human_latent_intent_calibrated": (
                    self.human_latent_intent_calibrated
                ),
                "full_official_public_belief_calibrated": (
                    self.full_official_public_belief_calibrated
                ),
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "opponent_exploitation_compatible": (
                    self.opponent_exploitation_compatible
                ),
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPolicyHeldoutCalibration":
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            joint_calibration=OfficialJointBeliefCalibration.from_dict(
                raw["joint_calibration"]
            ),
            validation_intent_family_slices=tuple(
                OfficialIntentFamilySliceMetric(**dict(item))
                for item in raw["validation_intent_family_slices"]
            ),
            minimum_slice_top1=float(raw["minimum_slice_top1"]),
            source_phase6ae_gate_passed=bool(gates["source_phase6ae_gate_passed"]),
            policy_variant_split_gate_passed=bool(
                gates["policy_variant_split_gate_passed"]
            ),
            heldout_intent_family_slice_gate_passed=bool(
                gates["heldout_intent_family_slice_gate_passed"]
            ),
            implementation_heldout_joint_range_calibrated=bool(
                gates["implementation_heldout_joint_range_calibrated"]
            ),
            human_latent_intent_calibrated=bool(
                gates["human_latent_intent_calibrated"]
            ),
            full_official_public_belief_calibrated=bool(
                gates["full_official_public_belief_calibrated"]
            ),
            counterfactual_value_compatible=bool(
                gates["counterfactual_value_compatible"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            opponent_exploitation_compatible=bool(
                gates["opponent_exploitation_compatible"]
            ),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official policy calibration fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialPolicyGeneralizationEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    roots_per_split: int
    live_selection_count: int
    opponent_main_action_count: int
    total_public_consistent_particles: int
    validation_intent_nll: float
    validation_intent_top1_accuracy: float
    validation_intent_ece: float
    validation_joint_top1_accuracy: float
    validation_joint_ece: float
    validation_resource_ece: float
    minimum_validation_intent_slice_top1: float
    implementation_heldout_joint_range_calibrated: bool
    human_latent_intent_calibrated: bool = False
    full_official_public_belief_calibrated: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False
    claims: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AF evidence identity is missing")
        for name in (
            "dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if min(
            self.root_count,
            self.roots_per_split,
            self.live_selection_count,
            self.opponent_main_action_count,
            self.total_public_consistent_particles,
        ) <= 0:
            raise ValueError("Phase 6AF evidence accounting is invalid")
        for name in (
            "validation_intent_nll",
            "validation_intent_ece",
            "validation_joint_ece",
            "validation_resource_ece",
        ):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"Phase 6AF {name} is invalid")
        for name in (
            "validation_intent_top1_accuracy",
            "validation_joint_top1_accuracy",
            "minimum_validation_intent_slice_top1",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Phase 6AF {name} is invalid")
        if any(
            (
                self.human_latent_intent_calibrated,
                self.full_official_public_belief_calibrated,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AF evidence promoted a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_POLICY_GENERALIZATION_SCHEMA_VERSION,
            **asdict(self),
        }
        result["claims"] = dict(self.claims)
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AF implementation-heldout official intent evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Fresh official roots | {self.root_count} |
| Train / calibration / validation roots | {self.roots_per_split} / {self.roots_per_split} / {self.roots_per_split} |
| Genuine live selections | {self.live_selection_count} |
| Opponent public MAIN actions | {self.opponent_main_action_count} |
| Public-consistent allocation particles | {self.total_public_consistent_particles} |
| Validation intent NLL / top-1 / ECE | `{self.validation_intent_nll:.6f}` / `{self.validation_intent_top1_accuracy:.6f}` / `{self.validation_intent_ece:.6f}` |
| Validation joint top-1 / ECE | `{self.validation_joint_top1_accuracy:.6f}` / `{self.validation_joint_ece:.6f}` |
| Validation resource ECE | `{self.validation_resource_ece:.6f}` |
| Minimum validation intent-family slice top-1 | `{self.minimum_validation_intent_slice_top1:.6f}` |
| Implementation-heldout joint range | **{str(self.implementation_heldout_joint_range_calibrated).lower()}** |
| Human latent intent / full belief | **false / false** |
| Counterfactual values / safe resolver / official control | **false / false / false** |

Official battles, legal selections, and public action sequences are genuine.
The four policy implementations, six intent families, model, particles, and
joint composition are controlled abstractions. No strategy control, submission,
or strength claim is enabled.
"""

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "Phase6OfficialPolicyGeneralizationEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        values["claims"] = tuple(
            (str(name), str(value)) for name, value in values["claims"].items()
        )
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AF evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialPolicyGeneralizationArtifacts:
    dataset: OfficialPolicyVariantDataset
    model: OfficialPolicyHeldoutIntentModel
    calibration: OfficialPolicyHeldoutCalibration
    evidence: Phase6OfficialPolicyGeneralizationEvidence


def collect_official_policy_variant_dataset(
    *,
    source_phase6ae_dataset: OfficialIntentTruthDataset,
    source_phase6ae_model: OfficialIntentRangeModel,
    source_phase6ae_calibration: OfficialJointBeliefCalibration,
    source_phase6ae_evidence: Phase6OfficialIntentEvidence,
    source_phase6ad_calibration_fingerprint: str,
    source_phase6ad_evidence_fingerprint: str,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    visualize_data: Any,
    roots_per_candidate_intent_split: int,
    particles_per_candidate: int,
    maximum_live_selections: int,
    maximum_capture_attempts: int,
    seed: int,
) -> OfficialPolicyVariantDataset:
    if roots_per_candidate_intent_split <= 0 or particles_per_candidate <= 1:
        raise ValueError("Phase 6AF collection sizes are invalid")
    examples = []
    variant_by_split_ordinal = {
        (split, ordinal): variant
        for split, ordinal, variant in OFFICIAL_POLICY_VARIANT_BY_SPLIT
    }
    for split_index, split in enumerate(ALIGNED_VALUE_SPLITS):
        for candidate_index, candidate in enumerate(candidates):
            for intent_index, intent in enumerate(OPPONENT_INTENTS):
                for collection_ordinal in range(roots_per_candidate_intent_split):
                    variant = variant_by_split_ordinal[(split, collection_ordinal)]
                    capture_error: RuntimeError | None = None
                    for attempt in range(1, maximum_capture_attempts + 1):
                        target_reverse = bool(
                            (
                                split_index
                                + candidate_index
                                + intent_index
                                + collection_ordinal
                                + attempt
                                - 1
                            )
                            % 2
                        )
                        try:
                            base = _capture_official_intent_root(
                                battle_start,
                                battle_select,
                                battle_finish,
                                visualize_data,
                                target=target,
                                candidate=candidate,
                                candidates=candidates,
                                intent=intent,
                                split=split,
                                collection_ordinal=collection_ordinal,
                                capture_attempt=attempt,
                                target_reverse=target_reverse,
                                particles_per_candidate=particles_per_candidate,
                                maximum_live_selections=maximum_live_selections,
                                seed=seed,
                                opponent_policy_variant_id=variant,
                            )
                            examples.append(
                                OfficialPolicyVariantExample(
                                    base_example=base,
                                    policy_variant_id_supervision_only=variant,
                                    policy_variant_fingerprint=(
                                        official_intent_policy_variant_fingerprint(
                                            variant
                                        )
                                    ),
                                )
                            )
                            break
                        except RuntimeError as exc:
                            capture_error = exc
                    else:
                        raise RuntimeError(
                            "Phase 6AF exhausted capture attempts for "
                            f"{split}/{candidate.archetype}/{intent.value}/"
                            f"{collection_ordinal}"
                        ) from capture_error
    dataset = OfficialPolicyVariantDataset(
        source_phase6ae_dataset_fingerprint=source_phase6ae_dataset.fingerprint,
        source_phase6ae_model_fingerprint=source_phase6ae_model.fingerprint,
        source_phase6ae_calibration_fingerprint=(
            source_phase6ae_calibration.fingerprint
        ),
        source_phase6ae_evidence_fingerprint=source_phase6ae_evidence.fingerprint,
        source_phase6ad_dataset_fingerprint=(
            source_phase6ae_dataset.source_phase6ad_dataset_fingerprint
        ),
        source_phase6ad_model_fingerprint=(
            source_phase6ae_dataset.source_phase6ad_model_fingerprint
        ),
        source_phase6ad_calibration_fingerprint=(
            source_phase6ad_calibration_fingerprint
        ),
        source_phase6ad_evidence_fingerprint=source_phase6ad_evidence_fingerprint,
        runtime_fingerprint=source_phase6ae_dataset.runtime_fingerprint,
        deck_fingerprint=source_phase6ae_dataset.deck_fingerprint,
        candidate_ids=source_phase6ae_dataset.candidate_ids,
        candidate_deck_fingerprints=(
            source_phase6ae_dataset.candidate_deck_fingerprints
        ),
        intent_ids=source_phase6ae_dataset.intent_ids,
        policy_variant_by_split=OFFICIAL_POLICY_VARIANT_BY_SPLIT,
        roots_per_candidate_intent_split=roots_per_candidate_intent_split,
        particles_per_candidate=particles_per_candidate,
        examples=tuple(examples),
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AF serialized a forbidden raw private key")
    restored = OfficialPolicyVariantDataset.from_dict(dataset.to_dict())
    if restored != dataset:
        raise RuntimeError("Phase 6AF dataset roundtrip changed values")
    return dataset


def run_phase6_official_policy_generalization(
    *,
    project_root: str | Path | None = None,
    source_phase6ad_artifact_directory: str | Path | None = None,
    source_phase6ae_artifact_directory: str | Path | None = None,
    roots_per_candidate_intent_split: int = (
        OFFICIAL_POLICY_ROOTS_PER_CANDIDATE_INTENT_SPLIT
    ),
    particles_per_candidate: int = OFFICIAL_POLICY_PARTICLES_PER_CANDIDATE,
    maximum_live_selections: int = 900,
    maximum_capture_attempts: int = 8,
    seed: int = PHASE6_OFFICIAL_POLICY_GENERALIZATION_SEED,
) -> Phase6OfficialPolicyGeneralizationArtifacts:
    if min(
        roots_per_candidate_intent_split,
        particles_per_candidate,
        maximum_live_selections,
        maximum_capture_attempts,
    ) <= 0:
        raise ValueError("Phase 6AF limits must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    phase6ad_directory = Path(
        source_phase6ad_artifact_directory
        or root / "artifacts" / "phase6_official_range_calibration"
    ).resolve()
    phase6ae_directory = Path(
        source_phase6ae_artifact_directory
        or root / "artifacts" / "phase6_official_intent_calibration"
    ).resolve()

    def load(directory: Path, name: str) -> Mapping[str, Any]:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    source_phase6ad_dataset = OfficialRangeTruthDataset.from_dict(
        load(phase6ad_directory, "official_range_truth_dataset.json")
    )
    source_phase6ad_model = OfficialPublicRangeModel.from_dict(
        load(phase6ad_directory, "official_public_range_model.json")
    )
    source_phase6ad_calibration = OfficialRangeCalibration.from_dict(
        load(phase6ad_directory, "official_range_calibration.json")
    )
    source_phase6ad_evidence = Phase6OfficialRangeEvidence.from_dict(
        load(phase6ad_directory, "phase6ad_evidence.json")
    )
    source_phase6ae_dataset = OfficialIntentTruthDataset.from_dict(
        load(phase6ae_directory, "official_intent_truth_dataset.json")
    )
    source_phase6ae_model = OfficialIntentRangeModel.from_dict(
        load(phase6ae_directory, "official_intent_range_model.json")
    )
    source_phase6ae_calibration = OfficialJointBeliefCalibration.from_dict(
        load(phase6ae_directory, "official_joint_belief_calibration.json")
    )
    source_phase6ae_evidence = Phase6OfficialIntentEvidence.from_dict(
        load(phase6ae_directory, "phase6ae_evidence.json")
    )
    source_phase6ae_gate = all(
        (
            source_phase6ae_calibration.in_scope_joint_candidate_intent_resource_range_calibrated,
            source_phase6ae_evidence.in_scope_joint_candidate_intent_resource_range_calibrated,
            source_phase6ae_model.source_dataset_fingerprint
            == source_phase6ae_dataset.fingerprint,
            source_phase6ad_calibration.in_scope_official_hidden_card_range_calibrated,
            source_phase6ad_evidence.in_scope_official_hidden_card_range_calibrated,
            source_phase6ad_model.source_dataset_fingerprint
            == source_phase6ad_dataset.fingerprint,
        )
    )
    if not source_phase6ae_gate:
        raise ValueError("Phase 6AF requires admitted Phase 6AD/6AE sources")

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    candidates = tuple(
        profile_by_id[name] for name in source_phase6ae_dataset.candidate_ids
    )
    if (
        set(profile_by_id) != set(source_phase6ae_dataset.candidate_ids)
        or target.fingerprint != source_phase6ae_dataset.deck_fingerprint
    ):
        raise ValueError("Phase 6AF candidate/deck source mismatch")

    runtime = root / "work" / "official_kaggle" / "sample_submission"
    import sys

    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
        visualize_data,
    )

    dataset = collect_official_policy_variant_dataset(
        source_phase6ae_dataset=source_phase6ae_dataset,
        source_phase6ae_model=source_phase6ae_model,
        source_phase6ae_calibration=source_phase6ae_calibration,
        source_phase6ae_evidence=source_phase6ae_evidence,
        source_phase6ad_calibration_fingerprint=(
            source_phase6ad_calibration.fingerprint
        ),
        source_phase6ad_evidence_fingerprint=source_phase6ad_evidence.fingerprint,
        target=target,
        candidates=candidates,
        battle_start=battle_start,
        battle_select=battle_select,
        battle_finish=battle_finish,
        visualize_data=visualize_data,
        roots_per_candidate_intent_split=roots_per_candidate_intent_split,
        particles_per_candidate=particles_per_candidate,
        maximum_live_selections=maximum_live_selections,
        maximum_capture_attempts=maximum_capture_attempts,
        seed=seed,
    )
    intent_view = dataset.intent_dataset_view
    intent_model = train_policy_robust_intent_range(dataset)
    model = OfficialPolicyHeldoutIntentModel(
        source_dataset_fingerprint=dataset.fingerprint,
        intent_dataset_view_fingerprint=intent_view.fingerprint,
        policy_variant_by_split=dataset.policy_variant_by_split,
        intent_model=intent_model,
    )
    restored_model = OfficialPolicyHeldoutIntentModel.from_dict(model.to_dict())
    if restored_model != model:
        raise RuntimeError("Phase 6AF model roundtrip changed values")
    joint = evaluate_official_joint_belief(
        intent_view,
        source_phase6ad_model,
        source_phase6ad_calibration,
        intent_model,
    )
    slices = _validation_intent_family_slices(joint.evaluation_rows)
    slice_gate = all(
        item.top1_accuracy >= OFFICIAL_POLICY_MINIMUM_SLICE_TOP1 for item in slices
    )
    calibration = OfficialPolicyHeldoutCalibration(
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        joint_calibration=joint,
        validation_intent_family_slices=slices,
        minimum_slice_top1=OFFICIAL_POLICY_MINIMUM_SLICE_TOP1,
        source_phase6ae_gate_passed=source_phase6ae_gate,
        policy_variant_split_gate_passed=dataset.policy_variants_split_disjoint,
        heldout_intent_family_slice_gate_passed=slice_gate,
        implementation_heldout_joint_range_calibrated=all(
            (
                source_phase6ae_gate,
                dataset.policy_variants_split_disjoint,
                slice_gate,
                joint.in_scope_joint_candidate_intent_resource_range_calibrated,
            )
        ),
    )
    restored_calibration = OfficialPolicyHeldoutCalibration.from_dict(
        calibration.to_dict()
    )
    if restored_calibration != calibration:
        raise RuntimeError("Phase 6AF calibration roundtrip changed values")
    evidence = Phase6OfficialPolicyGeneralizationEvidence(
        benchmark_id="phase6af-implementation-heldout-official-intent-v1",
        status=(
            "implementation_heldout_declared_intent_family_calibrated_latent_intent_fail_closed"
            if calibration.implementation_heldout_joint_range_calibrated
            else "implementation_heldout_intent_family_rejected_fail_closed"
        ),
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.examples),
        roots_per_split=dict(dataset.roots_per_split)["train"],
        live_selection_count=dataset.live_selection_count,
        opponent_main_action_count=dataset.opponent_main_action_count,
        total_public_consistent_particles=(
            len(dataset.examples)
            * len(dataset.candidate_ids)
            * dataset.particles_per_candidate
        ),
        validation_intent_nll=joint.validation.intent_nll,
        validation_intent_top1_accuracy=joint.validation.intent_top1_accuracy,
        validation_intent_ece=joint.validation.intent_ece,
        validation_joint_top1_accuracy=joint.validation.joint_top1_accuracy,
        validation_joint_ece=joint.validation.joint_ece,
        validation_resource_ece=joint.validation.resource_ece,
        minimum_validation_intent_slice_top1=min(
            item.top1_accuracy for item in slices
        ),
        implementation_heldout_joint_range_calibrated=(
            calibration.implementation_heldout_joint_range_calibrated
        ),
        claims=(
            ("genuine", "fresh official roots, selections, public actions, and transient resource labels"),
            ("heldout_nuisance", "opponent policy implementation is split-disjoint"),
            ("model_input", "public root plus opponent coarse MAIN actions; no variant identifier"),
            ("proxy_scope", "six controlled intent families and four policy implementations"),
            ("human_latent_intent", "not_established"),
            ("downstream", "values, safety, exploitation, and control disabled"),
            ("kaggle_submission", "not_attempted"),
            ("pokemon_strength", "not_claimed"),
            ("phase6_complete", "false"),
        ),
    )
    restored_evidence = Phase6OfficialPolicyGeneralizationEvidence.from_dict(
        evidence.to_dict()
    )
    if restored_evidence != evidence:
        raise RuntimeError("Phase 6AF evidence roundtrip changed values")
    return Phase6OfficialPolicyGeneralizationArtifacts(
        dataset,
        model,
        calibration,
        evidence,
    )


def main() -> None:
    print(run_phase6_official_policy_generalization().evidence.to_json())


if __name__ == "__main__":
    main()
