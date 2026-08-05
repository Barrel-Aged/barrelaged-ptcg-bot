"""Bounded semantic sequencing overlay for a frozen neural blueprint."""

from __future__ import annotations

import copy
import json
import math
import sys

import torch

from . import features as F
from .net import NNAgent, attack_safe_logits, feat_to_tensors


MARNIE_MARKERS = frozenset({646, 647, 648, 1259})


def load_semantic_rules(path: str):
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("schema") != "rocket-semantic-sequencing-rules/v1":
        raise ValueError("unsupported semantic sequencing rule artifact")
    rules = {}
    for row in artifact.get("rules") or ():
        key = (tuple(map(int, row["baseline"])), tuple(map(int, row["alternative"])))
        rules[key] = dict(row.get("statistics") or {})
    if not rules:
        raise ValueError("semantic sequencing artifact contains no rules")
    return rules


def _visible_ids(player):
    result = []
    if not isinstance(player, dict):
        return result
    for zone in ("discard", "active", "bench"):
        for item in player.get(zone) or []:
            if not isinstance(item, dict):
                continue
            if item.get("id") is not None:
                result.append(int(item["id"]))
            for key in ("preEvolution", "energyCards", "tools"):
                result.extend(
                    int(card["id"])
                    for card in item.get(key) or []
                    if isinstance(card, dict) and card.get("id") is not None
                )
    return result


class SemanticRuleRouter:
    def __init__(
        self,
        blueprint_model,
        rules,
        *,
        deck_ids,
        opponent_deck_priors,
        candidates=4,
        latest_turn=8,
    ):
        self.model = blueprint_model
        self.rules = dict(rules)
        self.candidates = int(candidates)
        self.latest_turn = int(latest_turn)
        self.blueprint = NNAgent(
            blueprint_model,
            greedy=True,
            deck_ids=deck_ids,
            forbid_end_when_attack=True,
            opponent_deck_priors=opponent_deck_priors,
        )
        self.used_turns = set()
        self.last_turn = -1
        self.last_decision = None
        self.intervention_count = 0
        self.exception_fallback_count = 0

    @staticmethod
    def _semantic_signature(observation, option):
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        me = int(current.get("yourIndex") or 0)
        return (
            int(option.get("type") or 0),
            int(F._opt_card_id(option, current, me, select) or 0),
            int(option.get("inPlayArea") or 0),
            int(option.get("attackId") or 0),
        )

    def _is_marnie(self, observation):
        current = observation.get("current") or {}
        me = int(current.get("yourIndex") or 0)
        players = current.get("players") or ({}, {})
        visible = set(_visible_ids(players[1 - me]))
        visible.update(
            int(card["id"])
            for card in current.get("stadium") or ()
            if isinstance(card, dict) and card.get("id") is not None
        )
        family = str(getattr(self.blueprint.belief, "opp_policy_family", "") or "").lower()
        return bool(visible & MARNIE_MARKERS) or "marnie" in family

    def _eligible(self, observation, baseline):
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        turn = int(current.get("turn") or 0)
        if turn < self.last_turn:
            self.used_turns.clear()
        self.last_turn = turn
        return bool(
            select.get("type") == 0
            and select.get("minCount") == 1
            and select.get("maxCount") == 1
            and len(select.get("option") or ()) >= 2
            and len(baseline) == 1
            and current.get("yourIndex") in (0, 1)
            and current.get("firstPlayer") in (0, 1)
            and current.get("yourIndex") != current.get("firstPlayer")
            and turn <= self.latest_turn
            and turn not in self.used_turns
            and self._is_marnie(observation)
        )

    @torch.no_grad()
    def _candidate_indices(self, observation, baseline_index):
        belief = self.blueprint.belief
        snapshot = copy.deepcopy(belief)
        try:
            feature = F.extract(
                observation,
                belief=belief,
                slot_link=self.blueprint.slot_link,
            )
        finally:
            if belief is not None:
                belief.__dict__.clear()
                belief.__dict__.update(snapshot.__dict__)
        feat = feat_to_tensors(feature, self.blueprint.device)
        logits, _ = self.model(feat)
        if self.blueprint.forbid_end_when_attack:
            logits = attack_safe_logits(logits, observation["select"]["option"])
        options = observation["select"]["option"]
        result = [int(baseline_index)]
        signatures = {self._semantic_signature(observation, options[baseline_index])}
        for raw_index in torch.argsort(logits, descending=True).tolist():
            index = int(raw_index)
            signature = self._semantic_signature(observation, options[index])
            if (
                index not in result
                and signature not in signatures
                and math.isfinite(float(logits[index]))
            ):
                result.append(index)
                signatures.add(signature)
            if len(result) >= self.candidates:
                break
        return result

    def __call__(self, observation):
        baseline = self.blueprint(observation)
        self.last_decision = {"used": False, "baseline": baseline}
        try:
            if not self._eligible(observation, baseline):
                return baseline
            turn = int((observation.get("current") or {}).get("turn") or 0)
            self.used_turns.add(turn)
            baseline_index = int(baseline[0])
            shortlist = self._candidate_indices(observation, baseline_index)
            options = observation["select"]["option"]
            baseline_signature = self._semantic_signature(
                observation, options[baseline_index]
            )
            matches = []
            for alternative_index in shortlist:
                if alternative_index == baseline_index:
                    continue
                alternative_signature = self._semantic_signature(
                    observation, options[alternative_index]
                )
                statistics = self.rules.get(
                    (baseline_signature, alternative_signature)
                )
                if statistics is not None:
                    matches.append(
                        (
                            float(statistics.get("mean") or 0.0),
                            -float(statistics.get("negative") or 0.0)
                            / max(1.0, float(statistics.get("count") or 0.0)),
                            int(statistics.get("count") or 0),
                            -int(alternative_index),
                            int(alternative_index),
                            alternative_signature,
                        )
                    )
            if not matches:
                self.last_decision = {
                    "used": True,
                    "changed": False,
                    "turn": turn,
                    "baseline": baseline_index,
                    "baseline_signature": baseline_signature,
                    "shortlist": shortlist,
                }
                return baseline
            selected = max(matches)
            chosen = selected[-2]
            self.intervention_count += 1
            self.last_decision = {
                "used": True,
                "changed": True,
                "turn": turn,
                "baseline": baseline_index,
                "chosen": chosen,
                "baseline_signature": baseline_signature,
                "alternative_signature": selected[-1],
                "shortlist": shortlist,
            }
            print(
                json.dumps({"semantic_rule": self.last_decision}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            return [chosen]
        except Exception as error:
            self.exception_fallback_count += 1
            self.last_decision = {
                "used": False,
                "baseline": baseline,
                "error": f"{type(error).__name__}: {str(error)[:160]}",
            }
            return baseline
