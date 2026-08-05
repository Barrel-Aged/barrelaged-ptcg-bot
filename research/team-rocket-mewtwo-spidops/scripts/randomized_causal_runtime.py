"""Conservative turn-2 causal ridge overlay for a frozen neural blueprint."""

from __future__ import annotations

from collections import Counter
import copy
import json
import math
import sys

import numpy as np
import torch

from . import features as F
from .net import NNAgent, attack_safe_logits, feat_to_tensors


MAIN_SELECT = 0
MARNIE_MARKERS = frozenset({646, 647, 648, 1259})


def load_causal_ridge(path):
    with np.load(path, allow_pickle=False) as source:
        required = {
            "global_mean",
            "global_components",
            "option_mean",
            "option_components",
            "difference_mean",
            "difference_components",
            "feature_mean",
            "feature_scale",
            "weights",
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"causal ridge artifact is missing {sorted(missing)}")
        return {name: np.asarray(source[name], dtype=np.float32).copy() for name in required}


def _visible_ids(player):
    result = []
    if not isinstance(player, dict):
        return result
    for zone in ("discard", "active", "bench"):
        for item in player.get(zone) or ():
            if not isinstance(item, dict):
                continue
            if item.get("id") is not None:
                result.append(int(item["id"]))
            for key in ("preEvolution", "energyCards", "tools"):
                result.extend(
                    int(card["id"])
                    for card in item.get(key) or ()
                    if isinstance(card, dict) and card.get("id") is not None
                )
    return result


class CausalRidgeRouter:
    def __init__(
        self,
        blueprint_model,
        artifact,
        config,
        *,
        deck_ids,
        opponent_deck_priors,
        candidates=4,
        eligible_turn=2,
    ):
        self.model = blueprint_model
        self.artifact = artifact
        self.confidence = float(config["confidence_multiplier"])
        self.margin = float(config["margin"])
        self.candidates = int(candidates)
        self.eligible_turn = int(eligible_turn)
        self.deck_library = dict(opponent_deck_priors or {})
        self.blueprint = NNAgent(
            blueprint_model,
            greedy=True,
            deck_ids=deck_ids,
            forbid_end_when_attack=True,
            opponent_deck_priors=opponent_deck_priors,
        )
        self.used = False
        self.last_turn = -1
        self.last_decision = None
        self.intervention_count = 0
        self.exception_fallback_count = 0

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
        family = str(getattr(self.blueprint.belief, "opp_policy_family", "") or "")
        prior = self.deck_library.get(family) or ()
        return bool(visible & MARNIE_MARKERS) or Counter(map(int, prior))[646] >= 3

    def _eligible(self, observation, baseline):
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        turn = int(current.get("turn") or 0)
        if turn < self.last_turn:
            self.used = False
        self.last_turn = turn
        return bool(
            not self.used
            and turn == self.eligible_turn
            and select.get("type") == MAIN_SELECT
            and select.get("minCount") == 1
            and select.get("maxCount") == 1
            and len(select.get("option") or ()) >= 2
            and len(baseline) == 1
            and current.get("yourIndex") in (0, 1)
            and current.get("firstPlayer") in (0, 1)
            and current.get("yourIndex") != current.get("firstPlayer")
            and self._is_marnie(observation)
        )

    @staticmethod
    def _semantic_signature(observation, option):
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        me = int(current.get("yourIndex") or 0)
        card_id = int(F._opt_card_id(option, current, me, select) or 0)
        target_area = option.get("inPlayArea")
        target_index = option.get("inPlayIndex")
        if target_area is None and option.get("area") in (4, 5):
            target_area = option.get("area")
            target_index = option.get("index")
        target_player = option.get("playerIndex")
        target_player = me if target_player is None else int(target_player)
        target = None
        if target_area in (4, 5) and target_index is not None:
            players = current.get("players") or ({}, {})
            zone = "active" if int(target_area) == 4 else "bench"
            slots = (players[target_player] or {}).get(zone) or ()
            if 0 <= int(target_index) < len(slots):
                pokemon = slots[int(target_index)]
                if isinstance(pokemon, dict):
                    target = (
                        int(pokemon.get("id") or 0),
                        int(pokemon.get("hp") or 0),
                        tuple(sorted(int(value) for value in pokemon.get("energies") or ())),
                        tuple(
                            sorted(
                                int(card.get("id") or 0)
                                for card in pokemon.get("tools") or ()
                                if isinstance(card, dict)
                            )
                        ),
                    )
        unresolved_source = (
            (option.get("area"), option.get("index")) if card_id == 0 else None
        )
        return (
            int(option.get("type") or 0),
            card_id,
            unresolved_source,
            int(option.get("attackId") or 0),
            int(option.get("number") or 0),
            int(option.get("count") or 0),
            int(option.get("energyIndex") or 0),
            int(option.get("toolIndex") or 0),
            target_player,
            target,
        )

    @torch.no_grad()
    def _encoded(self, observation):
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
        if getattr(self.model, "mem", 0):
            raise ValueError("causal ridge v1 supports only the frozen memory-free blueprint")
        unsqueeze = lambda key: feat[key].unsqueeze(0)
        poke_count = feat["poke_card"].shape[0]
        option_count = feat["opt_card"].shape[0]
        poke_mask = torch.ones(1, poke_count, device=feat["global"].device)
        poke_mask[0, -1] = 0.0
        option_mask = torch.ones(1, option_count, device=feat["global"].device)
        global_token, option_tokens = self.model._tokens(
            unsqueeze("global"),
            unsqueeze("belief_global"),
            unsqueeze("hand_card"),
            unsqueeze("my_discard_card"),
            unsqueeze("opp_discard_card"),
            unsqueeze("stadium_card"),
            unsqueeze("my_unseen_card"),
            unsqueeze("opp_unseen_card"),
            unsqueeze("poke_feats"),
            unsqueeze("poke_card"),
            unsqueeze("opt_feats"),
            unsqueeze("opt_unseen"),
            unsqueeze("opt_card"),
            unsqueeze("opt_attack"),
            unsqueeze("opt_target"),
            poke_mask,
            option_mask,
            kpc=unsqueeze("known_prized_card") if self.model.v4 else None,
            efc=unsqueeze("effect_card") if self.model.v4 else None,
            bpz=unsqueeze("belief_prized") if self.model.v4 else None,
            ctc=unsqueeze("context_card") if self.model.v5 else None,
            pwk=unsqueeze("poke_weak") if self.model.v5 else None,
        )
        logits = self.model.logit_head(option_tokens).squeeze(-1).squeeze(0)
        logits = attack_safe_logits(logits, observation["select"]["option"])
        value_input = torch.cat(
            (
                global_token.squeeze(0).detach(),
                global_token.new_zeros(F.PRIV_DIM),
            )
        )
        state_value = self.model.value(value_input).squeeze(-1)
        return (
            global_token.squeeze(0).detach().cpu().numpy().astype(np.float32),
            option_tokens.squeeze(0).detach().cpu().numpy().astype(np.float32),
            logits.detach().cpu().numpy().astype(np.float32),
            float(state_value),
        )

    def _shortlist(self, observation, baseline_index, logits):
        options = observation["select"]["option"]
        result = [int(baseline_index)]
        signatures = {self._semantic_signature(observation, options[baseline_index])}
        for raw_index in np.argsort(-logits).tolist():
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

    @staticmethod
    def _project(values, mean, components):
        return (np.asarray(values, dtype=np.float32) - mean) @ components.T

    def _candidate_features(
        self, global_token, option_tokens, logits, state_value, shortlist
    ):
        artifact = self.artifact
        tokens = option_tokens[np.asarray(shortlist, dtype=np.int64)]
        selected_logits = np.nan_to_num(
            logits[np.asarray(shortlist, dtype=np.int64)],
            neginf=-50.0,
            posinf=50.0,
        ).astype(np.float32)
        count = len(shortlist)
        global_features = self._project(
            global_token.reshape(1, -1),
            artifact["global_mean"],
            artifact["global_components"],
        )
        global_features = np.repeat(global_features, count, axis=0)
        option_features = self._project(
            tokens,
            artifact["option_mean"],
            artifact["option_components"],
        )
        difference_features = self._project(
            tokens - tokens[0:1],
            artifact["difference_mean"],
            artifact["difference_components"],
        )
        logit_scale = max(1e-4, float(selected_logits.std()))
        normalized_logits = (
            selected_logits - float(selected_logits.mean())
        ) / logit_scale
        logit_delta = selected_logits - selected_logits[0]
        positions = np.zeros((count, 4), dtype=np.float32)
        positions[np.arange(count), np.arange(count)] = 1.0
        raw = np.concatenate(
            (
                global_features,
                np.full((count, 1), state_value, dtype=np.float32),
                option_features,
                difference_features,
                normalized_logits.reshape(-1, 1),
                logit_delta.reshape(-1, 1),
                positions,
                np.full((count, 1), count / 4.0, dtype=np.float32),
            ),
            axis=1,
        )
        return (raw - artifact["feature_mean"]) / artifact["feature_scale"]

    def _policy(self, features):
        design = np.concatenate(
            (np.ones((len(features), 1), dtype=np.float32), features), axis=1
        )
        values = np.clip(self.artifact["weights"] @ design.T, -1.0, 1.0)
        advantages = values[:, 1:] - values[:, 0:1]
        mean = advantages.mean(axis=0)
        std = advantages.std(axis=0, ddof=1)
        lower = mean - self.confidence * std
        local = int(np.argmax(lower))
        score = float(lower[local])
        return (local + 1 if score >= self.margin else 0), values, lower

    def __call__(self, observation):
        baseline = self.blueprint(observation)
        self.last_decision = {"used": False, "baseline": baseline}
        try:
            if not self._eligible(observation, baseline):
                return baseline
            self.used = True
            baseline_index = int(baseline[0])
            global_token, option_tokens, logits, state_value = self._encoded(observation)
            shortlist = self._shortlist(observation, baseline_index, logits)
            if len(shortlist) < 2:
                return baseline
            features = self._candidate_features(
                global_token, option_tokens, logits, state_value, shortlist
            )
            selected_position, values, lower = self._policy(features)
            chosen = int(shortlist[selected_position])
            changed = selected_position != 0
            self.intervention_count += int(changed)
            self.last_decision = {
                "used": True,
                "changed": changed,
                "turn": int((observation.get("current") or {}).get("turn") or 0),
                "baseline": baseline_index,
                "chosen": chosen,
                "selected_position": selected_position,
                "shortlist": shortlist,
                "lower_advantages": lower.tolist(),
                "ensemble_means": values.mean(axis=0).tolist(),
                "confidence_multiplier": self.confidence,
                "margin": self.margin,
            }
            print(
                json.dumps({"randomized_causal_overlay": self.last_decision}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            return [chosen] if changed else baseline
        except Exception as error:
            self.exception_fallback_count += 1
            self.last_decision = {
                "used": False,
                "baseline": baseline,
                "error": f"{type(error).__name__}: {str(error)[:180]}",
            }
            print(
                json.dumps({"randomized_causal_overlay_error": self.last_decision}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            return baseline

