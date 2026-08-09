"""Conservative runtime overlay for a frozen causal-advantage Q head.

This file is copied into ``soldier/causal_q.py`` only for candidates that pass
their offline admission gate.  The frozen neural policy remains the blueprint;
the overlay can rerank only its semantic top-four shortlist at the first early
main-phase decision of a turn when Rocket is second into public Marnie markers.
"""

from __future__ import annotations

import copy
import json
import math
import sys

import torch
import torch.nn as nn

from . import features as F
from .net import NNAgent, attack_safe_logits, feat_to_tensors


MARNIE_MARKERS = frozenset({646, 647, 648, 1259})


class CausalQHead(nn.Module):
    def __init__(self, token_width: int, hidden: int = 192) -> None:
        super().__init__()
        self.token_width = int(token_width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.LayerNorm(2 * token_width + 1),
            nn.Linear(2 * token_width + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, global_token, option_tokens, base_logits):
        count = option_tokens.shape[0]
        state = global_token.reshape(1, -1).expand(count, -1)
        value = torch.cat((state, option_tokens, base_logits.reshape(-1, 1)), dim=-1)
        return self.network(value).squeeze(-1)


def load_causal_q(path: str) -> CausalQHead:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != "rocket-live-truth-causal-q/v1":
        raise ValueError("unsupported causal Q artifact")
    model = CausalQHead(**artifact["config"])
    model.load_state_dict(artifact["state_dict"])
    return model.eval()


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


class CausalQRouter:
    def __init__(
        self,
        blueprint_model,
        q_model: CausalQHead,
        *,
        deck_ids,
        opponent_deck_priors,
        threshold: float,
        candidates: int = 4,
        latest_turn: int = 8,
    ) -> None:
        self.model = blueprint_model
        self.q_model = q_model
        self.threshold = float(threshold)
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
        target_player = me if option.get("playerIndex") is None else int(option["playerIndex"])
        target = None
        if target_area in (4, 5) and target_index is not None:
            players = current.get("players") or ({}, {})
            zone = "active" if int(target_area) == 4 else "bench"
            slots = (players[target_player] or {}).get(zone) or ()
            if 0 <= int(target_index) < len(slots) and isinstance(slots[int(target_index)], dict):
                pokemon = slots[int(target_index)]
                target = (
                    int(pokemon.get("id") or 0),
                    int(pokemon.get("hp") or 0),
                    tuple(sorted(int(value) for value in pokemon.get("energies") or ())),
                )
        unresolved = (option.get("area"), option.get("index")) if card_id == 0 else None
        return (
            int(option.get("type") or 0),
            card_id,
            unresolved,
            int(option.get("attackId") or 0),
            int(option.get("number") or 0),
            int(option.get("count") or 0),
            target_player,
            target,
        )

    def _is_marnie(self, observation) -> bool:
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

    def _eligible(self, observation, baseline) -> bool:
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
    def _tokens_and_logits(self, observation):
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
        feat = feat_to_tensors(feature)
        u = lambda key: feat[key].unsqueeze(0)
        poke_count = feat["poke_card"].shape[0]
        option_count = feat["opt_card"].shape[0]
        poke_mask = torch.ones(1, poke_count)
        poke_mask[0, -1] = 0.0
        option_mask = torch.ones(1, option_count)
        global_token, option_tokens = self.model._tokens(
            u("global"), u("belief_global"), u("hand_card"),
            u("my_discard_card"), u("opp_discard_card"), u("stadium_card"),
            u("my_unseen_card"), u("opp_unseen_card"), u("poke_feats"),
            u("poke_card"), u("opt_feats"), u("opt_unseen"), u("opt_card"),
            u("opt_attack"), u("opt_target"), poke_mask, option_mask,
            kpc=u("known_prized_card") if self.model.v4 else None,
            efc=u("effect_card") if self.model.v4 else None,
            bpz=u("belief_prized") if self.model.v4 else None,
            ctc=u("context_card") if self.model.v5 else None,
            pwk=u("poke_weak") if self.model.v5 else None,
        )
        logits = self.model.logit_head(option_tokens).squeeze(-1).squeeze(0)
        logits = attack_safe_logits(logits, observation["select"]["option"])
        return global_token.squeeze(0), option_tokens.squeeze(0), logits

    def _shortlist(self, observation, baseline_index, logits):
        options = observation["select"]["option"]
        result = [int(baseline_index)]
        signatures = {self._semantic_signature(observation, options[baseline_index])}
        for raw_index in torch.argsort(logits, descending=True).tolist():
            index = int(raw_index)
            signature = self._semantic_signature(observation, options[index])
            if index not in result and signature not in signatures and math.isfinite(float(logits[index])):
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
            global_token, option_tokens, logits = self._tokens_and_logits(observation)
            shortlist = self._shortlist(observation, int(baseline[0]), logits)
            index_tensor = torch.tensor(shortlist, dtype=torch.long)
            values = self.q_model(
                global_token,
                option_tokens[index_tensor],
                logits[index_tensor],
            )
            baseline_position = shortlist.index(int(baseline[0]))
            best_position = int(values.argmax())
            chosen = shortlist[best_position]
            advantage = float(values[best_position] - values[baseline_position])
            changed = bool(chosen != baseline[0] and advantage >= self.threshold)
            self.last_decision = {
                "used": True,
                "turn": turn,
                "baseline": int(baseline[0]),
                "chosen": int(chosen) if changed else int(baseline[0]),
                "shortlist": shortlist,
                "predicted_advantage": advantage,
                "threshold": self.threshold,
                "changed": changed,
            }
            if changed:
                print(json.dumps({"causal_q": self.last_decision}, sort_keys=True), file=sys.stderr, flush=True)
                return [int(chosen)]
            return baseline
        except Exception as error:
            self.last_decision = {
                "used": False,
                "baseline": baseline,
                "error": f"{type(error).__name__}: {str(error)[:160]}",
            }
            return baseline
