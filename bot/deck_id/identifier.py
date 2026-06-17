"""DeckIdentifier (Phase 2) — infer the opponent's archetype from revealed cards.

The project's efficiency thesis: instead of reasoning over the full ~1,267-card pool, identify the
opponent's deck from the meta prior + the cards they reveal, so MCTS (Sprint 6) determinizes against
a ~60-card hypothesis. v1 is a Bayesian archetype matcher over the 10 meta decks.

Model: P(arch | evidence) ∝ P(arch) · Π_card P(card | arch). With Laplace smoothing α and a constant
60-card deck size, the per-archetype denominator is identical across archetypes, so it cancels in the
softmax — the log-likelihood of a seen card reduces to log(count_in_arch + α). Cards unique to one
archetype (the ace ex) swing the posterior hard; ubiquitous staples (Ultra Ball, basic energy) barely
move it. Prior = meta share.
"""
from __future__ import annotations

import json
import math
import os

import os as _os

ALPHA = 0.02            # Laplace smoothing: a card absent from an archetype is unlikely, not impossible
# Prior tempering: the meta-share prior is a weak nudge, not a dominant factor — revealed cards should
# decide the archetype. PRIOR_TEMP in [0,1]: 0 = uniform prior (pure evidence), 1 = full meta share.
PRIOR_TEMP = float(_os.environ.get("PTCG_PRIOR_TEMP", "0.3"))
_ARCHETYPES = None


def _load() -> list[dict]:
    global _ARCHETYPES
    if _ARCHETYPES is None:
        with open(os.path.join(os.path.dirname(__file__), "archetypes.json")) as f:
            _ARCHETYPES = json.load(f)
        for a in _ARCHETYPES:
            a["_cards"] = {int(k): v for k, v in a["cards"].items()}
    return _ARCHETYPES


class DeckIdentifier:
    def __init__(self, your_index: int | None = None):
        self._arch = _load()
        self._your_index = your_index
        self._seen: dict[int, int] = {}   # opponent card serial -> cardId (dedup physical cards)

    # ---- evidence accumulation ---------------------------------------------------------------
    def observe(self, obs) -> None:
        """Record the opponent's currently-visible cards (active/bench + attached + evolutions +
        discard). `obs` is a typed Observation (cg.api). Hand/deck/prizes are hidden, so skipped."""
        state = obs.current
        if state is None:
            return
        you = self._your_index if self._your_index is not None else state.yourIndex
        opp = state.players[1 - you]
        for poke in list(opp.active or []) + list(opp.bench or []):
            if poke is None:
                continue
            self._add(poke.serial, poke.id)
            for grp in (poke.energyCards, poke.tools, poke.preEvolution):
                for c in grp or []:
                    self._add(c.serial, c.id)
        for c in opp.discard or []:
            self._add(c.serial, c.id)

    def _add(self, serial, card_id) -> None:
        if serial is not None and card_id is not None:
            self._seen[serial] = card_id

    # ---- inference ---------------------------------------------------------------------------
    def posterior(self) -> dict[str, float]:
        seen = list(self._seen.values())
        logp = []
        for a in self._arch:
            lp = PRIOR_TEMP * math.log(max(a["meta_share"], 1e-6))   # tempered meta-share prior
            cards = a["_cards"]
            for cid in seen:
                lp += math.log(cards.get(cid, 0) + ALPHA)
            logp.append(lp)
        m = max(logp)
        exps = [math.exp(x - m) for x in logp]
        z = sum(exps)
        return {a["id"]: e / z for a, e in zip(self._arch, exps)}

    def top(self, k: int = 1) -> list[tuple[str, float]]:
        return sorted(self.posterior().items(), key=lambda kv: kv[1], reverse=True)[:k]

    def best(self) -> str:
        return self.top(1)[0][0]

    def evidence_count(self) -> int:
        return len(self._seen)

    def predicted_deck(self, archetype: str | None = None) -> list[int]:
        """The most-likely opponent's full 60-card list (card IDs, with multiplicity). This is the
        ~60-card hypothesis Sprint-6 MCTS determinizes against instead of the full pool."""
        aid = archetype or self.best()
        arch = next(a for a in self._arch if a["id"] == aid)
        ids = []
        for cid, cnt in arch["_cards"].items():
            ids.extend([cid] * cnt)
        return ids
