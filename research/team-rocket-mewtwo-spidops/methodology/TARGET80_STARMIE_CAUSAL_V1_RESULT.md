# Target-80 Starmie causal-action V1 result

Status: **closed; no policy construction or submission authority**  
Recorded: 2026-08-05

## Outcome

The exact official-engine panel completed cleanly but found no deployable
public rule that met the preregistered discovery threshold. The action oracle
had meaningful but heterogeneous headroom: 17 of 32 roots had a sampled
alternative above baseline, 12 tied, and three had all sampled alternatives
below baseline. Mean best-alternative gain was `+0.3802` per root:

- Froslass: 9/16 positive roots, `+0.3021` mean oracle gain;
- MistFlat: 8/16 positive roots, `+0.4583` mean oracle gain.

Those numbers describe an oracle that selects after observing continuations;
they are not achievable-policy performance.

## Integrity

Both rows passed every locked integrity check:

- 16 accepted loss games and 32 retained roots;
- four distinct semantic actions and six terminal samples per root;
- 768 scheduled official-engine descendant continuations;
- zero rollout or legality errors;
- exact packaged-opponent history-action reproduction at every branch;
- both Rocket deployment guards active;
- no hidden-completion keys serialized; and
- output hashes matched both manifests.

Artifacts:

- Froslass JSONL SHA-256:
  `874d2dad4216aadf758069719eb385cd98d1b967a6326d5d0eb81d2b214725f5`;
- Froslass manifest SHA-256:
  `fbe7512871327fafda448cfb41d4ebbf77adad3ccf4d5d1465b0e4a6dd59a5eb`;
- MistFlat JSONL SHA-256:
  `d9dd6d8f4dcd771b2000b5f777ea9f7d39a2848d4fab0bfe26c99256c109aa58`;
- MistFlat manifest SHA-256:
  `66bf31ec075562f94deff932102ae4d9afcc868f2d39d7418ba03a47b4231578`.

## Discovery-gate audit

The strongest repeated exact base-to-alternative action pattern appeared only
four times and averaged negative. The best cross-opponent exact pair appeared
three times and averaged only `+0.2222`, below both the six-root support and
`+0.50` gain gates.

As an intentionally optimistic check, the opened panel was enumerated across:

- fixed exact alternative actions;
- fixed alternative action types;
- baseline-type to alternative-type mappings;
- exact-baseline to alternative-type mappings;
- baseline-type to exact-alternative mappings; and
- exact base-to-alternative pairs.

For each condition, the best matching candidate at every root was allowed,
which favors finding a rule. The enumeration was repeated pooled and within
each exactly identifiable Starmie family. **Zero conditions** met the locked
minimum of six roots, three games, `+0.50` mean, at least two-thirds positive,
and at most one-sixth negative.

The large individual gains were strategically inconsistent: examples included
retreating Articuno instead of Ariana, Ariana instead of Proton or
Transceiver, attaching before evolving, Giovanni instead of another Mewtwo
attachment, ending instead of Bug Catching Set, and several other one-game
choices. Turning these into one-off guards would be development-set mining.

## Decision

Close exact Starmie causal-action V1. Do not package, submit, or weaken its
thresholds. It confirms that terminal-reward PPO was hiding action-level
headroom, but 32 loss roots do not support a stable hand-authored correction.
Any later learned selector needs a substantially larger game-disjoint corpus
and independent confirmation. V5 and V6 remain immutable, and no Kaggle action
is authorized.
