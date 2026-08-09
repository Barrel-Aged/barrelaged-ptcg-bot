# Phase 6M native external-sampling wiring probe

Phase 6M connects the Phase 6L traversal, the Phase 6K behavior checkpoint, the
Phase 6J branchable native-search game, the Phase 6I-compatible feature bridge,
and bounded advantage/strategy reservoirs.

The completed probe is intentionally narrower than a training run. Full-game
external sampling expands every future action of the traverser and quickly
exhausts practical transition budgets in this game. Phase 6M therefore admits
only policy-spine states where every legal action transitions immediately to a
terminal native state. Scenarios without such a root are skipped; no partial
traversal becomes supervision.

## Official execution versus abstraction

The following is genuine official-engine execution:

- every accepted endgame action branch is passed to the licensed native
  `search_step` implementation;
- legality comes from the native selection contract; and
- all 45 external-sampling transitions terminate through native results.

The following remains a research abstraction:

- the equal-weight hidden-completion prior and each fixed completion selected
  before an endgame root;
- the balanced and player-0-weakened policy spines used to locate roots;
- the Phase 6K checkpoint used only as a behavior strategy;
- semantic public/private/action features and hash buckets;
- the MCCFR interpretation and bounded reservoirs; and
- every extrapolation beyond the immediate terminal subgames.

The selected completion is never a policy feature. Serialized samples contain
111 public features, 75 acting-player-private features, and 11 features per
legal action. They retain only hashes/fingerprints for private observations;
the recursive forbidden-key audit reports zero raw private zone fields.

## Live evidence

The unseeded native battle in this checked-in run exposed five admissible
closing scenarios across both players. Seven other balanced/weakened scenarios
were not admitted because they lacked an immediate-terminal root.

| Check | Result |
|---|---:|
| Admitted closing scenarios | 5 |
| Player-0 / player-1 traversals | 2 / 3 |
| Complete external-sampling transitions | 45 |
| Total native transitions including spine/root checks | 9,875 |
| Advantage samples seen / retained | 5 / 5 |
| Strategy samples seen / retained | 0 / 0 |
| Nonzero advantage actions | 0 |
| Maximum absolute advantage | 0.0 |
| Raw private card IDs persisted | 0 |
| Dataset fingerprint | `54ff4e2eb1ea8a7d8afc4e03c08f63361c7adae56e7e12af669347290490c3a4` |
| Evidence fingerprint | `21d903ed43791ae24644924b4e32e8787f01338933f5af09a18a1bb2cf0bba5c` |

The zero advantages are expected: all actions at these closing selection
prompts produce the same terminal utility. Likewise, no opponent node is
encountered after an immediate-terminal root, so the native strategy reservoir
remains empty. This artifact validates interfaces, exact legal branching,
feature shapes, redaction, provenance, fail-closed budgets, and serialization;
it is not useful evidence that the current policy improved.

The source Phase 6K checkpoint was produced from a different unseeded live game
fingerprint. That mismatch is stored explicitly. Its use here is behavior-only
and cannot establish generalization.

## Durable artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `native_external_sampling_dataset.json` | 175,054 | `73cc199b3f49a021dccfbaa13173a64f658bd1f321dfc5f497d7a449aa7dc07a` |
| `phase6m_evidence.json` | 2,135 | `147ddfb68ff405fccf0637bee75d88342777578f8774c6edd6b05344a4ac41c3` |
| `phase6m_evidence.md` | 2,176 | `f00ee0fa690bf1c9e24c435ba3e2e4cc77fb584d5ce5ca9c64c6dbe056ba18fc` |

Native artifacts are live and are not claimed byte-deterministic across fresh
battle starts.

## Next gate

Phase 6N has since supplied the primary-source-correct scalable estimator, and
Phase 6O has supplied nontrivial both-player memories plus separate neural
regret and average-plan models. Those are documented in
`PHASE6N_OUTCOME_SAMPLING_MCCFR.md` and
`PHASE6O_NATIVE_TURN_PLAN_BLUEPRINT.md`.

Before calling the native neural strategy an equilibrium blueprint, the
project still needs:

1. stable, substantially larger iteration-indexed native memories without an
   estimator tail that dominates the neural objective;
2. a defensible held-out regret/exploitability proxy across unseen particles;
3. calibrated, history-conditioned public beliefs; and
4. full-game counterfactual bounds before safe resolving or exploitation.

No Kaggle submission or performance claim follows from Phase 6M.
