# N9 Training Execution Lock

Status: **frozen after exact corpus audit and before optimization**  
Protocol: `rocket-balanced-exact-pcgrad-awr/v1`

## Frozen evidence and code

- Protocol SHA-256:
  `157dc3d168d7b868b9d40f6aa008ab5303a1cd849c9f8a88d32af18f7dc38dbd`.
- Combined replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.
- Exact parity report SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- Parity auditor SHA-256:
  `ab74a1d135d380355f0b59d394addf4deba744ca75363fdc756a95c3c672b958`.
- Trainer SHA-256:
  `62182583002847e7ed0e18011050f35319ad5661b73f3fb2d83c2564286ed1ad`.
- PCGrad tests SHA-256:
  `cbfcf21383469d7afe3ffe45a27ee5baaa2b3c01a634ccaac1c0e2fb08566a90`.

Shard-manifest SHA-256 values:

- V110 seed 797101:
  `4342ad8f3610bead2106a9706164e70e2e2694be3b0017eb33446128aa0df46e`;
- V110 seed 797102:
  `ed440e650d46544be80c1ae0dfc77b515f18ac46fae54dbf5dcbec99cceb9708`;
- Recovery seed 797201:
  `f7828adff23764bc46b60b01bb330b85eb972416be44cd3b6508ece2f1b829d9`;
- Recovery seed 797202:
  `76bea2dd3ab52b2a6a90b787d7c6f00a1947d79520585e3806ce958d669ff459`.

## Corpus audit

The corpus contains 1,200 complete official-engine games and 72,499 focal
decisions. One draw remains immutable but is excluded from binary training.
All 72,499 decisions passed: 69,205 nontrivial autoregressive action steps
matched the frozen model exactly, 5,628 trivial prompts were canonical, and
1,695 forced full-set prompts selected the complete engine-equivalent set.
There were zero mismatches and zero excluded games.

## Tested optimizer invariants

Three deterministic tests passed:

- aligned gradients retain their mean;
- conflicting gradients project to a direction non-adverse to either original
  task gradient; and
- outcome weights normalize to equal V110/Recovery mass.

The frozen Transformer remains in deployment-time evaluation mode; only the
two logit-head tensors require gradients.

## Locked command

```powershell
python scripts/train_n9_balanced_pcgrad_awr.py --parity-report artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/exact_agent_parity.json
```

All four epoch files are mandatory development arms. Offline metrics cannot
select or discard an epoch, and training output has no deployment authority.
