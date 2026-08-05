# N12 Cached Execution Amendment

Status: **preregistered after computational invalidation and before any N12
optimization result or checkpoint was observed**  
Protocol: `rocket-n12-counterfactual-margin-adapter/v1`  
Amendment: `rocket-n12-cached-final-block-execution/v2`  
Frozen: 2026-08-01

## Why an execution amendment is required

The first v1 command completed all data/reference preflight checks and created
an empty output directory, but its first optimization step did not finish after
more than eight continuous CPU-minutes. It emitted no loss, correction-fit
count, checkpoint, model, or report. The process was terminated before any
model result could influence this amendment. The empty directory is preserved
at:

`artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v1_infeasible_empty`.

At that measured cost, the preregistered 400-step run would require days and
would be unusually vulnerable to another power interruption. The scientific
objective, examples, logical batches, training schedule, thresholds, and gates
remain unchanged. Only redundant execution of frozen computation is removed.

## Exact frozen-prefix cache

Only the final Transformer block and action head are trainable. For every N6R
and N9 example, v2 runs the frozen reference network once and captures the
exact input tensor to that final block. Masked padding tokens are removed;
there is no positional encoding, and masked tokens cannot participate in
attention. Each cache record retains the global token, every real Pokemon
token, every real option token, the option boundary, autoregressive
availability, and the deployed attack/END guard.

All later candidate steps execute only the trainable final block and action
head from those cached inputs. Before optimization, the cached path must match
the full frozen path on every top action and have maximum finite relative-logit
error no greater than `2e-5` across all 7,392 examples. Failure aborts the run.

## Memory-bounded but objective-equivalent gradients

Each 128-example N6R and N9 logical preservation batch is evaluated as four
32-example microbatches. Every microbatch loss is multiplied by its exact share
of the 128-example mean and backpropagated before the next microbatch; the
optimizer steps only after the correction loss, both complete logical anchor
means, and displacement penalty have accumulated. Thus the fixed objective and
example order are unchanged while intermediate graphs can be freed.

The fixed host thread count becomes four. Transformer dropout remains zero,
the seed remains `812101`, and all other v1 hyperparameters—including 400
steps, AdamW learning rate `5e-7`, gradient cap `0.25`, threshold semantics,
and admission gates—remain unchanged. A deterministic gradient-equivalence
unit test is mandatory before the cached run is locked.

## Output and authority

The replacement output is:

`artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v2_cached`.

The v1 empty directory has no training, selection, strength, or deployment
authority. The cached v2 output retains exactly the limited authority stated
in the original protocol: passing checkpoints may enter only the locked
official-engine development panel. The live bot remains unchanged.

