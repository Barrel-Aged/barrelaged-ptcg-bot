# Target-80 public-value V2 serialization amendment

Frozen: 2026-08-05, after the first training run and before rerun

The first frozen training run cleared every statistical and calibration gate
but reported `serialization_reload=false`. Inspection of the verifier found a
mechanical comparison error: the reference probabilities were produced in one
large matrix batch, while reload probabilities were produced one row at a
time. CPU matrix multiplication may differ by more than `1e-7` across those
two shapes even when every serialized tensor is identical.

This amendment changes only the reload verifier. It will:

1. require `torch.equal` for every original-versus-reloaded state tensor,
   mean tensor, and standard-deviation tensor; and
2. compare original and reloaded predictions using the same one-row tensor
   shape for both models, requiring exact float equality.

The source data and digest, split, examples, features, seed, architecture,
optimizer, learning rate, weight decay, batch size, epoch count, terminal-only
artifact, and every statistical admission threshold remain unchanged. The
same frozen command is rerun from epoch zero. Any statistical result remains
subject to the original gates; this correction cannot rescue a failed AUC,
Brier, opponent-family, or calibration result.
