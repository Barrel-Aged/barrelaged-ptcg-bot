from __future__ import annotations

import unittest

from scripts.build_n14_zero_residual_base import patched_source


MINIMAL = '''class RelationalNet(nn.Module):
    def __init__(self, d_card=64, d_atk=16, h=256, tlayers=3, heads=4,
                 arch="v10", aux=True, feats="v3", mem=0, value_detach=True,
                 aux_dim=F.PRIV_DIM, **_ignored):
        self.cfg = dict(d_card=d_card, d_atk=d_atk, h=h, tlayers=tlayers,
                        heads=heads, arch=arch, aux=aux, feats=feats, mem=mem,
                        value_detach=value_detach, aux_dim=aux_dim)
        self.logit_head = nn.Linear(h + mem, 1) if mem else nn.Linear(h, 1)
        else:
            m = None
            logits = self.logit_head(otok).squeeze(-1).squeeze(0)         # [O]
        if "opt_avail" in feat:
        else:
            m = None
            logits = self.logit_head(otok).squeeze(-1)                    # [B,O]
        logits = logits.masked_fill(~b["opt_mask"].bool(), float("-inf"))
'''


class N14PatchTests(unittest.TestCase):
    def test_patch_adds_zero_residual_architecture_once(self) -> None:
        result = patched_source(MINIMAL)
        self.assertIn("adapter_rank=0", result)
        self.assertIn("nn.init.zeros_(self.logit_adapter[-1].weight)", result)
        self.assertEqual(result.count("logits = logits + self.adapter_scale"), 2)
        self.assertEqual(result.count("self.logit_adapter = ("), 1)

    def test_patch_fails_closed_when_anchor_changes(self) -> None:
        with self.assertRaises(ValueError):
            patched_source(MINIMAL.replace("aux_dim=F.PRIV_DIM", "aux_dim=12"))


if __name__ == "__main__":
    unittest.main()

