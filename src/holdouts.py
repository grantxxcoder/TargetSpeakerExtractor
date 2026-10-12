"""The reported holdout splits. One list, imported by every guard.

Never trained, filtered or selected on, and never used to steer training. Until
2026-10-12 three scripts each kept their own copy, and none listed ami_eval or
eval_public (decisions-m8.md 2026-10-12).

- sir0_privval, eval_private: the holdout moved from the model to the data
  (decisions-m4.md 2026-09-15).
- eval_public: already seen, so nothing may be selected or tuned on it
  (decisions-m4.md).
- ami_eval: the AMI test, scored once per system (decisions-m8.md 2026-10-12).
"""

HOLDOUT_SPLITS = frozenset({"sir0_privval", "eval_private", "eval_public", "ami_eval"})
