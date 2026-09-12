# Fixture papers for the continuous-improvement gate

Drop paper Markdown files here to serve as the held-out and regression sets
that `improve/eval_gate.py` runs before/after every proposed prompt change:

- `heldout/*.md` — papers NOT used when diagnosing a failure or drafting a
  fix; they measure whether a change generalizes rather than just patching
  the specific run that triggered it.
- `regression/*.md` — papers the pipeline is already known to handle
  reasonably well; they catch a "fix" that improves one failure mode while
  breaking something else.

Aim for 3-5 short, diverse papers in each (different architectures/domains —
e.g. a CNN classifier, a transformer, a GAN, an RL paper) so the gate isn't
just noise from a single sample. Markdown only, same format `main.py` takes
as input (see `test2/part.md` for the existing sample).

Nothing in this directory is fabricated by the pipeline itself — it's the
fixed ground truth the improvement loop is graded against, so it should only
ever be added to by a human.
