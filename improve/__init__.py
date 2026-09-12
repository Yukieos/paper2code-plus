"""Continuous agent-improvement pipeline.

Mines recent failures (eval/mine_failures.py), diagnoses a root cause for
the most common failure mode, proposes a scoped prompt-only change,
validates it against a held-out + regression fixture set, and — only if
that gate passes — opens a GitHub PR for human review. Nothing here ever
merges its own change; `git push` + `gh pr create` is as far as automation
goes. See eval/fixtures/README.md for how to populate the fixture papers
this depends on, and .github/workflows/continuous_improvement.yml for the
manually-triggered CI entry point.
"""
