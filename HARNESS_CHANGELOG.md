# HARNESS_CHANGELOG

Append-only log of every behavior-controlling change proposed by the continuous-improvement loop (`improve/run_cycle.py`). Newest entries are appended at the bottom. Do not edit or remove past entries — reverse a change by appending a new entry that names the superseded commit.

## 2026-09-15T21:33:01+00:00 — lint_violation — 1a031aa1430b78b677c25f689000032b58a17f09

- **Triggering runs:** seed-cnn_classifier-20260910-000000, seed-simple_gan-20260911-000000, seed-transformer_encoder-20260912-000000
- **Failure mode / metric:** lint_violation
- **Reproduced:** 2/2 (threshold 2)
- **Observed behavior:** The extractor agent is likely not effectively determining which libraries are actually needed based on the context of the generated code. This could be due to a lack of specificity in the prompt regarding the importance of minimizing unused imports, leading to the inclusion of unnecessary imports that are flagged by flake8.
- **Smoking gun / evidence:** 'torch' imported but unused
- **Hypotheses considered:**
  - The code generation process is not properly analyzing the context in which imports are used, leading to unnecessary imports being included.
  - The prompt given to the code generation agent does not specify the need to avoid unused imports, resulting in code that does not adhere to best practices.
- **Change(s):**
  - RETRY_PROMPT_TEMPLATE in agents/extractor.py — The prompt was revised to explicitly instruct the extractor agent to include only necessary libraries and avoid unnecessary imports, addressing the root cause of the issue with flake8 warnings.
- **Files changed:**
  - agents/extractor.py
- **Targeted verification:** targeted lint_violation on seed-cnn_classifier-20260910-000000: still present in 0/1 post-fix run(s) -> FIXED
- **Held-out gate:** 0/3 passed
- **Regression gate:** 0/3 passed
- **Cost / token delta:** grader signals 9 -> 7 (token/$ not tracked)
- **Risks:** Prompt-only change; behavior on papers unlike the fixtures is unverified.
- **Commit:** 1a031aa1430b78b677c25f689000032b58a17f09
- **Supersedes:** none
