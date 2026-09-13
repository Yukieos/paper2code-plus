# Paper2Code

A multi-agent coding system that converts ML papers into executable PyTorch training code through agent planning, tool use, code generation, and end-to-end execution.

A lightweight multi-agent system that:
- **Stage 1 – Paper → UPS-IR**: turns a paper in Markdown into a rich, structured UPS-IR JSON and per-section artifacts.
- **Stage 2 – UPS-IR → Code**: turns `UPS-IR.json` into a production-ready repository via a ReasonFlow-style, multi-agent code generator (`codegen_pipeline.py`).  
  See `architecture.md` for the high-level ReasonFlow architecture.
- **Stage 3 – Iterative Code Debugging**: re-run codegen with different flags/models, inspect `generated_repo/` + `output/static_check_results.json`, and manually patch or regenerate specific files until tests and static checks are clean. There are multiple checker at each stage of generation, and will organize all the failures in code to generate better ones.

Recent UPS-IR (extraction) upgrades:
- Rich UPS-IR schema now captures sections, figures, algorithms, parameters, losses, training pipeline steps, and optimization settings with `source_reference` + `source_text` traceability.
- Extractor prompt explicitly demands model structure, submodules, hyperparameters, and experiment provenance rather than returning a minimal skeleton.
- Structurer + Verifier normalize and validate the expanded schema so downstream planners/codegen receive faithful paper-level detail.
- Synthesizer automatically emits JSON shards for every top-level UPS-IR key, including the new research-awareness fields.

## Quick Start

### Option A: Web UI (recommended for trying it out)
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (or enter the key directly in the sidebar at runtime — it's kept in-session only, never written to disk).
3. `streamlit run app.py`, then open the local URL Streamlit prints.
4. Paste or upload a paper as Markdown (or upload a PDF for best-effort text extraction), click **Run pipeline**, watch Stage 1/Stage 2 logs stream live, and download the generated repo as a zip.

Each UI run is isolated under `runs/<timestamp>/` so it never touches the sample `UPS-IR.json` / `generated_repo/` checked into this repo.

### Sharing the UI with other people
`streamlit run app.py` only binds to your machine — nobody else can reach it. To give other people a real link:

1. Push this repo to GitHub (already done if you're reading this from the remote).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, **New app** → pick this repo/branch → main file `app.py`. Streamlit Cloud can deploy private repos once you grant its GitHub App access.
3. In **Advanced settings → Secrets**, paste from `.streamlit/secrets.toml.example`. **Leave `OPENAI_API_KEY` out unless you want to personally pay for every visitor's usage** — by default each visitor pastes their own key in the sidebar; only set `OPENAI_API_KEY` + `PAPER2CODE_ALLOW_SHARED_KEY=1` for a small trusted audience you're OK covering.
4. Free-tier Streamlit Cloud is ~1 CPU/1GB RAM and runs one instance — fine for a few people trying it sequentially, not for heavy concurrent use.

### Option B: CLI
- **Install deps**: `pip install -r requirements.txt`
- **Stage 1 – run UPS-IR agent pipeline**: `python main.py test2/part.md`
  - Faster (skip image captioning): `python main.py test2/part.md --skip-annotation`
  - Specify agents: `python main.py test2/part.md --agents reader,extractor,structurer`
  - LangGraph mode: `python main.py test2/part.md --use-graph`
- **Stage 2 – run code generator on UPS-IR**: `python codegen_pipeline.py UPS-IR.json --output-dir generated_repo`

## What It Does
- **Reader**: load Markdown; optional image descriptions via `agents/picture.py` (needs `ARK_API_KEY`).
- **Extractor**: LLM extracts UPS-IR. Prompt enforces using entity ids in references and now captures sections/figures/algorithms/parameters/losses/training/optimization details with citations.
- **Structurer**: normalize types, ids, metrics, contextual arrays; writes `UPS-IR.json`.
- **Verifier**: schema + cross-reference check (accepts ids and source references across all entities).
- **Synthesizer**: split UPS-IR into `UPS-IR_Output/<section>.json`.
- **Codegen (multi-agent)**: `codegen_pipeline.py` classifies the paper type, plans a repository, generates code via specialized agents (Dataset/Execution/Evaluation), then runs integration + static checks to produce `generated_repo/`.

## Evaluation & Failure-Mining Framework (`eval/`)

Every agent step *can* be instrumented, shipped to S3 as a structured trace, and later mined for recurring failure patterns across a fixed 18-mode taxonomy (`eval/taxonomy.py`) spanning tool use, reasoning/planning, and generated code.

**It's off by default** so plain CLI/UI usage never needs AWS credentials. To turn it on:

1. Set `PAPER2CODE_TRACE_ENABLED=1` in `.env` (and the usual `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION`, or rely on `~/.aws/credentials` / an instance role instead).
2. Run the pipeline as usual (CLI or UI). Each `agent.run(state)` call in `main.py` / `codegen_pipeline.py` is wrapped with `eval.instrumentation.run_traced(...)`, which uploads one JSON trace per agent step to `s3://<PAPER2CODE_TRACE_BUCKET>/traces/<run_id>/...` (falls back to `runs/_traces_local/` if S3 is unreachable — tracing failures never break the actual run).
3. Mine what's accumulated:
   ```bash
   python -m eval.mine_failures                # every run in the bucket
   python -m eval.mine_failures --run-ids <id>  # just one run
   python -m eval.mine_failures --no-judge      # skip the LLM classification, grader signals only
   python -m eval.mine_failures --cluster       # also embed+cluster failures (see below)
   ```

**How a run gets classified:**
- `eval/graders.py` — deterministic signals: `TraceGrader` reads the trace itself (a rejected verification, a malformed-JSON retry loop, a timeout); `CodeGrader` runs flake8/mypy/syntax/import-resolution/TODO-detection over `generated_repo/`.
- `eval/judge.py` — an LLM-as-judge pass for what graders can't pin down mechanically (e.g. "did the extractor actually understand the paper"), given the taxonomy rubric + grader signals + a condensed trace.
- `eval/mine_failures.py` — aggregates both into a JSON report (`output/failure_report_<timestamp>.json`) with per-run classifications and a taxonomy-wide failure-mode histogram.

**Clustering (`eval/cluster_failures.py`) — finding what the taxonomy can't distinguish:**
the 18-mode taxonomy answers "which known category is this", which flattens
real variety: 12 `VERIFICATION_REJECTED` failures across different runs
might be 12 unrelated causes, or one root cause worded 12 different ways
(this is literally how the relation key-drift bug was actually found — by
manually reading output, which clustering exists specifically to avoid
needing to do again). `--cluster` embeds every grader signal + judge
rationale (`OpenAIEmbeddings`), groups them with DBSCAN over cosine
distance (no cluster count to pick up front; a truly one-off failure stays
its own singleton rather than getting forced into the nearest group), and
asks an LLM for a one-line root-cause label per cluster of 2+. Output
lands in the report's `failure_clusters` list, largest first. This is a
separate, additive lens on the same underlying signals — it doesn't
replace the fixed taxonomy (which stays the stable, interpretable
baseline `improve/diagnose.py` targets), it surfaces what's actually
recurring underneath it.

## Continuous Agent Improvement (`improve/`)

Takes the mining above one step further: periodically diagnoses *why* the most common failure mode is happening, drafts a fix, and — only if it survives a held-out + regression gate — opens a PR for human review. It never merges its own change.

```
mine failures → pick most common mode → diagnose root cause → propose a
prompt-only fix → run held-out+regression gate before vs. after → if no
regressions: push branch + open PR (human reviews & merges) → else: discard
```

**Scope is deliberately narrow**, by design decision, not by accident:
- Only prompt *wording* can be changed — `improve/prompt_registry.py` is an explicit allowlist of every prompt actually wired into the live pipeline, and refuses any edit that changes the `{placeholder}` variables a prompt depends on (that would break the call site, not just reword it).
- The gate is conservative: a proposed change is rejected if it newly breaks any regression fixture, or increases the total grader-signal count across held-out fixtures. Anything more nuanced is left for the human reading the PR.
- It's manually triggered (`.github/workflows/continuous_improvement.yml`, `workflow_dispatch` only) — every run spends real OpenAI (and mining, S3) usage, so it's opt-in per cycle rather than on a schedule.

**Before running it**, populate `eval/fixtures/heldout/` and `eval/fixtures/regression/` with a handful of paper Markdown files (see `eval/fixtures/README.md`) — without those, there's nothing to gate against and `improve.run_cycle` refuses to run.

```bash
# locally, after collecting some traces (PAPER2CODE_TRACE_ENABLED=1):
python -m improve.run_cycle --dry-run              # diagnose + propose only, no git/gh calls
python -m improve.run_cycle                         # full cycle: gate + push branch + open PR
python -m improve.run_cycle --failure-mode syntax_error
```

Or trigger the `Continuous Agent Improvement` workflow from the GitHub Actions tab. It needs these repo secrets: `OPENAI_API_KEY` (required), `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` / `PAPER2CODE_TRACE_BUCKET` (only if traces live in S3 rather than being mined locally).

## Extended UPS-IR Schema (Key Fields)
- `sections`: `{id, title, level, summary, key_points[], source_reference, source_text}`
- `figures`: `{id, name, description, related_components[], source_reference}`
- `algorithms`: `{id, name, inputs[], outputs[], steps[], source_reference}`
- `parameters`: `{id, name, symbol, value, description, constraints, source_reference}`
- `losses`: `{id, name, formula, description, related_methods[], source_reference}`
- `training_pipeline`: `{id, name, description, inputs[], outputs[], hyperparameters[], source_reference}`
- `optimizations`: `{id, name, objective, optimizer, schedule, source_reference}`
- Existing `tasks/methods/datasets/equations/experiments/relations` all gain optional `description` + referencing guidance.

Each object should cite `source_reference` (e.g., “Sec. 3.2”, “Fig. 4”) and optionally `source_text` excerpts so later agents can quote the original paper verbatim.

## Useful Paths
- Input sample: `test2/part.md`
- Intermediates: `output/` (annotated md, extracted_info.json)
- Final: `UPS-IR.json`, `UPS-IR_Output/`
- Logs: `logs/agent_pipeline.log`

## JSON → Code Pipeline (Enhanced with Multi-Stage Prompting)
- Once `UPS-IR.json` exists, run `python codegen_pipeline.py UPS-IR.json --output-dir generated_repo`.
- The enhanced pipeline now includes:
  
### Core Features

#### Context-Aware Parameter Tracking
- **Problem Solved**: Parameters with the same name (e.g., `learning_rate`, `hidden_dim`) in different modules are now tracked separately with context
- **Implementation**: Uses `parameter_name@file_path` keys to differentiate context-specific parameters
- **Benefits**: Each module can have different suggested values and risk assessments for the same parameter name

#### Robust JSON/Code Parsing  
- **Multi-Stage Error Recovery**: 
  1. Basic fixes (trailing commas, unescaped quotes)
  2. Aggressive extraction (regex-based JSON detection)
  3. LLM-based correction as fallback
- **Automatic Retry Logic**: Up to 3 attempts with progressive error correction
- **Fallback Mechanisms**: Generates minimal valid structures if all parsing attempts fail

### Enhanced Features
- **Multi-Stage Prompting (MTI)**: 
  - Plan generation → Critical review → Revision cycle
  - Code generation → Review against acceptance criteria → Automatic revision
  - Intermediate drafts saved in `.intermediate/` folder for comparison
  
- **Structured Validation**: 
  - Pydantic models for plan/file specifications with auto-fixing
  - JSON schema validation with automatic field completion
  - Function-level skeletons with signatures, docstrings, and step-by-step logic
  
- **Transparent Parameter Management**:
  - `ParameterPlaceholderAgent`: Lists every parameter still needing a user-specified value
  - `ConfigGeneratorAgent`: Emits config templates with null placeholders + risk notes
  - Parameters categorized by risk level (low/medium/high) so users can prioritize fills
  
- **Code Quality Assurance**:
  - Static checking with flake8, mypy, and syntax validation
  - Code review against acceptance criteria before finalization
  - Negative prompting to prevent common anti-patterns (no TODOs, no empty functions)

### Pipeline Outputs
- `output/code_plan.json`: Initial repository blueprint with function skeletons
- `output/revised_code_plan.json`: Improved plan after critique cycle
- `output/missing_parameters.json`: Registry of undefined parameters with risk levels
- `output/parameters_to_fill.json`: User-facing checklist of parameters that still require manual values
- `output/config_template.json`: Ready-to-use configuration file template
- `output/static_check_results.json`: Linting and type-checking results
- `generated_repo/`: Production-ready source files (no TODOs, complete implementations)
- `generated_repo/.intermediate/`: Version history of each generated file (v1, v2)

### Command-Line Flags
- `--plan-only`: Generate plan and parameter analysis without writing code
- `--skip-review`: Skip the plan review/revision stage (faster but less thorough)
- `--skip-static-check`: Skip static analysis of generated code
- `--skip-missing-params`: Skip parameter analysis and auto-filling
- `--no-intermediate`: Don't save intermediate code drafts
- `--limit-files N`: Only generate first N files (useful for testing)
- `--model`: LLM model name (default: gpt-4o-mini)
- `--temperature`: Planning temperature (default: 0.0)
- `--writer-temperature`: Code generation temperature (default: 0.2)

### Example Usage
```bash
# Full pipeline with all enhancements
python codegen_pipeline.py UPS-IR.json --output-dir generated_repo

# Quick iteration without reviews
python codegen_pipeline.py UPS-IR.json --skip-review --limit-files 3

# Plan-only mode for analysis
python codegen_pipeline.py UPS-IR.json --plan-only
```

The enhanced pipeline ensures generated code is production-ready with proper error handling, complete implementations, and intelligent parameter defaults based on ML best practices.

## Troubleshooting
- Missing md_path: run `python main.py test2/part.md`.
- JSON escape errors: extractor auto-fixes stray backslashes in LaTeX.
- Reference errors: ensure references use ids from their lists.
