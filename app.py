"""Streamlit entry point: upload a paper, watch Paper2Code turn it into a
runnable PyTorch training repo.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

STAGE1_MARKERS = [
    "Running pipeline",
    "Pipeline execution completed",
]
STAGE2_MARKERS = [
    "Stage 0:",
    "Stage 1:",
    "Stage 2:",
    "Stage 3:",
    "Stage 4:",
    "Stage 4.5:",
    "Stage 4.6:",
    "Stage 5:",
    "Stage 5a:",
    "Stage 5b:",
    "Stage 5c:",
    "Stage 6:",
    "Stage 7:",
    "Stage 8:",
    "Enhanced codegen pipeline completed",
]


def new_job_dir() -> Path:
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def pdf_to_markdown(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        parts.append(f"## Page {i + 1}\n\n{text}")
    return "\n\n".join(parts)


def zip_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src))
    return buf.getvalue()


def stream_subprocess(cmd, cwd, env, log_placeholder, progress_placeholder, markers) -> tuple[int, str]:
    process = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: list[str] = []
    seen = set()
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        log_placeholder.code("\n".join(lines[-300:]), language="log")
        for idx, marker in enumerate(markers):
            if marker in line and marker not in seen:
                seen.add(marker)
                progress_placeholder.progress(min(1.0, (idx + 1) / len(markers)))
    process.wait()
    return process.returncode, "\n".join(lines)


st.set_page_config(page_title="Paper2Code", page_icon="🧪", layout="wide")

# When this app is deployed somewhere public (e.g. Streamlit Community Cloud),
# a host-configured OPENAI_API_KEY must NOT be silently handed to every visitor
# — that would let strangers spend the host's money. It's only pre-filled when
# the host explicitly opts in (for their own private/local use).
ALLOW_SHARED_KEY = os.environ.get("PAPER2CODE_ALLOW_SHARED_KEY", "").lower() in ("1", "true", "yes")

st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "OpenAI API key",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", "") if ALLOW_SHARED_KEY else "",
    help="Bring your own key — used only for this session's requests, never written to disk.",
)
base_url = st.sidebar.text_input(
    "OpenAI base URL", value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)
skip_annotation = st.sidebar.checkbox(
    "Skip image captioning", value=True,
    help="Image captioning needs a separate ARK_API_KEY; leave this on unless you have one set.",
)

st.title("🧪 Paper2Code")
st.caption(
    "Multi-agent pipeline that turns an ML paper into a structured spec (UPS-IR) "
    "and then into a runnable PyTorch training repo."
)

tab_run, tab_history = st.tabs(["Run a paper", "Past runs"])

with tab_run:
    input_mode = st.radio(
        "Paper input", ["Paste Markdown", "Upload Markdown", "Upload PDF (experimental)"],
        horizontal=True,
    )
    md_text = None
    if input_mode == "Paste Markdown":
        md_text = st.text_area("Paste the paper as Markdown", height=280, placeholder="# Title\n\n## Abstract\n...")
    elif input_mode == "Upload Markdown":
        uploaded = st.file_uploader("Markdown file", type=["md"])
        if uploaded:
            md_text = uploaded.read().decode("utf-8")
    else:
        uploaded = st.file_uploader("PDF file", type=["pdf"])
        if uploaded:
            st.info(
                "This does a basic text extraction only (layout, tables, and figures are lost). "
                "For higher-fidelity extraction, convert the PDF to Markdown yourself first "
                "(e.g. with MinerU) and use the Markdown option instead."
            )
            md_text = pdf_to_markdown(uploaded.read())

    run_codegen = st.checkbox("Also generate code (Stage 2: UPS-IR → PyTorch repo)", value=True)

    run_clicked = st.button("🚀 Run pipeline", type="primary", disabled=not md_text)

    if run_clicked:
        if not api_key:
            st.error("Please provide an OpenAI API key in the sidebar.")
            st.stop()

        job_dir = new_job_dir()
        (job_dir / "input.md").write_text(md_text, encoding="utf-8")

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_BASE"] = base_url
        env["PYTHONUNBUFFERED"] = "1"

        st.subheader("Stage 1 — Paper → UPS-IR")
        progress1 = st.progress(0.0)
        log1 = st.empty()
        cmd1 = [sys.executable, str(REPO_ROOT / "main.py"), "input.md", "--model", model]
        if skip_annotation:
            cmd1.append("--skip-annotation")
        code1, _ = stream_subprocess(cmd1, job_dir, env, log1, progress1, STAGE1_MARKERS)

        if code1 != 0:
            st.error("Stage 1 failed — see log above.")
            st.stop()
        st.success("Stage 1 complete.")

        ups_ir_path = job_dir / "UPS-IR.json"
        if ups_ir_path.exists():
            with st.expander("View UPS-IR.json"):
                st.json(json.loads(ups_ir_path.read_text(encoding="utf-8")))

        if run_codegen:
            st.subheader("Stage 2 — UPS-IR → PyTorch repo")
            progress2 = st.progress(0.0)
            log2 = st.empty()
            cmd2 = [sys.executable, str(REPO_ROOT / "codegen_pipeline.py"), "UPS-IR.json", "--model", model]
            code2, _ = stream_subprocess(cmd2, job_dir, env, log2, progress2, STAGE2_MARKERS)

            if code2 != 0:
                st.error("Stage 2 failed — see log above.")
            else:
                st.success("Stage 2 complete.")
                repo_dir = job_dir / "generated_repo"
                if repo_dir.exists() and any(repo_dir.iterdir()):
                    st.download_button(
                        "⬇️ Download generated_repo.zip",
                        data=zip_dir(repo_dir),
                        file_name=f"{job_dir.name}_generated_repo.zip",
                        mime="application/zip",
                    )
                    with st.expander("Generated files"):
                        for p in sorted(repo_dir.rglob("*")):
                            if p.is_file():
                                st.text(str(p.relative_to(repo_dir)))

with tab_history:
    jobs = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    if not jobs:
        st.info("No runs yet — go to the 'Run a paper' tab to start one.")
    for job in jobs:
        with st.expander(job.name):
            repo_dir = job / "generated_repo"
            ups_ir = job / "UPS-IR.json"
            if ups_ir.exists():
                st.caption("UPS-IR.json available")
            if repo_dir.exists() and any(repo_dir.iterdir()):
                st.download_button(
                    f"Download {job.name}_generated_repo.zip",
                    data=zip_dir(repo_dir),
                    file_name=f"{job.name}_generated_repo.zip",
                    key=f"dl-{job.name}",
                )
            else:
                st.caption("No generated code for this run (Stage 2 not run, or it failed).")
