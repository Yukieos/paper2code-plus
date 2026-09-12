"""Agent evaluation and failure-mining framework for the Paper2Code pipeline.

Instruments the Stage 1 (paper -> UPS-IR) and Stage 2 (UPS-IR -> code) agent
runs, ships execution traces to S3, and grades/classifies failures against a
fixed taxonomy so recurring breakdowns in tool use, reasoning, and generated
code can be surfaced and tracked over time.
"""
