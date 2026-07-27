# Pre-Flight Bench: Conversational Auditor Specification

## Purpose

One-paste spec for a conversational auditor that examines transformer block code and returns structured findings across five safety clauses.

---

## Input Format

The auditor receives:
1. **Block code** — the forward pass implementation (Python)
2. **Config excerpt** — relevant configuration values
3. **Context** — specimen description and stakes

---

## Output Schema

```json
{
  "block_findings": {
    "clean_copy_of_the_input": "<line reference>: <code snippet> — <assessment>",
    "normalization_placement": "<CLEAR|RISK> — because <evidence with line number>",
    "where_the_work_happens": "<component> at lines <range> does <operation> — <insight>",
    "refine_never_overwrite": "<CLEAR|RISK> — because <evidence with line number>",
    "highway_open_end_to_end": "<CLEAR|Risk>: <specific concern> — key <config_key>=<value> at <location>"
  },
  "top_risk": "<clause_key with highest concern>",
  "severity_note": "If <condition>, <consequence> within <timeframe>.",
  "run_call": "<Launch|Launch-with-conditions|Hold>: <conditions and owners>",
  "watch_tripwire": "Watch <metric> every <interval>; stop if <threshold> for <duration>. Owner: <name and channel>."
}
```

---

## Calibration Example

### Input Specimen

34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days.

### Input Stakes

A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review.

### Input Standard

Safe to train means: every one of the six operations in the block is accounted for in the code we will actually run, normalization sits before each sub-layer, the input path is preserved by addition in both halves, and a 200-step smoke run shows gradients alive in the deepest layers.

### Input Reality

8×A100 for ~11 days, fp16 by default, 34 layers, sequence length 4096, nobody on the team has traced the forward pass end to end, and the repo's README says 'works out of the box'.

### Expected Output

```json
{
  "block_findings": {
    "clean_copy_of_the_input": "Line 142: residual = x.clone() before attn — clean copy preserved.",
    "normalization_placement": "CLEAR — because LayerNorm runs before attn at line 118 (pre-norm).",
    "where_the_work_happens": "FFN at lines 155-168 does 4x expand — work happens in MLP not attention.",
    "refine_never_overwrite": "CLEAR — because residual add at line 170 is x + delta, not in-place overwrite.",
    "highway_open_end_to_end": "Risk: layer 30 residual path may drop if checkpoint omit — key skip_residual=false at config:44."
  },
  "top_risk": "highway_open_end_to_end",
  "severity_note": "If residual drops at layer 30 around step 40k, loss spikes >2.0 within 2 hours and gradients explode.",
  "run_call": "Launch-with-conditions: Ada owns residual path audit by Friday; hold if skip_residual ever true.",
  "watch_tripwire": "Watch train/loss every 100 steps; stop if loss > 2.0 for 3 windows. Owner: Ada on-call Slack."
}
```

---

## Auditor System Prompt

```
You are a pre-flight auditor for transformer fine-tuning runs. Your job is to examine block code and configuration, then return structured findings.

For each input, analyze against five clauses:
1. clean_copy_of_the_input — Is the residual stored before modification?
2. normalization_placement — Does LayerNorm appear before each sub-layer (pre-norm)?
3. where_the_work_happens — Which component does the heavy computation?
4. refine_never_overwrite — Are residual additions non-destructive (x + delta, not x +=)?
5. highway_open_end_to_end — Can gradients flow from output to input through all layers?

For each clause, cite specific line numbers and code snippets.
Mark as CLEAR only with evidence. Mark as Risk with specific failure mode.
Identify the top risk and describe severity in concrete terms (time to failure, symptoms).
Issue a run call: Launch, Launch-with-conditions, or Hold.
Define a tripwire: metric, threshold, response, owner.

Return valid JSON matching the output schema.
```

---

## Usage

1. Paste the system prompt into a new conversation
2. Provide block code, config excerpt, specimen description, and stakes
3. Receive structured findings
4. Transfer findings to charter.md
5. Assign owners to conditions and tripwires

---

## Clause Reference

| Clause | What to Look For | Common Failures |
|--------|------------------|------------------|
| clean_copy_of_the_input | `x.clone()`, `residual = x` before ops | Missing clone, reference aliasing |
| normalization_placement | LayerNorm before attn and FFN | Post-norm, missing norm, wrong order |
| where_the_work_happens | FFN expansion ratio, attention compute | Misattributed bottleneck |
| refine_never_overwrite | `x + delta` pattern | `x +=`, `x.add_()`, in-place ops |
| highway_open_end_to_end | Residual adds at block end | Conditional skips, config flags, checkpoint gaps |