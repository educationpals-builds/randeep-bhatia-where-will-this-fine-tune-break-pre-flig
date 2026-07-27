# METHOD: The BLOCK Framework

## Framework Name

**BLOCK** — **B**ase-path **L**iveness **O**perations **C**heck for **K**ept gradients

---

## Purpose

BLOCK is a pre-flight audit framework for transformer fine-tuning. It systematically verifies that the residual stream—the highway gradients travel—remains open and healthy throughout training.

---

## The Five Clauses

### B — Base-path preserved

**Question:** Is a clean copy of the input saved before any modifications?

**Why:** The residual stream requires the original input to be added back after each sub-layer. If the input is modified in-place before saving, the residual connection carries corrupted data.

**Evidence pattern:** `residual = x.clone()` or equivalent before attention/FFN operations.

---

### L — LayerNorm placement

**Question:** Does normalization occur before each sub-layer (pre-norm)?

**Why:** Pre-norm architectures are more stable for deep networks. Post-norm can cause gradient explosion in early layers and vanishing in late layers.

**Evidence pattern:** `norm(x)` passed to attention/FFN, not `norm(x + attn_out)`.

---

### O — Operations traced

**Question:** Where does the computational work happen, and is it accounted for?

**Why:** Understanding compute distribution helps diagnose memory issues, identify bottlenecks, and verify the architecture matches expectations.

**Evidence pattern:** FFN expansion ratio documented, attention dimensions verified, all six standard operations present (embed, norm, attn, norm, ffn, output).

---

### C — Cumulative not destructive

**Question:** Do residual additions refine rather than overwrite?

**Why:** In-place operations (`+=`, `.add_()`) break autograd graphs. The gradient computation requires the original tensors; overwriting them causes silent training failure.

**Evidence pattern:** `x = x + delta` not `x += delta`.

---

### K — Kept open end-to-end

**Question:** Can gradients flow from the final output back to the first layer without interruption?

**Why:** Any break in the residual highway—a missing add, a detach call, a config flag—means early layers stop learning. The model becomes a shallow network wearing a deep network's parameter count.

**Evidence pattern:** Residual add at every block exit, no `skip_residual=true`, no `.detach()` in residual path, gradient checkpointing handled correctly.

---

## Severity Hierarchy

1. **K (Highway)** — Total training failure if broken
2. **C (Cumulative)** — Silent gradient death
3. **B (Base-path)** — Corrupted residuals, slow divergence
4. **L (LayerNorm)** — Instability, especially in deep networks
5. **O (Operations)** — Misdiagnosis, unexpected resource usage

---

## Run Call Decision Tree

```
Any clause shows RISK?
├── No → Launch
└── Yes → Is the risk in K or C?
    ├── Yes → Hold until fixed
    └── No → Is there a mitigation owner and deadline?
        ├── Yes → Launch-with-conditions
        └── No → Hold until owner assigned
```

---

## Tripwire Protocol

Every Launch-with-conditions requires:

1. **Metric** — What to watch (usually `train/loss`)
2. **Interval** — How often to check (e.g., every 100 steps)
3. **Threshold** — When to worry (e.g., loss > 2.0)
4. **Trigger** — When to stop (e.g., 3 consecutive windows above threshold)
5. **Owner** — Who responds and how (e.g., Ada on-call Slack)

---

## Application Sequence

1. **Gather** — Collect block code, config, specimen description, stakes
2. **Walk** — Run each clause prompt against the code
3. **Synthesize** — Combine findings, identify top risk
4. **Decide** — Apply decision tree for run call
5. **Instrument** — Set up tripwire monitoring
6. **Document** — Record in charter.md
7. **Verify** — Stranger test with VERIFY.md protocol

---

## Framework Principles

- **Evidence over assertion** — Every CLEAR requires a line number
- **Specificity over generality** — "Line 142" not "somewhere in the code"
- **Consequence over category** — "Loss spikes in 2 hours" not "gradient issues"
- **Ownership over awareness** — "Ada by Friday" not "team should check"

---

*BLOCK framework version 1.0 — designed for pre-flight audits of transformer fine-tuning runs.*