# Pre-Flight Charter: Fine-Tune Break-Point Audit

## Specimen Under Review

**Model:** 34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days.

**Stakes:** A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review.

---

## Standard Applied

> Safe to train means: every one of the six operations in the block is accounted for in the code we will actually run, normalization sits before each sub-layer, the input path is preserved by addition in both halves, and a 200-step smoke run shows gradients alive in the deepest layers.

---

## Run Reality

- **Compute:** 8×A100 for ~11 days
- **Precision:** fp16 by default
- **Architecture:** 34 layers, sequence length 4096
- **Team state:** Nobody on the team has traced the forward pass end to end
- **Repo claim:** README says "works out of the box"

---

## Five Clause Findings

### 1. Clean Copy of the Input

**Status:** ✓ CLEAR

**Evidence:** Line 142: `residual = x.clone()` before attn — clean copy preserved.

**Why it matters:** Without a clean copy, the residual stream gets corrupted by in-place operations, and gradients cannot flow back through the original path.

---

### 2. Normalization Placement

**Status:** ✓ CLEAR

**Evidence:** LayerNorm runs before attn at line 118 (pre-norm architecture).

**Why it matters:** Pre-norm stabilizes training by normalizing inputs to each sub-layer. Post-norm or missing norm causes gradient instability in deep networks.

---

### 3. Where the Work Happens

**Status:** ✓ CLEAR

**Evidence:** FFN at lines 155-168 does 4x expand — work happens in MLP not attention.

**Why it matters:** Understanding where computation concentrates helps diagnose bottlenecks and memory issues. The 4x expansion in FFN is the memory-heavy operation.

---

### 4. Refine Never Overwrite

**Status:** ✓ CLEAR

**Evidence:** Residual add at line 170 is `x + delta`, not in-place overwrite.

**Why it matters:** In-place operations (`x += delta` or `x.add_()`) break autograd graphs and cause silent gradient death. Additive refinement preserves the computation graph.

---

### 5. Highway Open End-to-End

**Status:** ⚠ RISK IDENTIFIED

**Evidence:** Layer 30 residual path may drop if checkpoint omit — key `skip_residual=false` at config:44.

**Why it matters:** If the residual highway closes at any layer, gradients cannot flow from loss back to early layers. The model forgets how to learn.

**Specific concern:** Config line 44 contains `skip_residual=false`. If this flag flips during checkpoint loading or config merge, layer 30 onward loses the residual path.

---

## Severity Story

If residual drops at layer 30 around step 40k:
- Loss spikes >2.0 within 2 hours
- Gradients explode
- Run becomes unrecoverable
- Quarter's GPU budget burned
- Delivery date pushed past contract review

---

## Launch Call

**Decision:** Launch-with-conditions

**Conditions:**
1. Ada owns residual path audit by Friday
2. Hold launch if `skip_residual` ever evaluates to true in any config path
3. 200-step smoke run must show gradients alive at layer 33

---

## Tripwire Protocol

**Metric:** `train/loss`

**Frequency:** Every 100 steps

**Threshold:** loss > 2.0

**Trigger:** 3 consecutive windows above threshold → immediate stop

**Owner:** Ada on-call Slack

**Escalation:** If triggered, dump gradient norms per layer before killing run.

---

## The Builder's Run Checklist

- [ ] Trace forward pass from input embedding to output logits
- [ ] Confirm `skip_residual=false` in merged config at runtime
- [ ] Run 200-step smoke test
- [ ] Log gradient norms at layers 1, 17, 30, 33
- [ ] Set up loss monitoring with 100-step windows
- [ ] Ada confirms residual audit complete
- [ ] Document checkpoint loading sequence

---

*Charter generated as part of BLOCK pre-flight methodology. See METHOD.md for framework details.*