# Verification Protocol: Stranger Test

## Purpose

Confirm that the pre-flight bench correctly surfaces findings when given specimen code. A stranger (someone unfamiliar with this specific audit) should be able to reproduce the normalization-placement finding.

---

## Test Procedure

### Step 1: Prepare the Seeded Specimen

Use this minimal block code that contains the patterns from our calibration:

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)  # line 115
        self.attn = MultiHeadAttention(config)   # line 116
        self.ln2 = nn.LayerNorm(config.d_model)  # line 117
        self.ffn = FeedForward(config)           # line 118
    
    def forward(self, x):
        # Line 118: LayerNorm before attention (pre-norm)
        normed = self.ln1(x)
        
        # Line 142: Clean copy preserved
        residual = x.clone()
        
        # Attention computation
        attn_out = self.attn(normed)
        
        # Line 170: Additive residual (not in-place)
        x = residual + attn_out
        
        # Second sub-layer
        residual = x.clone()
        normed = self.ln2(x)
        
        # Lines 155-168: FFN with 4x expansion
        ffn_out = self.ffn(normed)  # 4x expand inside
        
        # Additive residual
        x = residual + ffn_out
        
        return x
```

### Step 2: Prepare the Config Excerpt

```yaml
# config.yaml
model:
  d_model: 1024
  n_layers: 34
  n_heads: 16
  d_ff: 4096  # 4x expansion
  
training:
  skip_residual: false  # line 44 - CRITICAL FLAG
  gradient_checkpointing: true
```

### Step 3: Run the Audit

1. Open a new conversation with any capable chat model
2. Paste the system prompt from `blueprints/pre-flight-bench.md`
3. Provide the seeded specimen code and config above
4. Add context:
   - Specimen: "34-layer decoder, tutorial-sourced, fine-tuning on clinical notes"
   - Stakes: "Divergence burns quarterly GPU budget"

### Step 4: Verify the Finding

The tool must surface the normalization-placement finding with:

**Required elements:**
- [ ] Status: CLEAR
- [ ] Component: LayerNorm
- [ ] Position: before attention
- [ ] Line reference: line 118 (or equivalent reference to `self.ln1(x)` call)
- [ ] Architecture identification: pre-norm

**Acceptable response patterns:**
- "CLEAR — because LayerNorm runs before attn at line 118 (pre-norm)"
- "CLEAR — LayerNorm at line 118 precedes attention, confirming pre-norm architecture"
- "normalization_placement: CLEAR — pre-norm confirmed, ln1 applied before attn call"

**Unacceptable responses:**
- No mention of line numbers
- Incorrect status (RISK when code is actually pre-norm)
- Missing architecture identification
- Generic "looks fine" without evidence

---

## Expected Full Output

The complete audit should return findings matching this structure:

```json
{
  "block_findings": {
    "clean_copy_of_the_input": "Line 142: residual = x.clone() — clean copy preserved",
    "normalization_placement": "CLEAR — LayerNorm at line 118 runs before attention (pre-norm)",
    "where_the_work_happens": "FFN does 4x expand (1024 → 4096) — bulk of parameters in MLP",
    "refine_never_overwrite": "CLEAR — line 170 uses x = residual + attn_out, not in-place",
    "highway_open_end_to_end": "Risk: skip_residual=false at config:44 must stay false; gradient_checkpointing=true needs verification"
  },
  "top_risk": "highway_open_end_to_end",
  "severity_note": "If skip_residual flips true, gradients die at that layer, loss diverges within hours",
  "run_call": "Launch-with-conditions: verify skip_residual stays false through config merge",
  "watch_tripwire": "Watch train/loss every 100 steps; stop if > 2.0 for 3 windows"
}
```

---

## Failure Modes

If verification fails:

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No line numbers in output | System prompt not loaded | Re-paste system prompt, confirm it's in context |
| Wrong status on normalization | Model confused by code structure | Simplify code sample, add comments |
| Missing findings | Output truncated | Request JSON output explicitly |
| Hallucinated line numbers | Model inventing evidence | Use code with actual line numbers in comments |

---

## Sign-Off

Verification complete when:

- [ ] Stranger successfully ran the test
- [ ] Normalization-placement finding surfaced with line citation
- [ ] All five clauses returned findings
- [ ] Top risk correctly identified as highway_open_end_to_end
- [ ] Output parseable as valid JSON

**Verified by:** _________________  
**Date:** _________________  
**Model used:** _________________