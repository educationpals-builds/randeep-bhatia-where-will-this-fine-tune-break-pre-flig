# Where Will This Fine-Tune Break? — Pre-Flight Check for Open Models

## The Specimen

34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days.

**Stakes:** A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review.

## The Verdict

**Launch-with-conditions:** Ada owns residual path audit by Friday; hold if `skip_residual` ever true.

### Block Findings Summary

| Clause | Status | Evidence |
|--------|--------|----------|
| Clean copy of the input | ✓ CLEAR | Line 142: `residual = x.clone()` before attn — clean copy preserved |
| Normalization placement | ✓ CLEAR | LayerNorm runs before attn at line 118 (pre-norm) |
| Where the work happens | ✓ CLEAR | FFN at lines 155-168 does 4x expand — work happens in MLP not attention |
| Refine never overwrite | ✓ CLEAR | Residual add at line 170 is `x + delta`, not in-place overwrite |
| Highway open end-to-end | ⚠ RISK | Layer 30 residual path may drop if checkpoint omit — key `skip_residual=false` at config:44 |

**Top Risk:** `highway_open_end_to_end`

## The Tripwire

Watch `train/loss` every 100 steps; stop if loss > 2.0 for 3 consecutive windows.

**Owner:** Ada on-call Slack.

**Severity:** If residual drops at layer 30 around step 40k, loss spikes >2.0 within 2 hours and gradients explode.

## One-Paste Rebuild Block

```
Specimen: 34-layer open decoder, tutorial-sourced block, 180k clinical notes, 9-day runway
Standard: Six ops accounted, pre-norm placement, additive residuals both halves, 200-step smoke with deep gradients alive
Reality: 8×A100 ~11 days, fp16, seq 4096, forward pass untraced, README claims plug-and-play
Top Risk: highway_open_end_to_end — layer 30 residual may drop
Call: Launch-with-conditions — Ada audits residual path by Friday, hold if skip_residual=true
Tripwire: loss > 2.0 for 3 windows → stop. Owner: Ada.
```

---

**Method:** See [METHOD.md](METHOD.md) for the BLOCK framework.

**Verification:** See [VERIFY.md](VERIFY.md) for stranger-test instructions.

<!-- educationpals-build-verified -->