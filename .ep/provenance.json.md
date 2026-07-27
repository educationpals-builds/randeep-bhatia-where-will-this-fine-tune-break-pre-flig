{
  "schema_version": "1.0",
  "build_name": "Where will this fine-tune break? pre-flight check for any open model you're about to train",
  "methodology": "baw.v3",
  "generated_at": "2024",
  "eu_ai_act_article_50_marking": "ai_drafted",
  "disclosure": "This workshop content was drafted by an AI system and is disclosed as such per EU AI Act Article 50 transparency requirements.",
  "field_provenance": {
    "learner_provided": [
      "specimen",
      "specimen_stakes",
      "standard_line",
      "run_reality",
      "block_findings",
      "top_risk",
      "severity_note",
      "run_call",
      "watch_tripwire"
    ],
    "ai_drafted": [
      "README.md structure and prose",
      "charter.md expanded narrative",
      "blueprints/pre-flight-bench.md auditor specification",
      "prompts/clause-walk-pack.md prompt templates",
      "METHOD.md framework documentation",
      "VERIFY.md verification protocol",
      "file organization and formatting"
    ]
  },
  "source_fields": {
    "specimen": {
      "value": "34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days",
      "source": "learner"
    },
    "specimen_stakes": {
      "value": "A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review",
      "source": "learner"
    },
    "standard_line": {
      "value": "Safe to train means: every one of the six operations in the block is accounted for in the code we will actually run, normalization sits before each sub-layer, the input path is preserved by addition in both halves, and a 200-step smoke run shows gradients alive in the deepest layers",
      "source": "learner"
    },
    "run_reality": {
      "value": "8×A100 for ~11 days, fp16 by default, 34 layers, sequence length 4096, nobody on the team has traced the forward pass end to end, and the repo's README says 'works out of the box'",
      "source": "learner"
    },
    "block_findings": {
      "value": {
        "clean_copy_of_the_input": "Line 142: residual = x.clone() before attn — clean copy preserved.",
        "normalization_placement": "CLEAR — because LayerNorm runs before attn at line 118 (pre-norm).",
        "where_the_work_happens": "FFN at lines 155-168 does 4x expand — work happens in MLP not attention.",
        "refine_never_overwrite": "CLEAR — because residual add at line 170 is x + delta, not in-place overwrite.",
        "highway_open_end_to_end": "Risk: layer 30 residual path may drop if checkpoint omit — key skip_residual=false at config:44."
      },
      "source": "learner"
    },
    "top_risk": {
      "value": "highway_open_end_to_end",
      "source": "learner"
    },
    "severity_note": {
      "value": "If residual drops at layer 30 around step 40k, loss spikes >2.0 within 2 hours and gradients explode.",
      "source": "learner"
    },
    "run_call": {
      "value": "Launch-with-conditions: Ada owns residual path audit by Friday; hold if skip_residual ever true.",
      "source": "learner"
    },
    "watch_tripwire": {
      "value": "Watch train/loss every 100 steps; stop if loss > 2.0 for 3 windows. Owner: Ada on-call Slack.",
      "source": "learner"
    }
  },
  "pooled_sources": [],
  "human_review_status": "pending",
  "intended_use": "Educational workshop for pre-flight auditing of transformer fine-tuning runs",
  "limitations": [
    "Framework assumes standard transformer architecture with residual connections",
    "Line numbers in examples are illustrative and must be verified against actual code",
    "Tripwire thresholds may need adjustment for specific model scales",
    "Does not cover distributed training edge cases"
  ]
}