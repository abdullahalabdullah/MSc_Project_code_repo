"""Self-contained loader for PKU-Alignment/beaver-7b-unified-cost.

The official model card loads this via `safe_rlhf.models.AutoModelForScore`, but that
package pins an old `transformers` and pulls DeepSpeed — fragile to install alongside the
vLLM / Qwen3 stack. The beaver cost model is architecturally just a LLaMA backbone plus a
linear score head, so we reimplement that here with plain `transformers`. This lets the
whole pipeline run in ONE environment.

Correctness note: safe-rlhf may apply an affine score *normalizer* at inference. We output
the raw pre-normalizer score. Because normalization is a positive-scale affine map, it
preserves (a) the per-prompt argmin/argmax used to pick chosen/rejected and (b) which group
(Unsafe vs Safe) has the larger mean in the sign-convention sanity check. Only the absolute
magnitudes differ from the reference values on the model card — the pipeline's decisions do
not depend on those. Higher cost still = more harmful.

If `safe_rlhf` IS installed, stage 04 prefers it; this is the fallback.
"""
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel


class LlamaModelForScore(LlamaPreTrainedModel):
    """LLaMA backbone + linear score head, returning the score at the last real token."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        score_dim = getattr(config, "score_dim", 1)
        # safe-rlhf's score head is bias-free; checkpoint holds only "score_head.weight".
        self.score_head = nn.Linear(config.hidden_size, score_dim, bias=False)
        self.post_init()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state              # (B, L, H)
        scores = self.score_head(hidden)                # (B, L, score_dim)

        if attention_mask is None:
            end_index = torch.full(
                (input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device
            )
        else:
            # Right-padded: last real token is at sum(mask) - 1.
            end_index = attention_mask.sum(dim=1) - 1    # (B,)

        end_scores = scores[torch.arange(scores.size(0), device=scores.device), end_index]
        return end_scores                                # (B, score_dim)


def load_cost_model(model_id, torch_dtype=torch.bfloat16):
    """Return (model, source_tag). Prefers safe_rlhf if present, else the local class."""
    try:
        from safe_rlhf.models import AutoModelForScore
        model = AutoModelForScore.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        ).eval()
        return model, "safe_rlhf"
    except Exception:
        model = LlamaModelForScore.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        ).eval()
        return model, "local"


def end_scores(model, source_tag, enc):
    """Run a forward pass and return the per-example scalar cost tensor (B,)."""
    if source_tag == "safe_rlhf":
        out = model(**enc)
        return out.end_scores.squeeze(-1)
    return model(**enc).squeeze(-1)
