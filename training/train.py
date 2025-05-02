# --------------------------------------------------------------------
# Hard-patch for TRL 0.7.x on modern Transformers (≥ 4.32)
# --------------------------------------------------------------------
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import os
import torch
import transformers

# ── Patch missing symbols that TRL tries to import ────────────────────
if not hasattr(transformers, "EncoderDecoderCache"):
    class _DummyCache(dict):
        """Placeholder for transformers.EncoderDecoderCache (unused by TRL)."""
        pass
    transformers.EncoderDecoderCache = _DummyCache


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
):
    """Lightweight replacement for the helper removed after transformers 4.31."""
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth_vals, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = probs > top_p
        if min_tokens_to_keep > 1:
            sorted_mask[..., :min_tokens_to_keep] = False
        mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
        logits = logits.masked_fill(mask, filter_value)

    return logits


transformers.top_k_top_p_filtering = _top_k_top_p_filtering
setattr(transformers.generation, "top_k_top_p_filtering", _top_k_top_p_filtering)
setattr(transformers.generation.utils,
        "top_k_top_p_filtering", _top_k_top_p_filtering)

# ── Disable TensorFlow / Keras imports completely ─────────────────────
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["DISABLE_TF_IMPORTS"] = "1"

# --------------------------------------------------------------------
# Standard imports (after patch).
# --------------------------------------------------------------------

# ── LoRA configuration ────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Base model & tokenizer (BF16 fits in 80 GB H100) ──────────────────
model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, add_bos_token=True, use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # everything on GPU‑0
)
model.gradient_checkpointing_enable()

# attach LoRA adapters
model = get_peft_model(model, lora_config)

# ── Dataset loading ───────────────────────────────────────────────────
train_ds = load_dataset("json", data_files="train.jsonl", split="train")
eval_ds = load_dataset("json", data_files="eval.jsonl", split="train")


def format_chat(example):
    """Convert list‑of‑messages → single prompt string using Gemma template."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


# ── Trainer args (replace per‑device kwargs) ──────────────────────────
training_args = TrainingArguments(
    output_dir="paul-graham-lora",
    num_train_epochs=2,
    weight_decay=0.1,
    eval_steps=100,
    load_best_model_at_end=True if eval_ds is not None else False,
    per_device_train_batch_size=8,  # fits H100 in BF16
    gradient_accumulation_steps=1,
    logging_steps=10,
    save_steps=500,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    formatting_func=format_chat,
    max_seq_length=2048,
)

if __name__ == "__main__":
    trainer.train()
