# Local training on macOS (Apple Silicon)

This reference explains how to run **small fine-tuning jobs locally on a Mac** (best for smoke tests and quick iteration).

> **Clarification:** This is not about "training on phone." The workflow is: **train locally or cloud → export/quantize → run inference on-device**.

For general issues, see [troubleshooting.md](troubleshooting.md). This doc focuses on **macOS-specific** setup and issues.

## Quickstart (TL;DR)

```bash
# 1. Setup (see detailed steps in Setup section below)
xcode-select --install
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U "torch>=2.2" "transformers>=4.40" "trl>=0.12" "peft>=0.10" \
    datasets accelerate safetensors huggingface_hub

# 2. Run smoke test (see full script below)
python train_lora_sft.py

# 3. Verify
ls outputs/local-lora/  # should contain adapter weights (*.safetensors) + adapter_config.json
```

> **Note:** If you hit `TypeError` or version conflicts, see [troubleshooting.md](troubleshooting.md) for pinning guidance.

<details>
<summary><strong>Recommended requirements.txt for reproducibility</strong></summary>

```txt
torch>=2.2,<3.0
transformers>=4.40,<5.0
trl>=0.12,<1.0
peft>=0.10,<1.0
datasets>=2.18,<3.0
accelerate>=0.28,<1.0
safetensors>=0.4,<1.0
huggingface_hub>=0.21,<1.0
```

Install with:
```bash
pip install -r requirements.txt
```

</details>

## Agent Decision Rubric

**Run locally on Mac when:**
- Model ≤3B parameters (text-only)
- Short context (≤1024 tokens)
- LoRA/PEFT fine-tuning
- Quick smoke test or dataset validation

**Recommend HF Jobs / cloud GPU when:**
- Model 7B+ parameters
- Vision-language models (VLMs)
- QLoRA 4-bit training (CUDA/bitsandbytes-centric)
- Long context or full fine-tuning
- Production training runs

## Scope

✅ Good for (local Mac):
- Quick experiments / smoke tests
- **Text-only** models ~0.5B–3B
- **LoRA SFT** with small batches + short context

⚠️ Usually not good on a Mac laptop:
- Large models (7B+) for a pleasant dev loop
- **QLoRA 4-bit training** (often CUDA/bitsandbytes-centric)
- Vision-language fine-tuning (VLMs) at real scale

If you need VLM training (e.g., LLaVA/Qwen-VL) or larger models, prefer **HF Jobs / cloud GPU** and keep local for validation.

## How this fits the model-trainer skill

This skill primarily targets **HF Jobs** (cloud training). Local Mac training is most useful to:
- Validate dataset formatting and prompt templates
- Confirm your LoRA setup works end-to-end
- Run quick regression tests before submitting a real HF Jobs run

Typical workflow:
1. Run a small local LoRA smoke test (this doc)
2. Move the "real run" to HF Jobs with the same model/dataset/hyperparams
3. After training, export/quantize as needed (see [gguf_conversion.md](gguf_conversion.md) for on-device inference)

## Before you start

### Recommended "local-friendly" defaults
- Model: 0.5B–1.5B for first run
- Max seq length: 512–1024
- Batch size: 1
- Gradient accumulation: 8–16
- LoRA: r=8–16, alpha=16–32
- Save only adapters (small artifacts)

### Rough memory guidance (Apple Silicon unified memory)
Very approximate; depends on context length and model:
- 16GB: start with ~0.5B–1.5B
- 32GB: ~1.5B–3B
- 64GB: larger experiments possible, but long-context can still blow up

## Setup

### 0) Xcode CLT
```bash
xcode-select --install
```

### 1) Create a venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install training deps
```bash
python -m pip install -U "torch>=2.2" "transformers>=4.40" "trl>=0.12" "peft>=0.10" \
    datasets accelerate safetensors huggingface_hub
```

**Note:** Use a recent stable PyTorch version; MPS support improves frequently. Check your version:
```bash
python -c "import torch; print(torch.__version__, '| MPS available:', torch.backends.mps.is_available())"
```

### 3) (Optional) Configure Accelerate
```bash
accelerate config
```

Suggested answers for local Mac:
- compute environment: local machine
- distributed: no
- mixed precision: no (recommended for MPS stability)
- device: MPS (if offered)

### 4) (Optional) Login to Hugging Face
Only needed if you'll push artifacts to the Hub:
```bash
huggingface-cli login
```

## Run: Local LoRA SFT smoke test

### Why this recipe?
- Works without CUDA-specific toolchains
- Uses conservative settings for Mac (small batch, gradient accumulation, checkpointing)
- Uses TRL API (`SFTConfig` + `processing_class`)

### Fastest first run

For the quickest smoke test, limit steps (not just dataset size):
```bash
# Ultra-fast: only 50 training steps
MAX_STEPS=50 python train_lora_sft.py
```

Or use a tiny dataset slice:
```bash
DATASET_SPLIT="train_sft[:100]" MAX_SEQ_LENGTH=512 python train_lora_sft.py
```

### Using a local JSONL file

Create `test_data.jsonl`:
```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

Run with:
```bash
DATA_FILES="test_data.jsonl" python train_lora_sft.py
```

For text-only format (no chat template), create `test_text.jsonl`:
```jsonl
{"text": "User: Hello\nAssistant: Hi there!"}
{"text": "User: What is 2+2?\nAssistant: 4"}
```

Run with:
```bash
DATA_FILES="test_text.jsonl" TEXT_FIELD="text" MESSAGES_FIELD="" python train_lora_sft.py
```

<details>
<summary><strong>Full training script: train_lora_sft.py</strong></summary>

```python
import os
from dataclasses import dataclass
from typing import Optional
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Reproducibility
set_seed(42)

@dataclass
class Cfg:
    model_id: str = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    dataset_id: str = os.environ.get("DATASET_ID", "HuggingFaceH4/ultrachat_200k")
    dataset_split: str = os.environ.get("DATASET_SPLIT", "train_sft[:500]")
    data_files: Optional[str] = os.environ.get("DATA_FILES", None)  # For local JSONL
    text_field: str = os.environ.get("TEXT_FIELD", "")
    messages_field: str = os.environ.get("MESSAGES_FIELD", "messages")
    out_dir: str = os.environ.get("OUT_DIR", "outputs/local-lora")
    max_seq_length: int = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
    max_steps: int = int(os.environ.get("MAX_STEPS", "-1"))  # <=0 = use epochs

cfg = Cfg()

def get_device():
    """Get the best available device for Mac training."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype():
    """
    Get optimal dtype for stability.
    - fp32 is the safest default for MPS
    - bf16 works on M1 Pro/Max/Ultra and all M2/M3/M4 chips (PyTorch 2.1+)
    - bf16 is NOT supported on base M1 or Intel Macs
    - fp16 often causes NaN issues on MPS
    """
    return torch.float32

device = get_device()
print(f"Model: {cfg.model_id}")
if cfg.data_files:
    print(f"Dataset: local file ({cfg.data_files})")
else:
    print(f"Dataset: {cfg.dataset_id} ({cfg.dataset_split})")
print(f"Device: {device}")
print(f"Dtype: {get_dtype()}")
if cfg.max_steps > 0:
    print(f"Max steps: {cfg.max_steps}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for causal LM training

# Load model
# Note: Some models require trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=get_dtype(),
    # trust_remote_code=True,  # Uncomment if model requires custom code
)
model.to(device)

# Important: disable cache when using gradient checkpointing
model.config.use_cache = False

# Load dataset
if cfg.data_files:
    # Local JSONL file
    ds = load_dataset("json", data_files=cfg.data_files, split="train")
else:
    # Hub dataset
    ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split)

def format_example(ex):
    # Case A: dataset already has plain text
    if cfg.text_field and isinstance(ex.get(cfg.text_field), str):
        ex["text"] = ex[cfg.text_field]
        return ex

    # Case B: chat-like datasets with messages list
    msgs = ex.get(cfg.messages_field)
    if isinstance(msgs, list):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                ex["text"] = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                return ex
            except Exception:
                pass
        # Fallback: naive join (smoke test only)
        ex["text"] = "\n".join([str(m) for m in msgs])
        return ex

    # Last resort
    ex["text"] = str(ex)
    return ex

ds = ds.map(format_example)

# Drop unused columns to reduce memory
cols_to_keep = ["text"]
ds = ds.remove_columns([c for c in ds.column_names if c not in cols_to_keep])

# LoRA config
# Note: target_modules vary by architecture.
# Common patterns:
#   - Llama/Qwen/Mistral: q_proj, k_proj, v_proj, o_proj
#   - Some models add: gate_proj, up_proj, down_proj
# If you get "module not found" errors, see Troubleshooting section below.
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# SFT config
# Note: TRL API evolves; some param names may differ across versions.
# If you hit TypeErrors, see troubleshooting.md for API mismatches.
# Build config dict conditionally to avoid passing None (which can break some TRL versions)
sft_kwargs = dict(
    output_dir=cfg.out_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none",
    # MPS stability: fp32 is safest. Only enable fp16/bf16 if tested on your machine.
    fp16=False,
    bf16=False,
    max_seq_length=cfg.max_seq_length,
    dataset_text_field="text",
)

if cfg.max_steps > 0:
    sft_kwargs["max_steps"] = cfg.max_steps
else:
    sft_kwargs["num_train_epochs"] = 1

sft_args = SFTConfig(**sft_kwargs)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=lora,
    args=sft_args,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(cfg.out_dir)
print(f"✅ Saved to: {cfg.out_dir}")
```

</details>

### Run it

```bash
python train_lora_sft.py
```

Optional overrides:
```bash
# Different model
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" python train_lora_sft.py

# Quick 50-step test
MAX_STEPS=50 python train_lora_sft.py

# Local data file
DATA_FILES="my_data.jsonl" python train_lora_sft.py
```

If MPS is flaky, run with fallback:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_lora_sft.py
```

To help with memory pressure on Mac:
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_lora_sft.py
```

> **Caution:** Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` disables memory limits and may cause system instability or unresponsiveness if memory is exhausted. Use with care and monitor Activity Monitor.

## What success looks like

✅ Good signs:
- Loss decreases over steps
- Output directory contains adapter weights + config
- A quick generation test runs without errors

**Expected training output:**
```
Model: Qwen/Qwen2.5-0.5B-Instruct
Dataset: HuggingFaceH4/ultrachat_200k (train_sft[:500])
Device: mps
Dtype: torch.float32
{'loss': 2.1453, 'grad_norm': 1.234, 'learning_rate': 0.0002, 'epoch': 0.16}
{'loss': 1.8721, 'grad_norm': 0.987, 'learning_rate': 0.0002, 'epoch': 0.32}
...
✅ Saved to: outputs/local-lora
```

If loss is NaN / exploding:
- Ensure `fp16=False` (default in the script above)
- Reduce learning rate (e.g., 2e-4 → 1e-4 or 5e-5)
- Shorten max sequence length

## After: quick local evaluation

<details>
<summary><strong>Evaluation script: eval_generate.py</strong></summary>

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER_DIR", "outputs/local-lora")

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model.to(device)
model = PeftModel.from_pretrained(model, ADAPTER)

prompt = os.environ.get("PROMPT", "Explain gradient accumulation in 3 bullet points.")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

</details>

Run:
```bash
python eval_generate.py
```

## After: push artifacts to the Hub (optional)

Prefer uploading adapters for local runs (smaller artifacts).
See also: [hub_saving.md](hub_saving.md)

```bash
huggingface-cli login
```

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="outputs/local-lora",
    repo_id="YOUR_USERNAME/YOUR_ADAPTER_REPO",
    repo_type="model",
)
```

## After: transition from local → HF Jobs

Use local runs to validate dataset formatting, chat template, LoRA target modules, and general stability.

Then run the real job on HF Jobs:
- Keep `MODEL_ID`, dataset id, prompt formatting, and hyperparams consistent
- Bump dataset size + context length gradually
- Track results with the skill's normal workflow

## After: export for on-device inference

If your end goal is on-device inference (e.g., GGUF/llama.cpp):
1. Train (local or cloud)
2. Merge adapters (if needed)
3. Convert + quantize

See [gguf_conversion.md](gguf_conversion.md) for details.

## Troubleshooting (macOS)

For broader issues, see [troubleshooting.md](troubleshooting.md).
This section covers **Mac-specific** problems.

### MPS errors (unsupported op / crash)
Some PyTorch ops aren't fully supported on MPS. Use CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_lora_sft.py
```

This can be slower but usually prevents crashes.

### Monitoring GPU memory usage
To monitor MPS memory during training:

**Activity Monitor (GUI):**
1. Open Activity Monitor → Window → GPU History
2. Watch "Memory Used" during training

**Command line:**
```bash
# Monitor GPU power and memory (requires sudo)
sudo powermetrics --samplers gpu_power -i 1000

# Or use this Python snippet during training:
python -c "import torch; print(f'MPS allocated: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB')"
```

**In your training script**, you can add periodic memory logging:
```python
if torch.backends.mps.is_available():
    print(f"MPS memory: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
```

### Out of memory (OOM)
If training crashes or your Mac becomes unstable:
- Reduce `MAX_SEQ_LENGTH` (1024 → 768 → 512)
- Use a smaller model (e.g., 0.5B instead of 1.5B)
- Set memory high watermark (use with caution):
  ```bash
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_lora_sft.py
  ```
  **Warning:** This may cause system instability. Monitor Activity Monitor → Memory Pressure.
- Use fewer examples (e.g., `train_sft[:500]` → `train_sft[:100]`)
- Keep batch size at 1 and scale with gradient accumulation
- Close other memory-heavy apps

### fp16 instability (NaNs / loss explodes)
If loss becomes `nan` or suddenly explodes:
- Set `fp16=False` in the script (recommended default for MPS)
- Lower learning rate (2e-4 → 1e-4 → 5e-5)
- Reduce `MAX_SEQ_LENGTH`

### Intel Macs (not supported)
Intel Macs don't have MPS acceleration, so training would be CPU-only and impractically slow. This guide has not been tested on Intel Macs. If you're on Intel, use HF Jobs or cloud GPU for all training.

### LoRA target module mismatch
If you see "module not found" errors, the model architecture uses different module names.

**Quick debug:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-model-id")
# Print likely LoRA targets (attention projections)
for name, _ in model.named_modules():
    if any(x in name.lower() for x in ["q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "dense"]):
        print(name)
```

**Common patterns:**
| Architecture | target_modules |
|--------------|----------------|
| Llama/Qwen/Mistral | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| GPT-2/GPT-J | `c_attn`, `c_proj` |
| BLOOM | `query_key_value`, `dense` |

### TRL version differences
The script uses `SFTConfig` + `processing_class` (TRL ≥0.12). If you hit `TypeError` on arguments like `max_seq_length` vs `max_length`, or `dataset_text_field` issues, check your TRL version and see [troubleshooting.md](troubleshooting.md) for API mismatches.

## Alternative: MLX for Apple Silicon

[MLX](https://github.com/ml-explore/mlx) is Apple's machine learning framework optimized for Apple Silicon. While this guide focuses on PyTorch + MPS (for compatibility with HF ecosystem), MLX can offer better performance on Mac for some workflows.

**When to consider MLX:**
- You're doing inference-heavy workflows on Apple Silicon
- You want tighter Metal/GPU integration
- You're building Mac-native ML applications

**MLX limitations for this skill:**
- Smaller ecosystem than PyTorch/HF
- Not all HF models have MLX ports
- Training APIs are less mature than TRL/PEFT
- Harder to transfer workflows to cloud GPU

**Resources:**
- [mlx-lm](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm) – LLM inference and fine-tuning with MLX
- [MLX documentation](https://ml-explore.github.io/mlx/)

For this skill's workflow (local validation → HF Jobs), PyTorch + MPS remains the recommended path for consistency.

## See Also

- [troubleshooting.md](troubleshooting.md) – General TRL troubleshooting
- [hardware_guide.md](hardware_guide.md) – GPU selection for HF Jobs
- [gguf_conversion.md](gguf_conversion.md) – Export for on-device inference
- [training_methods.md](training_methods.md) – SFT, DPO, GRPO overview