# Blockwise-SFT

Blockwise SFT: Bridging the Gap between Bidirectional Attention and Autoaggressive Decoding

A lightweight recipe to (1) preprocess an instruction dataset into a compact `.pt` file and (2) fine-tune with either **blockwise masked SFT** (discrete-diffusion-friendly) or **full-sequence SFT** using LoRA + DeepSpeed.

---

## What you get

* **`pt.py`** – preprocesses MetaMathQA into `MathQA_256.pt` containing
  token IDs and prompt lengths ready for training.
* **`train_blockwise.py`** – Random-Block Masked SFT with
  prefix/suffix ablations and automatic one-block-per-sample sampling.
* **`train_fullseq.py`** – Full-sequence SFT with random masking over the
  entire answer span.
* Both trainers:

  * No validation loop (by design).
  * Write `(step, loss)` to CSV for easy plotting.
  * Save **fully merged** Hugging Face checkpoints (LoRA merged into the base).

> **Model & dataset defaults**
>
> * Base model: `GSAI-ML/LLaDA-8B-Instruct`
> * Dataset: `meta-math/MetaMathQA` (`train` split)

---

## Setup

```bash
# Python 3.10+ recommended
pip install -U torch deepspeed transformers peft datasets tqdm

# (Optional) if you use bf16 on Ampere+ GPUs
# Ensure your CUDA/driver stack supports bf16
```

> Tip: If you plan to push/download models from Hugging Face, make sure
> your environment has access (e.g., `huggingface-cli login`).

---

## 1) Preprocess with `pt.py`

`pt.py` renders conversations using the model’s chat template, computes
`prompt_len` (token count of the user-only prompt), and stores
`full_ids` (prompt + answer) as token IDs. Samples longer than `--max_len` are
skipped.

### CLI

```bash
python pt.py \
  --output MathQA_256.pt \
  --model GSAI-ML/LLaDA-8B-Instruct \
  --dataset meta-math/MetaMathQA \
  --split train \
  --max_len 256 \
  --max_samples 20000 \
  --filter_type GSM
```

**Arguments (most used)**

* `--output` (str): output path for the `.pt` file. Default: `MathQA_256.pt`
* `--model` (str): tokenizer/model for the chat template. Default: `GSAI-ML/LLaDA-8B-Instruct`
* `--dataset` (str): HF dataset ID. Default: `meta-math/MetaMathQA`
* `--split` (str): dataset split. Default: `train`
* `--max_len` (int): drop samples with `len(full_ids) > max_len`. Default: `256`
* `--max_samples` (int): cap total kept samples (0 = unlimited). Default: `20000`
* `--filter_type` (str): skip examples whose `ex["type"]` contains this substring (empty string disables). Default: `GSM`

### Output format

The file is a `list` of dictionaries, saved with `torch.save`:

```python
[
  {"prompt_len": <int>, "full_ids": torch.LongTensor([...])},
  ...
]
```

* `prompt_len` ≤ `len(full_ids)`
* `full_ids` already matches the tokenizer used during preprocessing

> **Important:** The tokenizer’s chat template may append small trailing text.
> The script applies a conservative workaround (strip 47 tail characters) to
> match the training pipeline. Adjust if you change models/templates.

---

## 2) Training

Both trainers expect the `.pt` file (default: `MathQA_256.pt`).
They create a DeepSpeed config file (`ds_zero2.json`) automatically.

### Launching

We recommend using the **DeepSpeed launcher**:

```bash
deepspeed --num_gpus 8 train_blockwise.py
# or
deepspeed --num_gpus 8 train_fullseq.py
```

For single-GPU experiments you can also try:

```bash
deepspeed --num_gpus 1 train_blockwise.py
```

> The scripts call `deepspeed.init_distributed()` internally, so launching with
> the DeepSpeed/Torch distributed runner is preferred.

---

### Option A – Blockwise SFT (`train_blockwise.py`)

**What it does**

* Splits the answer into fixed-size blocks (`BLOCK_SIZE`).
* At each training step, picks **one block** per sample and masks it with a
  random rate `p ~ U(eps, 1)`. Only the active block contributes to the loss.
* **Prefix ablation** (`PREFIX_MASK_RATE`): randomly mask the answer prefix
  (between prompt end and active block), **excluded** from loss.
* **Suffix ablation** (`SUFFIX_MASK_PROB`): randomly mask tokens after the
  active block, **excluded** from loss. `1.0` reproduces “mask all suffix”.

**Key knobs (edit in the script)**

* `BLOCK_SIZE` (default: `32`)
* `PREFIX_MASK_RATE` (default: `0.0`)
* `SUFFIX_MASK_PROB` (default: `1.0`)
* `MICRO_BSIZE`, `GRAD_ACC`, `LR`
* `TOTAL_TRAIN_STEPS` (computed from `MAX_SAMPLES`, `EPOCHS`, `MICRO_BSIZE`)
* `SAVE_EVERY` (defaults to saving **once at the end**; lower it to save more often)
* `CKPT_DIR` (checkpoint root directory)

**Artifacts**

* Checkpoints: `ckpts_stagewise_sft_32_stepXXXX/` (directory name is uniquified
  if it already exists)
* Logs: `ckpts_stagewise_sft_32_logs/train_loss.csv`
* Ablation config snapshot: `ckpts_stagewise_sft_32_logs/ablation_config.json`

> **Merged checkpoints:** Saving uses **LoRA-merged** weights (full base + LoRA),
> serialized in safetensors, so disk usage is similar to a full model.

---

### Option B – Full-sequence SFT (`train_fullseq.py`)

**What it does**

* Randomly masks **the whole answer span** each step with a rate
  `p ~ U(eps, 1)`; masked tokens contribute to the loss.

**Key knobs (edit in the script)**

* `MICRO_BSIZE`, `GRAD_ACC`, `LR`
* `TOTAL_TRAIN_STEPS` and `SAVE_EVERY` (same semantics as above)
* `CKPT_DIR` (default: `ckpts_fullseq_sft`)

**Artifacts**

* Checkpoints: `ckpts_fullseq_sft_stepXXXX/`
* Logs: `ckpts_fullseq_sft_logs/train_loss.csv`

---

## Reproducibility & logging

* Seeds: set via `set_seed(SEED)` (default `SEED=42`).
* Loss curve: `(step, loss)` CSV under `*_logs/train_loss.csv`.
* Progress: `tqdm` progress bar on rank 0.
* Early stop (blockwise only): if every `(sample, block)` has been used, training stops early.

---

## Practical tips

* **Memory / OOM**

  * Decrease `MICRO_BSIZE`, increase `GRAD_ACC`.
  * BF16 is enabled by default in DeepSpeed config; ensure your GPU supports it.
* **Save frequency**

  * By default `SAVE_EVERY == TOTAL_TRAIN_STEPS` (save at the end).
  * Lower `SAVE_EVERY` to get mid-training snapshots.
* **Tokenizer alignment**

  * Keep the **same tokenizer** for preprocessing and training.
  * If you switch models/templates, regenerate the `.pt` file.
* **MASK\_ID**

  * Both trainers assume `MASK_ID=126336` (the special `<mask>` used in the
    dataset/pipeline). Adjust consistently if your tokenizer uses a different id.

---

## Minimal end-to-end example

```bash
# 1) Preprocess (creates MathQA_256.pt)
python pt.py --output MathQA_256.pt --max_len 256 --max_samples 20000

# 2) Train (blockwise)
deepspeed --num_gpus 8 train_blockwise.py

#    or full-seq SFT
deepspeed --num_gpus 8 train_fullseq.py
```

After training, you’ll find merged Hugging Face checkpoints under
`ckpts_*_stepXXXX/`. You can load them with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("path/to/ckpts_stagewise_sft_32_step0001")
mdl = AutoModelForCausalLM.from_pretrained("path/to/ckpts_stagewise_sft_32_step0001", torch_dtype="auto")
```

---

## License & attribution

* Code is intended for research use. Check the licenses of:

  * The base model `GSAI-ML/LLaDA-8B-Instruct`.
  * The dataset `meta-math/MetaMathQA`.
* Respect dataset and model terms when redistributing derivatives.
