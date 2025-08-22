#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random-Block Masked SFT (discrete-diffusion friendly) + Prefix/Suffix Ablations

Key behavior:
- Each (sample, block) is sampled at most once during training; training stops
  automatically after all blocks are consumed or when TOTAL_TRAIN_STEPS is reached.
- No validation: the validation set is empty and no evaluation is performed.
- Checkpoint saving: every SAVE_EVERY steps, save a full-merged HF checkpoint
  exactly the same as the final save (unique directory to avoid name clashes).
- Training logs: (step, loss) are appended to a CSV file for plotting later.
- Ablations:
    • Prefix noise: tokens in the answer prefix (after the prompt, before the
      active block) are randomly masked with probability PREFIX_MASK_RATE and
      are excluded from the loss.
    • Suffix masking: tokens after the active block are masked with probability
      SUFFIX_MASK_PROB and are excluded from the loss.
"""

import json
import math
import os
import random
import shutil
import tempfile
from typing import List, Tuple

import deepspeed
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ===================== Hyperparameters =====================
CKPT_DIR = "ckpts_stagewise_sft_32"
DATA_PT = "MathQA_256.pt"
MAX_SAMPLES = 200 * 32
MAX_LEN = 256

BLOCK_SIZE = 32
MASK_PROB_EPS = 1e-3  # block's random masking rate p ~ U(eps, 1)

# ---- Prefix / Suffix ablation settings ----
# Prefix noise perturbs the context (excluded from loss).
# Suffix masking controls visibility of tokens after the active block (excluded from loss).
PREFIX_MASK_RATE = 0.0  # e.g., 0.1 / 0.2 / 0.5; 0.0 disables prefix noise
SUFFIX_MASK_PROB = 1.0  # 1.0 = mask all suffix (original behavior), 0.0 = fully visible

EPOCHS = 1
MICRO_BSIZE = 32
GRAD_ACC = 1
LR, EPS = 1e-5, 1e-8

# Training steps & save frequency (explicit int to avoid float drift)
TOTAL_TRAIN_STEPS = int(MAX_SAMPLES * EPOCHS / MICRO_BSIZE)
SAVE_EVERY = TOTAL_TRAIN_STEPS

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
LORA_RANK = 256
LORA_ALPHA = LORA_RANK * 2
LORA_DROPOUT = 0.05

TARGET_MODS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PAD_LEFT = False  # LLaDA uses right padding
SEED = 42

# Validation disabled (kept as placeholders, not used)
VAL_RATIO = 0.0
VAL_STEP = 0
EVAL_EVERY = 0


# ===================== DeepSpeed config =====================
DS_CONFIG = {
    "bf16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": LR, "betas": [0.9, 0.95], "eps": EPS, "weight_decay": 0.0},
    },
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_ratio": 0,
            "warmup_num_steps": int(0.1 * TOTAL_TRAIN_STEPS),
            "total_num_steps": int(TOTAL_TRAIN_STEPS),
            "cos_min_ratio": 0.01,
        },
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "none"},
        "gather_16bit_weights_on_model_save": False,
    },
    "gradient_accumulation_steps": GRAD_ACC,
    "train_micro_batch_size_per_gpu": MICRO_BSIZE,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": False,
}
json.dump(DS_CONFIG, open("ds_zero2.json", "w"), indent=2)


# ===================== Utilities =====================
def set_seed(s: int) -> None:
    """Set all related random seeds for reproducibility."""
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def two_phase_mask_at_block(
    ids: List[int],
    plen: int,
    blk_idx: int,
    blk_size: int,
    mask_id: int,
    prefix_mask_rate: float = 0.0,
    suffix_mask_prob: float = 1.0,
) -> Tuple[List[int], List[bool]]:
    """
    Two-phase masking with prefix/suffix ablations.

    Layout:
      - [0, plen): prompt is visible
      - Answer region [plen, end):
          • Prefix [plen, s0): random masking with prob=prefix_mask_rate (excluded from loss)
          • Active block [s0, e0): random masking with prob ~ U(MASK_PROB_EPS, 1.0) (included in loss)
          • Suffix (e0, end): random masking with prob=suffix_mask_prob (excluded from loss)

    Returns:
      noisy_ids: masked tokens
      mask_flags: boolean flags (True -> include in cross-entropy loss)
    """
    L = len(ids)
    ans_s = plen
    ans_e = L

    s0 = ans_s + blk_idx * blk_size
    e0 = min(ans_s + (blk_idx + 1) * blk_size, ans_e)

    noisy = ids.copy()
    mflag = [False] * L

    # Prefix noise on [ans_s, s0), excluded from loss
    if ans_s < ans_e:
        prefix_s = ans_s
        prefix_e = max(min(s0, ans_e), ans_s)
        if prefix_mask_rate > 0.0 and prefix_e > prefix_s:
            for pos in range(prefix_s, prefix_e):
                if random.random() < prefix_mask_rate:
                    noisy[pos] = mask_id  # excluded from loss

    # If the active block falls beyond the answer, we still allow suffix masking
    if s0 >= ans_e:
        e0 = ans_e
    else:
        # Active block: random masking; positions contribute to the loss
        p = random.uniform(MASK_PROB_EPS, 1.0)
        for pos in range(max(s0, ans_s), e0):
            if random.random() < p:
                noisy[pos] = mask_id
                mflag[pos] = True

    # Suffix: mask with suffix_mask_prob, excluded from loss
    if suffix_mask_prob >= 1.0 - 1e-12:
        for pos in range(e0, ans_e):
            noisy[pos] = mask_id
    elif suffix_mask_prob <= 1e-12:
        pass  # fully visible
    else:
        for pos in range(e0, ans_e):
            if random.random() < suffix_mask_prob:
                noisy[pos] = mask_id

    return noisy, mflag


def load_and_filter_examples(pt_path: str, max_samples: int):
    """
    Load examples and drop those without answer tokens.
    Returns a list[dict] each containing 'full_ids' and 'prompt_len'.
    """
    raw = torch.load(pt_path, map_location="cpu")
    if max_samples:
        raw = raw[:max_samples]
    out = []
    for ex in raw:
        plen = int(ex["prompt_len"])
        total = len(ex["full_ids"])
        if total - plen > 0:  # keep samples with at least one answer token
            out.append(ex)
    return out


class RandomBlockDataset(Dataset):
    """
    If mutate=True (training):
      - After sampling, the chosen block index is removed from the available set.
      - 'remaining' counts the total number of available blocks across all samples
        (used for early stop when all blocks are consumed).
    """

    def __init__(
        self,
        examples,
        rank: int,
        world: int,
        blk_size: int,
        max_len: int,
        mask_id: int,
        pad_id: int,
        mutate: bool = True,
    ):
        # Shard data to ensure no overlap across ranks
        sharded = examples[rank::world]
        self.data = sharded
        self.blk_size = blk_size
        self.max_len = max_len
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.mutate = mutate

        # Build available-block sets per sample
        self.sample_blks = []
        for ex in self.data:
            ids_raw = ex["full_ids"]
            if isinstance(ids_raw, torch.Tensor):
                ids_raw = ids_raw.tolist()
            plen = int(ex["prompt_len"])
            ans_len = max(len(ids_raw[:self.max_len]) - plen, 0)
            n_blks = max(math.ceil(ans_len / self.blk_size), 0)
            self.sample_blks.append(set(range(n_blks)))

        # 'remaining' is only used for training to support early stop
        self.remaining = sum(len(s) for s in self.sample_blks) if mutate else 0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        ids_raw = sample["full_ids"]
        if isinstance(ids_raw, torch.Tensor):
            ids_raw = ids_raw.tolist()

        # Truncate to max length
        ids = ids_raw[:self.max_len]
        plen = min(int(sample["prompt_len"]), len(ids))

        avail = self.sample_blks[idx]
        if len(avail) == 0:
            noisy, mask = ids.copy(), [False] * len(ids)
            blk_idx = -1
        else:
            blk_idx = random.choice(tuple(avail))
            if self.mutate:
                avail.remove(blk_idx)
                self.remaining -= 1

            noisy, mask = two_phase_mask_at_block(
                ids,
                plen,
                blk_idx,
                self.blk_size,
                self.mask_id,
                prefix_mask_rate=PREFIX_MASK_RATE,
                suffix_mask_prob=SUFFIX_MASK_PROB,
            )

        # Right padding
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [self.pad_id] * pad
            noisy += [self.pad_id] * pad
            mask += [False] * pad

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "noisy_ids": torch.tensor(noisy, dtype=torch.long),
            "mask_flags": torch.tensor(mask, dtype=torch.bool),
            "blk_index": torch.tensor(blk_idx, dtype=torch.long),
            "prompt_len": torch.tensor(plen, dtype=torch.long),
        }


def collate(batch):
    """Simple collator that stacks all tensor fields."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ===================== Saving helpers =====================
def _unique_mkdir(path: str) -> str:
    """
    Create a unique directory. If it exists, append -v2/-v3/... automatically.
    Returns the final directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        return path
    base = path
    i = 2
    while True:
        cand = f"{base}-v{i}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=False)
            return cand
        i += 1


@torch.no_grad()
def save_full_merged(
    engine,
    tok: AutoTokenizer,
    cfg: AutoConfig,
    out_dir: str,
    model_name: str,
    rank: int,
) -> None:
    """
    Execute only on rank==0:

    1) Save tokenizer/config.
    2) Merge LoRA into a CPU base model to avoid meta tensors
       (low_cpu_mem_usage=False and force .to('cpu')).
    3) Save a HuggingFace checkpoint (safetensors) identical in format to the final save.
    """
    if rank != 0:
        return
    out_dir = _unique_mkdir(out_dir)

    # (1) tokenizer / config
    tok.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)

    # (2) temporarily save current LoRA adapter; then merge on a CPU base
    tmp_adapter = tempfile.mkdtemp(prefix="lora_tmp_")
    try:
        # Save current LoRA adapter (usually small; not sharded)
        engine.module.save_pretrained(tmp_adapter)

        # Build CPU base and load adapter
        # Important: low_cpu_mem_usage=False + .to("cpu") to avoid meta tensors
        base_cpu = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        base_cpu.to("cpu")

        peft_cpu = PeftModel.from_pretrained(base_cpu, tmp_adapter, is_trainable=False)
        merged = peft_cpu.merge_and_unload()
        merged.to("cpu")

        # (3) save (safetensors)
        merged.save_pretrained(out_dir, safe_serialization=True)
    finally:
        shutil.rmtree(tmp_adapter, ignore_errors=True)


# ===================== Main =====================
def main() -> None:
    set_seed(SEED)
    deepspeed.init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left" if PAD_LEFT else "right"
    pad_id = tok.pad_token_id

    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    model.print_trainable_parameters()

    # NOTE: tokenizer's special <mask> id for this data pipeline
    MASK_ID = 126336

    # Training set only (validation disabled)
    all_examples = load_and_filter_examples(DATA_PT, MAX_SAMPLES)
    train_examples = all_examples

    ds_train = RandomBlockDataset(
        train_examples,
        rank,
        world,
        BLOCK_SIZE,
        MAX_LEN,
        MASK_ID,
        pad_id,
        mutate=True,
    )

    # No validation DataLoader
    val_loader = None  # placeholder for clarity

    trainable = [p for p in model.parameters() if p.requires_grad]
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=trainable, config_params=DS_CONFIG
    )

    IGNORE = -100
    final_dim = None  # lazily determined from first forward pass

    # DataLoader: shuffle; num_workers=0 to keep 'remaining' count correct
    loader = DataLoader(
        ds_train,
        batch_size=MICRO_BSIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    # ---- Logging: write (step, loss) to CSV ----
    if rank == 0:
        log_dir = _unique_mkdir(f"{CKPT_DIR}_logs")
        log_path = os.path.join(log_dir, "train_loss.csv")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
        print(f"[log] training curve will be written to: {log_path}")

        # Save ablation config for reproducibility
        with open(
            os.path.join(log_dir, "ablation_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "PREFIX_MASK_RATE": PREFIX_MASK_RATE,
                    "SUFFIX_MASK_PROB": SUFFIX_MASK_PROB,
                    "BLOCK_SIZE": BLOCK_SIZE,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[info] train={len(ds_train)} batches/epoch={len(loader)} "
            f"target_steps={TOTAL_TRAIN_STEPS}"
        )

    global_step = 0  # count steps only on gradient-accumulation boundaries
    end_sft = False

    for ep in range(EPOCHS):
        if end_sft:
            break

        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"epoch-{ep + 1}/{EPOCHS}")

        for batch in loader:
            batch = {
                k: v.to(engine.device, non_blocking=True) for k, v in batch.items()
            }

            logits = engine(batch["noisy_ids"]).logits  # (B, L, V)
            if final_dim is None:
                final_dim = logits.size(-1)

            labels = batch["input_ids"].clone()
            labels[~batch["mask_flags"]] = IGNORE
            labels[labels >= final_dim] = IGNORE  # safety check

            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, final_dim),
                labels.view(-1),
                ignore_index=IGNORE,
                reduction="none",
            ).view(labels.shape)

            masked_cnt = batch["mask_flags"].sum(-1).clamp(min=1)
            loss = (ce.sum(-1) / masked_cnt).mean()

            engine.backward(loss)
            engine.step()

            # Only log/step on accumulation boundaries
            if engine.is_gradient_accumulation_boundary():
                global_step += 1

                if rank == 0:
                    # progress bar info
                    with torch.no_grad():
                        blk_idx = batch["blk_index"].detach().cpu()
                        unique_k = sorted(set(blk_idx.tolist()))
                        k_info = ",".join(map(str, unique_k[:8]))
                        if len(unique_k) > 8:
                            k_info += "..."
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        k=bytes(k_info, "utf-8").decode("utf-8"),
                        step=global_step,
                    )
                    pbar.update(1)

                    # append (step, loss) to CSV
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{global_step},{loss.item():.8f}\n")

                # Stop once target steps are reached
                if global_step >= TOTAL_TRAIN_STEPS:
                    if rank == 0:
                        print(f"\n[info] reached target steps {TOTAL_TRAIN_STEPS}")
                    end_sft = True

                # Early stop if all blocks are consumed (rare if dataset is large enough)
                rem_local = torch.tensor(
                    ds_train.remaining, dtype=torch.long, device=engine.device
                )
                dist.all_reduce(rem_local, op=dist.ReduceOp.SUM)
                if rem_local.item() == 0:
                    if rank == 0:
                        print("\n[info] all blocks have been used — finishing training")
                    end_sft = True

                # Save checkpoint (full merged) at the configured interval
                if global_step % SAVE_EVERY == 0:
                    save_dir = f"{CKPT_DIR}_step{global_step:04d}"
                    if rank == 0:
                        print(f"\n[save@{global_step}] writing to {save_dir} ...")
                    save_full_merged(engine, tok, cfg, save_dir, MODEL_NAME, rank)

                if end_sft:
                    break

        if rank == 0:
            pbar.close()
        if end_sft:
            break


if __name__ == "__main__":
    main()
