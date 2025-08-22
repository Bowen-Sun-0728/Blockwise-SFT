#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-sequence SFT (random answer masking over the whole answer span)

Behavior:
- No validation: the validation set is disabled.
- Train for exactly TOTAL_TRAIN_STEPS (counted on grad-acc boundaries).
- Save a fully merged HF checkpoint every SAVE_EVERY steps (unique dir to avoid
  name clashes), identical to the final checkpoint format.
- Append (step, loss) to a CSV file to plot the training curve later.
"""

import json
import os
import random
import shutil
import tempfile
from typing import List, Optional

import deepspeed
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ===================== Hyperparameters =====================
CKPT_DIR = "ckpts_fullseq_sft"
DATA_PT = "MathQA_256.pt"
MAX_SAMPLES = 300 * 32
VAL_SAMPLES = 0  # validation disabled
MAX_LEN = 256

MASK_PROB_EPS = 1e-3  # per-step masking rate p ~ U(eps, 1)
MICRO_BSIZE = 32
GRAD_ACC = 1
LR, EPS = 1e-5, 1e-8

EPOCHS = 1
# Use explicit int casting to avoid float drift
TOTAL_TRAIN_STEPS = int(MAX_SAMPLES * EPOCHS / MICRO_BSIZE)
SAVE_EVERY = int(TOTAL_TRAIN_STEPS)

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336  # special <mask> id used by the dataset/pipeline

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
    """Set seeds for reproducibility."""
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def random_mask_answer(ids: List[int], plen: int, mask_id: int):
    """
    Randomly mask tokens in the answer span [plen, len(ids)).

    Returns:
        noisy_ids: list[int] with masked tokens
        mask_flags: list[bool], True indicates positions included in the CE loss
    """
    p = random.uniform(MASK_PROB_EPS, 1.0)
    noisy, flags = ids.copy(), [False] * len(ids)
    for pos in range(plen, len(ids)):
        if random.random() < p:
            noisy[pos] = mask_id
            flags[pos] = True
    return noisy, flags


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
    Rank-0 only: save a full, merged HuggingFace checkpoint.

    Steps:
      1) Save tokenizer/config.
      2) Merge LoRA into a CPU base model (avoid meta tensors).
      3) Save in safetensors format, identical to the final save layout.
    """
    if rank != 0:
        return
    out_dir = _unique_mkdir(out_dir)

    # (1) tokenizer / config
    tok.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)

    # (2) save current LoRA adapter and merge on CPU
    tmp_adapter = tempfile.mkdtemp(prefix="lora_tmp_")
    try:
        engine.module.save_pretrained(tmp_adapter)

        # Important: low_cpu_mem_usage=False + force .to('cpu') to avoid meta tensors
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
        merged.save_pretrained(out_dir, safe_serialization=True)
    finally:
        shutil.rmtree(tmp_adapter, ignore_errors=True)


# ===================== Dataset =====================
class FullSeqDataset(Dataset):
    """
    Load samples from a .pt file and shard them across ranks.
    Each sample is truncated to max_len, then answer tokens are randomly masked.
    """

    def __init__(
        self,
        pt: str,
        max_samples: Optional[int],
        offset: int,
        rank: int,
        world: int,
        max_len: int,
        mask_id: int,
        pad_id: int,
    ):
        data = torch.load(pt, map_location="cpu")[offset:]
        if max_samples:
            data = data[:max_samples]
        self.samples = data[rank::world]
        self.max_len, self.mask_id, self.pad_id = max_len, mask_id, pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        ids_raw = s["full_ids"]
        if isinstance(ids_raw, torch.Tensor):
            ids_raw = ids_raw.tolist()

        # Truncate and clip prompt length to the truncated length
        ids = ids_raw[:self.max_len]
        plen = min(int(s["prompt_len"]), len(ids))

        noisy, mflag = random_mask_answer(ids, plen, self.mask_id)

        # Right padding
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [self.pad_id] * pad
            noisy += [self.pad_id] * pad
            mflag += [False] * pad

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "noisy_ids": torch.tensor(noisy, dtype=torch.long),
            "mask_flags": torch.tensor(mflag, dtype=torch.bool),
        }


def collate(batch):
    """Stack all tensor fields into a batch dict."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


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

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    model.print_trainable_parameters()

    # ---------- Dataset (train only; validation disabled) ----------
    train_ds = FullSeqDataset(
        DATA_PT,
        MAX_SAMPLES,
        offset=VAL_SAMPLES,  # 0
        rank=rank,
        world=world,
        max_len=MAX_LEN,
        mask_id=MASK_ID,
        pad_id=pad_id,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=MICRO_BSIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    # ---------- DeepSpeed ----------
    trainable = [p for p in model.parameters() if p.requires_grad]
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=trainable, config_params=DS_CONFIG
    )

    IGNORE = -100
    final_dim: Optional[int] = None

    # ---- Logging: write (step, loss) to CSV ----
    if rank == 0:
        log_dir = _unique_mkdir(f"{CKPT_DIR}_logs")
        log_path = os.path.join(log_dir, "train_loss.csv")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
        print(f"[log] training curve will be written to: {log_path}")
        print(
            f"[info] train={len(train_ds)} val=0 "
            f"batches/epoch={len(train_dl)} target_steps={TOTAL_TRAIN_STEPS}"
        )

    # ---------- Training loop ----------
    global_step = 0
    end_sft = False
    for ep in range(EPOCHS):
        if end_sft:
            break

        if rank == 0:
            pbar = tqdm(total=len(train_dl), desc=f"epoch-{ep + 1}/{EPOCHS}")

        for batch in train_dl:
            batch = {k: v.to(engine.device, non_blocking=True) for k, v in batch.items()}

            logits = engine(batch["noisy_ids"]).logits
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

            # Count steps only on grad-acc boundaries
            if engine.is_gradient_accumulation_boundary():
                global_step += 1

                if rank == 0:
                    pbar.set_postfix(train_loss=f"{loss.item():.4f}", step=global_step)
                    pbar.update(1)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{global_step},{loss.item():.8f}\n")

                # Stop once target steps are reached
                if global_step >= TOTAL_TRAIN_STEPS:
                    end_sft = True

                # Save checkpoint (full merged) at configured interval
                if global_step % SAVE_EVERY == 0:
                    save_dir = f"{CKPT_DIR}_step{global_step:04d}"
                    if rank == 0:
                        print(f"\n[save@{global_step}] writing to {save_dir} ...")
                    save_full_merged(engine, tok, cfg, save_dir, MODEL_NAME, rank)

                if end_sft:
                    break

        if rank == 0:
            pbar.close()


if __name__ == "__main__":
    main()
