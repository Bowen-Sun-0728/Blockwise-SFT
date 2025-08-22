#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a tokenized SFT dataset (.pt) from MetaMathQA using the model's
chat template.

Notes:
- 'prompt_len' counts tokens of the user-only prompt (per chat template).
- 'full_ids' contains the entire conversation tokens (prompt + answer),
  produced from the chat template (with a small trailing-text workaround).
- Samples longer than --max_len are skipped.
"""

import argparse
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Workaround for a tokenizer-config quirk:
# Some chat templates append trailing assistant header/end text that should not
# be part of the training sequence. The original code sliced off 47 characters.
# We preserve this behavior as a conservative fix.
TAIL_CHARS_TO_STRIP = 47


def build_ids(
    tokenizer: AutoTokenizer, prompt_text: str, answer_text: str
) -> Tuple[torch.Tensor, int]:
    """
    Build token IDs for a (prompt, answer) pair using the chat template.

    Returns:
        full_ids (torch.LongTensor): token IDs for the full conversation.
        prompt_len (int): number of tokens for the user-only prompt sequence.
    """
    user_only = [{"role": "user", "content": prompt_text}]
    full_msgs = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": answer_text},
    ]

    # Render full conversation as text via chat template, then strip the tail to fix the bug of LLaDA's tokenizer.
    full_text = tokenizer.apply_chat_template(
        full_msgs,
        add_generation_prompt=False,
        tokenize=False,
        return_tensors=None,
    )
    if TAIL_CHARS_TO_STRIP > 0 and len(full_text) >= TAIL_CHARS_TO_STRIP:
        full_text = full_text[:-TAIL_CHARS_TO_STRIP]

    # Encode full conversation to ids (no extra special tokens).
    full_ids: List[int] = tokenizer.encode(full_text, add_special_tokens=False)

    # Tokenize user-only prompt via chat template to compute prompt_len.
    prompt_ids = tokenizer.apply_chat_template(
        user_only,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors=None,
    )
    # HF may return List[int] or List[List[int]] depending on tokenizer/setup.
    if isinstance(prompt_ids, list) and prompt_ids and isinstance(prompt_ids[0], list):
        prompt_len = len(prompt_ids[0])
    else:
        prompt_len = len(prompt_ids)

    # Safety: ensure prompt_len does not exceed the (possibly truncated) full_ids.
    prompt_len = min(prompt_len, len(full_ids))

    return torch.tensor(full_ids, dtype=torch.long), int(prompt_len)


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    ds = load_dataset(args.dataset, split=args.split)

    out_list = []
    count = 0

    for ex in tqdm(ds, desc="processing"):
        # Optional filter by substring in example["type"] (kept from original behavior).
        if args.filter_type and isinstance(ex.get("type"), str):
            if args.filter_type in ex["type"]:
                continue

        raw_prompt = ex["query"]
        prompt = (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n"
            f"### Instruction:\n{raw_prompt}\n\n"
            "### Response: Let's think step by step."
        )

        response = ex["response"]

        full_ids, prompt_len = build_ids(tokenizer, prompt, response)

        # Skip long samples (training scripts expect <= max_len).
        if len(full_ids) > args.max_len:
            continue

        out_list.append({"prompt_len": prompt_len, "full_ids": full_ids})
        count += 1

        if args.max_samples and count >= args.max_samples:
            break

    torch.save(out_list, args.output)
    print(f"Saved {len(out_list)} samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT dataset .pt from MetaMathQA")
    parser.add_argument("--output", type=str, default="MathQA_256.pt")
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="meta-math/MetaMathQA")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--max_len", type=int, default=256, help="Skip samples longer than this length"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20_000,
        help="Maximum number of samples to keep (0 = unlimited)",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        default="GSM",
        help='Skip samples whose ex["type"] contains this substring ("" disables)',
    )
    args = parser.parse_args()
    main(args)
