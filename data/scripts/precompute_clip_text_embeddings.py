#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Precompute CLIP text embeddings for a text-annotated motion sidecar JSON.

This script extracts unique text prompts from a filtered BABEL-style sidecar JSON,
encodes each unique text exactly once with a CLIP text encoder, and saves:

1. A compact lookup table containing:
   - the ordered list of unique texts
   - a dense embedding matrix
   - text -> embedding index mapping
   - simple usage counts
2. Optionally, a copy of the sidecar JSON where each text-bearing segment is
   annotated with ``text_embedding_idx`` for fast lookup during training.

Typical use:

    python data/scripts/precompute_clip_text_embeddings.py \
        --input-json data/Babel/accad/accad_g1_text_full.filtered.json \
        --output-pt data/Babel/accad/accad_g1_text_clip.pt \
        --output-indexed-json data/Babel/accad/accad_g1_text_full.clip_indexed.json \
        --model-name openai/clip-vit-base-patch32 \
        --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, CLIPModel


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def iter_text_entries(sidecar: dict) -> Iterable[Tuple[str, str]]:
    """Yield ``(motion_idx, text)`` pairs in stable file order."""
    for motion_idx, meta in sidecar.items():
        segments = meta.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                text = str(segment.get("text", "")).strip()
                if text:
                    yield motion_idx, text
            continue

        text = str(meta.get("text", "")).strip()
        if text:
            yield motion_idx, text


def collect_unique_texts(sidecar: dict) -> Tuple[List[str], Dict[str, int], Counter]:
    texts_in_order: List[str] = []
    text_to_idx: Dict[str, int] = {}
    counts: Counter = Counter()

    for _, text in iter_text_entries(sidecar):
        counts[text] += 1
        if text not in text_to_idx:
            text_to_idx[text] = len(texts_in_order)
            texts_in_order.append(text)

    return texts_in_order, text_to_idx, counts


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def choose_model_dtype(device: str, requested_dtype: torch.dtype) -> Optional[torch.dtype]:
    if device == "cpu":
        return None
    if requested_dtype in (torch.float16, torch.bfloat16):
        return requested_dtype
    return None


def encode_texts(
    texts: List[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    normalize: bool,
    local_files_only: bool,
    requested_save_dtype: torch.dtype,
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0, dtype=requested_save_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )

    model_dtype = choose_model_dtype(device, requested_save_dtype)
    model = CLIPModel.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.to(device)

    all_features: List[torch.Tensor] = []
    with torch.inference_mode():
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            tokenized = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}

            features = model.get_text_features(**tokenized)
            if not torch.is_tensor(features):
                if hasattr(features, "text_embeds") and features.text_embeds is not None:
                    features = features.text_embeds
                elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
                    # Fall back to CLS-style first token if a pooled embedding is not exposed.
                    features = features.last_hidden_state[:, 0, :]
                else:
                    raise TypeError(
                        "Unsupported CLIP text feature output type: "
                        f"{type(features).__name__}"
                    )
            if normalize:
                features = torch.nn.functional.normalize(features, dim=-1)

            all_features.append(features.to("cpu", dtype=requested_save_dtype))

    return torch.cat(all_features, dim=0)


def attach_embedding_indices(
    sidecar: dict,
    text_to_idx: Dict[str, int],
) -> dict:
    indexed_sidecar = copy.deepcopy(sidecar)

    for meta in indexed_sidecar.values():
        segments = meta.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                text = str(segment.get("text", "")).strip()
                if text:
                    segment["text_embedding_idx"] = int(text_to_idx[text])
            continue

        text = str(meta.get("text", "")).strip()
        if text:
            meta["text_embedding_idx"] = int(text_to_idx[text])

    return indexed_sidecar


def build_output_payload(
    *,
    model_name: str,
    input_json: Path,
    texts: List[str],
    text_to_idx: Dict[str, int],
    text_counts: Counter,
    embeddings: torch.Tensor,
    normalize: bool,
) -> dict:
    return {
        "metadata": {
            "source_json": str(input_json),
            "model_name": model_name,
            "normalize": bool(normalize),
            "embedding_dim": int(embeddings.shape[1]),
            "num_unique_texts": len(texts),
            "num_labeled_segments": int(sum(text_counts.values())),
            "dtype": str(embeddings.dtype).replace("torch.", ""),
        },
        "texts": list(texts),
        "text_to_idx": dict(text_to_idx),
        "text_counts": {text: int(text_counts[text]) for text in texts},
        "embeddings": embeddings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute CLIP embeddings for unique texts in a motion sidecar JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Filtered BABEL-style sidecar JSON containing text segments.",
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        required=True,
        help="Output .pt file with the text embedding lookup table.",
    )
    parser.add_argument(
        "--output-indexed-json",
        type=Path,
        default=None,
        help=(
            "Optional output JSON that copies the input sidecar and annotates each "
            "text-bearing segment with text_embedding_idx."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for CLIP encoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type used when saving the embedding matrix.",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="Disable L2 normalization of CLIP text features before saving.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the CLIP model/tokenizer to exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sidecar = load_json(args.input_json)
    texts, text_to_idx, text_counts = collect_unique_texts(sidecar)
    if not texts:
        raise ValueError(f"No non-empty texts found in {args.input_json}")

    save_dtype = torch_dtype_from_name(args.save_dtype)
    normalize = not args.disable_normalize

    embeddings = encode_texts(
        texts,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize=normalize,
        local_files_only=args.local_files_only,
        requested_save_dtype=save_dtype,
    )

    output_payload = build_output_payload(
        model_name=args.model_name,
        input_json=args.input_json,
        texts=texts,
        text_to_idx=text_to_idx,
        text_counts=text_counts,
        embeddings=embeddings,
        normalize=normalize,
    )

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output_pt)

    if args.output_indexed_json is not None:
        indexed_sidecar = attach_embedding_indices(sidecar, text_to_idx)
        save_json(args.output_indexed_json, indexed_sidecar)

    print(f"Loaded sidecar: {args.input_json}")
    print(f"Unique texts: {len(texts)}")
    print(f"Labeled segments: {sum(text_counts.values())}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Saved embedding table: {args.output_pt}")
    if args.output_indexed_json is not None:
        print(f"Saved indexed sidecar: {args.output_indexed_json}")


if __name__ == "__main__":
    main()
