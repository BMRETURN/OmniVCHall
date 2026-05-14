#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration


def generate_tool_descriptions(tools_json: Path) -> dict:
    with open(tools_json, "r", encoding="utf-8") as f:
        tools_data = json.load(f)

    template = "{type}干扰，具体是{description}。{explain}。"
    descriptions = {}
    for tool_name, tool_info in tools_data.items():
        descriptions[tool_name] = template.format(
            type=tool_info["type"],
            description=tool_info["description"],
            explain=tool_info["explain"],
        )
    return descriptions


def generate_embeddings(
    *,
    model_path: Path,
    tools_json: Path,
    output_path: Path,
    batch_size: int,
    max_length: int,
    save_dtype: str,
) -> None:
    descriptions = generate_tool_descriptions(tools_json)
    tool_names = list(descriptions.keys())
    texts = [descriptions[name] for name in tool_names]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device

    embeddings = {}
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_names = tool_names[start:end]
        batch_texts = texts[start:end]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]
            attn_mask = enc["attention_mask"]

        for idx, name in enumerate(batch_names):
            valid_len = int(attn_mask[idx].sum().item())
            emb = last_hidden[idx, :valid_len, :].detach().cpu()
            if save_dtype == "float16":
                emb = emb.to(torch.float16)
            else:
                emb = emb.to(torch.float32)
            embeddings[name] = emb

        print(f"Encoded {end}/{len(texts)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    first = next(iter(embeddings.values()))
    print(f"Saved embeddings to: {output_path}")
    print(f"Embedding dim: {first.shape[-1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen3-VL tool embeddings for VCD.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--tools_json",
        type=Path,
        default=Path(__file__).resolve().parent / "tools.json",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_embeddings(
        model_path=args.model_path,
        tools_json=args.tools_json,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_dtype=args.save_dtype,
    )


if __name__ == "__main__":
    main()
