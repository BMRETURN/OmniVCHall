import argparse
import json
import pickle
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def generate_tool_descriptions(json_file_path="tools.json"):
    with open(json_file_path, "r", encoding="utf-8") as f:
        tools_data = json.load(f)

    template = "{type}干扰，具体是{description}。{explain}。"
    out = {}
    for tool_name, tool_info in tools_data.items():
        out[tool_name] = template.format(
            type=tool_info["type"],
            description=tool_info["description"],
            explain=tool_info["explain"],
        )
    return out


def generate_embeddings_with_vchatf(
    description_tools_mapping,
    model_path="/home/storage/wenbinxing/checkpoints/VideoChat-Flash-Qwen2-7B_res224",
    output_path="tools_embeddings_vchatf.pkl",
    batch_size=8,
    max_length=512,
    save_dtype="float32",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    text_encoder = model.get_model() if hasattr(model, "get_model") else model
    if not hasattr(text_encoder, "llm_compress_type"):
        text_encoder.llm_compress_type = "attention"
    if not hasattr(text_encoder, "llm_compress_layer_list"):
        text_encoder.llm_compress_layer_list = []
    if not hasattr(text_encoder, "llm_image_token_ratio_list"):
        text_encoder.llm_image_token_ratio_list = []
    if not hasattr(text_encoder, "image_token_posi"):
        text_encoder.image_token_posi = [-1]
    if not hasattr(text_encoder, "prompt_len"):
        text_encoder.prompt_len = None
    if not hasattr(text_encoder, "image_tokens"):
        text_encoder.image_tokens = [0]
    device = next(text_encoder.parameters()).device

    tool_names = list(description_tools_mapping.keys())
    texts = [description_tools_mapping[name] for name in tool_names]

    embeddings_dict = {}
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
            embedding_layer = model.get_input_embeddings()
            last_hidden = embedding_layer(enc["input_ids"])
            attn_mask = enc["attention_mask"]

        for i, name in enumerate(batch_names):
            valid_len = int(attn_mask[i].sum().item())
            emb = last_hidden[i, :valid_len, :].detach().cpu()
            emb = emb.to(torch.float16 if save_dtype == "float16" else torch.float32)
            embeddings_dict[name] = emb

        print(f"Encoded {end}/{len(texts)}", flush=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"Saved embeddings to: {output_path}")
    return embeddings_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tool embeddings with VideoChat-Flash language backbone.")
    parser.add_argument(
        "--tools_json",
        type=str,
        default=str(Path(__file__).resolve().parent / "tools.json"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/storage/wenbinxing/checkpoints/VideoChat-Flash-Qwen2-7B_res224",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "tools_embeddings_vchatf.pkl"),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "float16"])
    return parser.parse_args()


def main():
    args = parse_args()
    descriptions = generate_tool_descriptions(args.tools_json)
    generate_embeddings_with_vchatf(
        description_tools_mapping=descriptions,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_dtype=args.save_dtype,
    )


if __name__ == "__main__":
    main()
