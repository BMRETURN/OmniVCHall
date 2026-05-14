import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
import pickle
import json


def generate_tool_descriptions(json_file_path="tools.json"):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    # 模板：{type}干扰 - {description}。{explain}。
    template = "{type}干扰，具体是{description}。{explain}。"
    
    # 生成描述映射
    description_tools_mapping = {}
    
    for tool_name, tool_info in tools_data.items():
        # 使用模板生成完整描述
        full_description = template.format(
            type=tool_info["type"],
            description=tool_info["description"],
            explain=tool_info["explain"]
        )
        description_tools_mapping[tool_name] = full_description
    return description_tools_mapping


def generate_embeddings_with_qwen3vl(
    description_tools_mapping,
    model_path="../../checkpoints/Qwen3-VL-8B-Instruct",
    output_path="tools_embeddings_qwen3vl.pkl",
    batch_size=8,
    max_length=512,
    save_dtype="float32",  
):
    """
    使用 Qwen3-VL（语言端）对文本进行编码，保存每条文本的 token-level embeddings。

    返回:
        embeddings_dict: Dict[str, torch.Tensor]
            每个工具名 -> Tensor(shape=[seq_len, hidden_dim])。
    """
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Qwen3-VL-8B-Instruct：这里用 ConditionalGeneration 类，取 hidden_states 做文本 embeddings
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    tool_names = list(description_tools_mapping.keys())
    texts = [description_tools_mapping[name] for name in tool_names]

    embeddings_dict = {}

    # 逐批编码
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_names = tool_names[start:end]
        batch_texts = texts[start:end]

        # tokenize + padding
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        try:
            embed_device = model.get_input_embeddings().weight.device
        except Exception:
            embed_device = next(model.parameters()).device

        enc = {k: v.to(embed_device) for k, v in enc.items()}

        with torch.inference_mode():
            outputs = model(
                **enc,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, D]
            attn_mask = enc["attention_mask"]        # [B, L]

        # 按样本保存（去 padding）
        for i, name in enumerate(batch_names):
            valid_len = int(attn_mask[i].sum().item())
            emb = last_hidden[i, :valid_len, :].detach().cpu()

            if save_dtype == "float16":
                emb = emb.to(torch.float16)
            else:
                emb = emb.to(torch.float32)

            embeddings_dict[name] = emb  # shape: [seq_len, hidden_dim]

        print(f"Encoded {end}/{len(texts)}")

    # 直接保存到 pkl
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"Saved embeddings to: {output_path}")
    return embeddings_dict


def load_embeddings(embeddings_path):
    """
    从本地加载embeddings
    """
    with open(embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict


if __name__ == "__main__":
    description_tools_mapping = generate_tool_descriptions()

    # qwen3-vl生成embeddings
    embeddings_dict = generate_embeddings_with_qwen3vl(description_tools_mapping)

    output_path = "tools_embeddings_qwen3vl.pkl"
    # 验证加载
    loaded_embeddings = load_embeddings(output_path)
    print("Embeddings loaded successfully!")
    print(f"Embedding dimensions for 'ReverseVideo': {loaded_embeddings['ReverseVideo'].shape}")
    print(f"Embedding dimensions for 'SampleVideo': {loaded_embeddings['SampleVideo'].shape}")
    print(f"Embedding dimensions for 'ShuffleVideo': {loaded_embeddings['ShuffleVideo'].shape}")
    print(f"Embedding dimensions for 'BlurVideo': {loaded_embeddings['BlurVideo'].shape}")
    print(f"Embedding dimensions for 'NoiseVideo': {loaded_embeddings['NoiseVideo'].shape}")
    print(f"Embedding dimensions for 'HorizontalMirrorVideo': {loaded_embeddings['HorizontalMirrorVideo'].shape}")
    print(f"Embedding dimensions for 'VerticalMirrorVideo': {loaded_embeddings['VerticalMirrorVideo'].shape}")
    print(f"Embedding dimensions for 'GrayscaleVideo': {loaded_embeddings['GrayscaleVideo'].shape}")
    # Embedding dimensions for 'ReverseVideo': torch.Size([72, 4096])
    # Embedding dimensions for 'SampleVideo': torch.Size([46, 4096])
    # Embedding dimensions for 'ShuffleVideo': torch.Size([40, 4096])
    # Embedding dimensions for 'BlurVideo': torch.Size([48, 4096])
    # Embedding dimensions for 'NoiseVideo': torch.Size([53, 4096])
    # Embedding dimensions for 'HorizontalMirrorVideo': torch.Size([58, 4096])
    # Embedding dimensions for 'VerticalMirrorVideo': torch.Size([65, 4096])
    # Embedding dimensions for 'GrayscaleVideo': torch.Size([40, 4096])
    # max 72 tokens, min 40 tokens