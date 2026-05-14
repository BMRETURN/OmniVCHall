import importlib
from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList

from vcd_new.utils import (
    PatchProcessor,
    VCDLogitsProcessor,
    load_embeddings,
    load_qa_data,
    read_video,
    transform_pixel_to_patch,
)


def _build_vqa_prompt(question, options=None):
    if options is not None:
        options_str = "\n".join(options)
        return f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下多项选择题。

要求：你的回答只能从给定的选项中选择（A, B, C），不要包含其他内容

问题：{question}
选项：
{options_str}
"""
    return f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下问题。

要求：你的回答只能是\"yes\"或\"no\"，不要包含其他内容

问题：{question}
"""


def _extract_generation_step_logits(generated_ids, input_ids=None):
    if getattr(generated_ids, "scores", None) is None:
        return []
    step_logits = []
    for score_t in generated_ids.scores:
        if score_t.dim() == 2:
            step_logits.append(score_t[0:1])
        elif score_t.dim() == 1:
            step_logits.append(score_t.unsqueeze(0))
        else:
            step_logits.append(score_t.reshape(1, -1))
    return step_logits


@lru_cache(maxsize=16)
def _load_videochat_runtime_symbols(module_name: str):
    mod = importlib.import_module(module_name)
    return {
        "conv_templates": getattr(mod, "conv_templates"),
        "tokenizer_image_token": getattr(mod, "tokenizer_image_token"),
        "DEFAULT_IMAGE_TOKEN": getattr(mod, "DEFAULT_IMAGE_TOKEN"),
        "IMAGE_TOKEN_INDEX": getattr(mod, "IMAGE_TOKEN_INDEX"),
    }


def _ensure_tokenizer_pad_token(tokenizer):
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return
    if "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643


def _sample_video_frames(
    video_frames: List[np.ndarray],
    max_num_frames: int,
    min_num_frames: int = 64,
):
    if not video_frames:
        raise ValueError("Empty video frames for VideoChat input")

    n = len(video_frames)
    cap = n if int(max_num_frames) <= 0 else min(n, int(max_num_frames))
    if n >= min_num_frames:
        cap = max(min_num_frames, cap)
    cap = max(1, cap)
    cap = max(4, int((cap // 4) * 4))

    if n == 1:
        indices = np.zeros((cap,), dtype=np.int64)
    elif cap == 1:
        indices = np.array([0], dtype=np.int64)
    else:
        indices = np.linspace(0, n - 1, cap).round().astype(np.int64)

    sampled_frames = np.stack([video_frames[int(i)] for i in indices.tolist()], axis=0)
    time_msg = f"\n{len(indices)} frames are uniformly sampled from the video."
    return sampled_frames, time_msg


def _prepare_multimodal_inputs(
    model,
    tokenizer,
    video_frames: List[np.ndarray],
    question: str,
    options=None,
    max_num_frames: int = 64,
):
    runtime = _load_videochat_runtime_symbols(model.__class__.__module__)
    sampled_frames, time_msg = _sample_video_frames(
        video_frames=video_frames,
        max_num_frames=max_num_frames,
        min_num_frames=64,
    )

    image_sizes = [sampled_frames[0].shape[:2]]
    vision_tower = model.get_vision_tower()
    pixel_values = vision_tower.image_processor.preprocess(
        sampled_frames,
        return_tensors="pt",
    )["pixel_values"]

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    images = [pixel_values.to(device=model_device, dtype=model_dtype)]

    conv = runtime["conv_templates"]["qwen_2"].copy()
    user_prompt = (
        f'{runtime["DEFAULT_IMAGE_TOKEN"]}\n'
        f"{time_msg.strip()} {_build_vqa_prompt(question, options)}"
    )
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = runtime["tokenizer_image_token"](
        prompt,
        tokenizer,
        runtime["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0)
    input_ids = input_ids.to(model_device)

    _ensure_tokenizer_pad_token(tokenizer)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(model_device)
    return input_ids, attention_mask, images, image_sizes


def _generate_videochat(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    images,
    image_sizes,
    max_new_tokens,
    logits_processor=None,
):
    kwargs = dict(
        inputs=input_ids,
        images=images,
        attention_mask=attention_mask,
        modalities=["video"],
        image_sizes=image_sizes,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor

    return model.generate(**kwargs)


def _align_patch_weights(patch_weights: torch.Tensor, target_len: int, ref_tensor: torch.Tensor):
    w = patch_weights.to(device=ref_tensor.device, dtype=ref_tensor.dtype).reshape(-1)
    if w.numel() == 0:
        return torch.ones(target_len, device=ref_tensor.device, dtype=ref_tensor.dtype)
    if w.shape[0] != target_len:
        w = F.interpolate(
            w.reshape(1, 1, -1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).reshape(-1)
    return w


def answer_question_original(
    model,
    processor,
    video_path,
    question,
    options=None,
    max_new_tokens=8,
    video_frames: Optional[List[np.ndarray]] = None,
    max_num_frames: int = 64,
):
    tokenizer = processor
    if video_frames is None:
        video_frames = read_video(video_path)

    input_ids, attention_mask, images, image_sizes = _prepare_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        video_frames=video_frames,
        question=question,
        options=options,
        max_num_frames=max_num_frames,
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            modalities=["video"],
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )
        generated_ids = _generate_videochat(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_sizes=image_sizes,
            max_new_tokens=max_new_tokens,
        )

    generated_logits = _extract_generation_step_logits(generated_ids, input_ids)
    last_hidden_states = outputs.hidden_states[-1]
    return generated_logits, last_hidden_states[0], input_ids


def answer_question_negative(
    model,
    processor,
    negative_video_path,
    question,
    options=None,
    max_new_tokens=8,
    video_frames: Optional[List[np.ndarray]] = None,
    max_num_frames: int = 64,
):
    tokenizer = processor
    if video_frames is None:
        video_frames = read_video(negative_video_path)

    input_ids, attention_mask, images, image_sizes = _prepare_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        video_frames=video_frames,
        question=question,
        options=options,
        max_num_frames=max_num_frames,
    )

    with torch.inference_mode():
        generated_ids = _generate_videochat(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_sizes=image_sizes,
            max_new_tokens=max_new_tokens,
        )

    generated_logits = _extract_generation_step_logits(generated_ids, input_ids)
    return generated_logits


def answer_question_positive(
    model,
    processor,
    video_path,
    patch_weights,
    question,
    original_logits,
    negative_logits,
    options=None,
    max_new_tokens=8,
    video_frames: Optional[List[np.ndarray]] = None,
    max_num_frames: int = 64,
):
    tokenizer = processor
    if video_frames is None:
        video_frames = read_video(video_path)

    input_ids, attention_mask, images, image_sizes = _prepare_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        video_frames=video_frames,
        question=question,
        options=options,
        max_num_frames=max_num_frames,
    )

    def projector_forward_hook(module, inputs, output):
        if not torch.is_tensor(output):
            return output

        out = output
        squeezed = False
        if out.dim() == 2:
            out = out.unsqueeze(0)
            squeezed = True
        if out.dim() != 3:
            return output

        token_count = int(out.shape[1])
        w = _align_patch_weights(patch_weights, token_count, out)
        out = out * w.view(1, token_count, 1)

        if squeezed:
            out = out.squeeze(0)
        return out

    projector = model.get_model().mm_projector
    hook_handle = projector.register_forward_hook(projector_forward_hook)

    try:
        vcd_processor = VCDLogitsProcessor(
            original_logits_list=original_logits,
            negative_logits_list=negative_logits,
            alpha=1.0,
        )

        with torch.inference_mode():
            generated_ids = _generate_videochat(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([vcd_processor]),
            )
    finally:
        hook_handle.remove()

    seq = generated_ids.sequences[0]
    input_len = int(input_ids.shape[1])
    if seq.shape[0] > input_len:
        new_tokens = seq[input_len:]
    else:
        new_tokens = seq
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return generated_text, generated_ids
