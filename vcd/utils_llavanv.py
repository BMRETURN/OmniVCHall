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


def _sample_video_frames(
    video_frames: List[np.ndarray],
    max_num_frames: int,
):
    if not video_frames:
        raise ValueError("Empty video frames for LLaVA-NeXT-Video input")

    n = len(video_frames)
    cap = n if int(max_num_frames) <= 0 else min(n, int(max_num_frames))
    cap = max(1, cap)

    if n == 1:
        indices = np.zeros((cap,), dtype=np.int64)
    elif cap == 1:
        indices = np.array([0], dtype=np.int64)
    else:
        indices = np.linspace(0, n - 1, cap).round().astype(np.int64)

    sampled_frames = np.stack([video_frames[int(i)] for i in indices.tolist()], axis=0)
    time_msg = f"\n{len(indices)} frames are uniformly sampled from the video."
    return sampled_frames, time_msg


def _build_prompt_with_video(processor, prompt_text: str):
    if hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    return f"USER: <video>\n{prompt_text}\nASSISTANT:"


def _move_inputs_to_model_device(model, inputs: dict):
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if torch.is_floating_point(v):
                moved[k] = v.to(device=model_device, dtype=model_dtype)
            else:
                moved[k] = v.to(device=model_device)
        else:
            moved[k] = v
    return moved


def _prepare_multimodal_inputs(
    model,
    processor,
    video_frames: List[np.ndarray],
    question: str,
    options=None,
    max_num_frames: int = 32,
):
    sampled_frames, time_msg = _sample_video_frames(
        video_frames=video_frames,
        max_num_frames=max_num_frames,
    )

    prompt_text = f"{time_msg.strip()} {_build_vqa_prompt(question, options)}"
    prompt = _build_prompt_with_video(processor, prompt_text)

    inputs = processor(
        text=prompt,
        videos=sampled_frames,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_model_device(model, inputs)
    return inputs


def _generate_llavanv(
    model,
    inputs,
    max_new_tokens,
    logits_processor=None,
):
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False,
        use_cache=True,
    )
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
    max_num_frames: int = 32,
):
    if video_frames is None:
        video_frames = read_video(video_path)

    inputs = _prepare_multimodal_inputs(
        model=model,
        processor=processor,
        video_frames=video_frames,
        question=question,
        options=options,
        max_num_frames=max_num_frames,
    )

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        generated_ids = _generate_llavanv(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_logits = _extract_generation_step_logits(generated_ids, inputs.get("input_ids"))
    last_hidden_states = outputs.hidden_states[-1]
    return generated_logits, last_hidden_states[0], inputs["input_ids"]


def answer_question_negative(
    model,
    processor,
    negative_video_path,
    question,
    options=None,
    max_new_tokens=8,
    video_frames: Optional[List[np.ndarray]] = None,
    max_num_frames: int = 32,
):
    if video_frames is None:
        video_frames = read_video(negative_video_path)

    inputs = _prepare_multimodal_inputs(
        model=model,
        processor=processor,
        video_frames=video_frames,
        question=question,
        options=options,
        max_num_frames=max_num_frames,
    )

    with torch.inference_mode():
        generated_ids = _generate_llavanv(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_logits = _extract_generation_step_logits(generated_ids, inputs.get("input_ids"))
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
    max_num_frames: int = 32,
):
    if video_frames is None:
        video_frames = read_video(video_path)

    inputs = _prepare_multimodal_inputs(
        model=model,
        processor=processor,
        video_frames=video_frames,
        question=question,
        options=options,
        max_num_frames=max_num_frames,
    )

    def projector_forward_hook(module, hook_inputs, output):
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output

        if not torch.is_tensor(out):
            return output

        squeezed = False
        if out.dim() == 2:
            out = out.unsqueeze(0)
            squeezed = True

        if out.dim() != 3:
            return output

        bsz, token_count, hid_dim = out.shape
        flat_count = int(bsz * token_count)
        w = _align_patch_weights(patch_weights, flat_count, out)

        out_flat = out.reshape(flat_count, hid_dim)
        out_flat = out_flat * w.view(flat_count, 1)
        out = out_flat.reshape(bsz, token_count, hid_dim)

        if squeezed:
            out = out.squeeze(0)

        if is_tuple:
            return (out,) + output[1:]
        return out

    projector = model.model.multi_modal_projector
    hook_handle = projector.register_forward_hook(projector_forward_hook)

    try:
        vcd_processor = VCDLogitsProcessor(
            original_logits_list=original_logits,
            negative_logits_list=negative_logits,
            alpha=1.0,
        )

        with torch.inference_mode():
            generated_ids = _generate_llavanv(
                model=model,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([vcd_processor]),
            )
    finally:
        hook_handle.remove()

    seq = generated_ids.sequences[0]
    input_len = int(inputs["input_ids"].shape[1])
    new_tokens = seq[input_len:] if seq.shape[0] > input_len else seq

    tokenizer = getattr(processor, "tokenizer", processor)
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return generated_text, generated_ids
