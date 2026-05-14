import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import LogitsProcessorList

from vcd_new.utils import (
    PatchProcessor,
    VCDLogitsProcessor,
    load_embeddings,
    load_qa_data,
    read_video,
    save_video_to_temp,
    transform_pixel_to_patch,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _extract_generation_step_logits(generated_ids, input_ids):
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids.sequences)]
    if len(trimmed) == 0:
        return []
    if len(trimmed) != 1:
        raise ValueError(f"Expected batch size 1, got {len(trimmed)}")
    gen_len = len(trimmed[0])
    step_logits = []
    for t in range(min(gen_len, len(generated_ids.scores))):
        score_t = generated_ids.scores[t]
        step_logits.append(score_t[0:1])
    return step_logits


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(orig_height, 1)

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _frames_to_pixel_values(
    frames: List[np.ndarray],
    num_segments: int = 8,
    max_num_tiles: int = 1,
    image_size: int = 448,
):
    if not frames:
        raise ValueError("Empty video frames for InternVL input")

    n = len(frames)
    seg = max(1, int(num_segments))
    if n == 1:
        frame_indices = np.array([0], dtype=np.int64)
    else:
        frame_indices = np.linspace(0, n - 1, seg).round().astype(np.int64)

    transform = _build_transform(image_size)
    pixel_values_list = []
    num_patches_list = []

    for idx in frame_indices.tolist():
        idx = max(0, min(int(idx), n - 1))
        img = Image.fromarray(frames[idx]).convert("RGB")
        tiles = _dynamic_preprocess(
            img,
            image_size=image_size,
            use_thumbnail=True,
            max_num=max_num_tiles,
        )
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(int(pixel_values.shape[0]))

    return torch.cat(pixel_values_list, dim=0), num_patches_list


def _build_internvl_query(model, tokenizer, prompt: str, num_patches_list: List[int]):
    video_prefix = "".join([f"Frame{i + 1}: <image>\\n" for i in range(len(num_patches_list))])
    full_question = video_prefix + prompt

    template = model.conv_template.copy()
    template.system_message = model.system_message
    template.append_message(template.roles[0], full_question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    return model_inputs, eos_token_id


def _prepare_multimodal_inputs(
    model,
    tokenizer,
    video_frames: List[np.ndarray],
    question: str,
    options=None,
    num_segments: int = 8,
    max_num_tiles: int = 1,
    image_size: int = 448,
):
    prompt = _build_vqa_prompt(question, options)
    pixel_values, num_patches_list = _frames_to_pixel_values(
        video_frames,
        num_segments=num_segments,
        max_num_tiles=max_num_tiles,
        image_size=image_size,
    )
    model_inputs, eos_token_id = _build_internvl_query(model, tokenizer, prompt, num_patches_list)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    model_dtype = next(model.parameters()).dtype
    pixel_values = pixel_values.to(device=model.device, dtype=model_dtype)
    return input_ids, attention_mask, pixel_values, eos_token_id


def _build_language_inputs_embeds(model, input_ids, visual_features):
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    bsz, seq_len, dim = input_embeds.shape

    flat_embeds = input_embeds.reshape(bsz * seq_len, dim)
    flat_ids = input_ids.reshape(bsz * seq_len)
    selected = torch.nonzero(flat_ids == model.img_context_token_id, as_tuple=False).squeeze(-1)

    visual_flat = visual_features.reshape(-1, dim).to(device=flat_embeds.device, dtype=flat_embeds.dtype)
    if selected.numel() > 0 and visual_flat.shape[0] > 0:
        n = min(int(selected.numel()), int(visual_flat.shape[0]))
        flat_embeds[selected[:n]] = visual_flat[:n]

    return flat_embeds.reshape(bsz, seq_len, dim)


def answer_question_original(
    model,
    processor,
    video_path,
    question,
    options=None,
    max_new_tokens=8,
    video_frames: Optional[List[np.ndarray]] = None,
    num_segments: int = 8,
    max_num_tiles: int = 1,
    image_size: int = 448,
):
    tokenizer = processor
    if video_frames is None:
        video_frames = read_video(video_path)

    input_ids, attention_mask, pixel_values, eos_token_id = _prepare_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        video_frames=video_frames,
        question=question,
        options=options,
        num_segments=num_segments,
        max_num_tiles=max_num_tiles,
        image_size=image_size,
    )

    with torch.inference_mode():
        vit_embeds = model.extract_feature(pixel_values)
        input_embeds = _build_language_inputs_embeds(model, input_ids, vit_embeds)

        outputs = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        generated_ids = model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
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
    num_segments: int = 8,
    max_num_tiles: int = 1,
    image_size: int = 448,
):
    tokenizer = processor
    if video_frames is None:
        video_frames = read_video(negative_video_path)

    input_ids, attention_mask, pixel_values, eos_token_id = _prepare_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        video_frames=video_frames,
        question=question,
        options=options,
        num_segments=num_segments,
        max_num_tiles=max_num_tiles,
        image_size=image_size,
    )

    with torch.inference_mode():
        vit_embeds = model.extract_feature(pixel_values)
        input_embeds = _build_language_inputs_embeds(model, input_ids, vit_embeds)
        generated_ids = model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
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
    num_segments: int = 8,
    max_num_tiles: int = 1,
    image_size: int = 448,
):
    tokenizer = processor
    if video_frames is None:
        video_frames = read_video(video_path)

    input_ids, attention_mask, pixel_values, eos_token_id = _prepare_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        video_frames=video_frames,
        question=question,
        options=options,
        num_segments=num_segments,
        max_num_tiles=max_num_tiles,
        image_size=image_size,
    )

    with torch.inference_mode():
        vit_embeds = model.extract_feature(pixel_values)
        flat_vit = vit_embeds.reshape(-1, vit_embeds.shape[-1])

        w = patch_weights.to(device=flat_vit.device, dtype=flat_vit.dtype).reshape(-1)
        if w.shape[0] != flat_vit.shape[0]:
            w = torch.nn.functional.interpolate(
                w.reshape(1, 1, -1),
                size=flat_vit.shape[0],
                mode="linear",
                align_corners=False,
            ).reshape(-1)

        weighted_vit = (flat_vit * w.unsqueeze(-1)).reshape_as(vit_embeds)
        pos_input_embeds = _build_language_inputs_embeds(model, input_ids, weighted_vit)

        vcd_processor = VCDLogitsProcessor(
            original_logits_list=original_logits,
            negative_logits_list=negative_logits,
            alpha=1.0,
        )

        generated_ids = model.language_model.generate(
            inputs_embeds=pos_input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([vcd_processor]),
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids.sequences[0][len(input_ids[0]):]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return generated_text, generated_ids
