import json
import os
import random
import re
import signal
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from vcd_new.tools.negative_tools import (
    BlurVideo,
    GrayscaleVideo,
    HorizontalMirrorVideo,
    NoiseVideo,
    ReverseVideo,
    SampleVideo,
    ShuffleVideo,
    VerticalMirrorVideo,
)
from vcd_new.tools.positive_tools import DINOv3SaliencyExtractor, MotionSaliencyExtractor
from vcd_new.utils import (
    PatchProcessor,
    load_embeddings,
    load_qa_data,
    read_video,
    transform_pixel_to_patch,
)


QUESTION_TYPES = ("s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
CACHE_STATS_KEYS = (
    "hit_v2",
    "hit_legacy",
    "miss",
    "saved_v2",
    "legacy_mismatch",
    "legacy_invalid",
    "v2_mismatch",
    "v2_invalid",
)
TIMING_KEYS = ("orig_gen", "tool_select", "neg_branch", "saliency", "pos_gen", "total")


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_answer(text):
    m = re.search(r"\b(a|b|c|d|yes|no)\b", str(text).lower())
    return m.group(1) if m else str(text).lower().strip()


def parse_qa_item(item):
    if "s_ynqa_id" in item:
        return "s_ynqa", int(item["s_ynqa_id"]), item["yn_question"], item["yn_answer"], None
    if "m_ynqa_id" in item:
        return "m_ynqa", int(item["m_ynqa_id"]), item["yn_question"], item["yn_answer"], None
    if "s_mcqa_id" in item:
        return "s_mcqa", int(item["s_mcqa_id"]), item["mc_question"], item["mc_answer"], item.get("mc_option")
    if "m_mcqa_id" in item:
        return "m_mcqa", int(item["m_mcqa_id"]), item["mc_question"], item["mc_answer"], item.get("mc_option")
    return None


def qa_uid(item) -> Optional[str]:
    parsed = parse_qa_item(item)
    if parsed is None:
        return None
    q_type, qa_id, _, _, _ = parsed
    return f"{q_type}:{qa_id}:{item.get('video_id', '')}"


def apply_limit(data, limit):
    if limit is None or int(limit) < 0:
        return data
    return data[: int(limit)]


def subset_by_type(data: List[dict], per_type: int):
    if per_type <= 0:
        return data
    buckets = {k: [] for k in QUESTION_TYPES}
    for item in data:
        parsed = parse_qa_item(item)
        if parsed is None:
            continue
        q_type = parsed[0]
        if len(buckets[q_type]) < per_type:
            buckets[q_type].append(item)
    out = []
    for q in QUESTION_TYPES:
        out.extend(buckets[q])
    return out


def load_train_subset_manifest(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    out = []
    for row in items:
        if isinstance(row, dict) and "item" in row:
            out.append(row["item"])
        else:
            out.append(row)
    return out, data


def build_video_index(video_dir: Path):
    index = {}
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            index[p.stem] = str(p)
    return index


def resolve_video_path(video_id: str, video_dir: Path, video_index, use_video_index: bool):
    if use_video_index and video_index is not None:
        return video_index.get(video_id)
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    return None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.data = OrderedDict()

    def get(self, key):
        if self.capacity <= 0:
            return None
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def build_cache_key(split_name: str, q_type: str, qa_id: int, video_id: str):
    return f"{split_name}_{q_type}_{qa_id}_{video_id}"


def _load_cache_tensor(path: Path, device, expected_video_id: Optional[str]):
    if not path.exists():
        return None, "missing"
    try:
        data = torch.load(path, map_location="cpu")
        cached_video_id = data.get("video_id")
        if (
            expected_video_id is not None
            and cached_video_id is not None
            and str(cached_video_id) != str(expected_video_id)
        ):
            return None, "video_mismatch"
        return {
            "w_m": data["w_m"].to(device=device, dtype=torch.float32),
            "w_v": data["w_v"].to(device=device, dtype=torch.float32),
            "video_id": cached_video_id,
        }, "ok"
    except Exception:
        return None, "invalid"


def get_cached_saliency(
    split_name: str,
    q_type: str,
    qa_id: int,
    video_id: str,
    cache_dir: Path,
    device,
    allow_legacy_cache: bool,
    cache_stats: dict,
):
    key_v2 = build_cache_key(split_name, q_type, qa_id, video_id)
    path_v2 = cache_dir / f"{key_v2}.pt"
    out, status = _load_cache_tensor(path_v2, device, video_id)
    if out is not None:
        cache_stats["hit_v2"] += 1
        return out, path_v2

    if status == "video_mismatch":
        cache_stats["v2_mismatch"] += 1
    elif status == "invalid":
        cache_stats["v2_invalid"] += 1

    if allow_legacy_cache:
        legacy_path = cache_dir / f"{qa_id}.pt"
        legacy_out, legacy_status = _load_cache_tensor(legacy_path, device, video_id)
        if legacy_out is not None:
            cache_stats["hit_legacy"] += 1
            return legacy_out, path_v2
        if legacy_status == "video_mismatch":
            cache_stats["legacy_mismatch"] += 1
        elif legacy_status == "invalid":
            cache_stats["legacy_invalid"] += 1

    cache_stats["miss"] += 1
    return None, path_v2


def compute_saliency_and_cache(
    video_path: str,
    video_id: str,
    cache_save_path: Path,
    patch_processor: PatchProcessor,
    motion_sal_extractor: MotionSaliencyExtractor,
    visual_sal_extractor: DINOv3SaliencyExtractor,
    frame_cache: LRUCache,
    meta_cache: LRUCache,
    device,
    cache_stats: dict,
):
    frames = frame_cache.get(video_path)
    if frames is None:
        frames = read_video(video_path)
        frame_cache.put(video_path, frames)

    metadata = meta_cache.get(video_path)
    if metadata is None:
        metadata = patch_processor.get_video_metadata(video_path)
        meta_cache.put(video_path, metadata)

    indices = patch_processor.get_sampling_indices(metadata["total_frames"], metadata["fps"])
    _, _, _, h_bar, w_bar = patch_processor.get_smart_resize_grid(
        len(indices), metadata["height"], metadata["width"]
    )

    with torch.inference_mode():
        motion_np = np.asarray(
            motion_sal_extractor.extract_motion_saliency(frames, indices=indices.tolist()),
            dtype=np.float32,
        )
        visual_np = np.asarray(
            visual_sal_extractor.extract_dino_video_pixel_last(frames, indices=indices.tolist()),
            dtype=np.float32,
        )
        motion_sal = torch.tensor(motion_np, dtype=torch.float32, device=device)
        visual_sal = torch.tensor(visual_np, dtype=torch.float32, device=device)
        motion_input = motion_sal.unsqueeze(0).unsqueeze(0)
        visual_input = visual_sal.unsqueeze(0).unsqueeze(0)
        w_m = transform_pixel_to_patch(indices, h_bar, w_bar, motion_input, patch_processor)
        w_v = transform_pixel_to_patch(indices, h_bar, w_bar, visual_input, patch_processor)

    ensure_dir(cache_save_path.parent)
    torch.save(
        {
            "w_m": w_m.detach().cpu().half(),
            "w_v": w_v.detach().cpu().half(),
            "video_id": video_id,
            "cache_version": 2,
        },
        cache_save_path,
    )
    cache_stats["saved_v2"] += 1
    return {"w_m": w_m.to(device=device, dtype=torch.float32), "w_v": w_v.to(device=device, dtype=torch.float32)}


def build_tools():
    return {
        "ReverseVideo": ReverseVideo(),
        "SampleVideo": SampleVideo(),
        "ShuffleVideo": ShuffleVideo(),
        "BlurVideo": BlurVideo(),
        "NoiseVideo": NoiseVideo(),
        "HorizontalMirrorVideo": HorizontalMirrorVideo(),
        "VerticalMirrorVideo": VerticalMirrorVideo(),
        "GrayscaleVideo": GrayscaleVideo(),
    }


def build_candidate_token_ids(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return {}
    variants = {
        "a": ["A", "a", " A", " a"],
        "b": ["B", "b", " B", " b"],
        "c": ["C", "c", " C", " c"],
        "yes": ["yes", "Yes", " YES", " yes", " Yes"],
        "no": ["no", "No", " NO", " no", " No"],
    }
    out = {}
    for key, vals in variants.items():
        ids = set()
        for text in vals:
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        out[key] = sorted(ids)
    return out


def constrained_pred_token(
    raw_pred: str,
    generated_ids,
    options,
    single_token_mode: bool,
    candidate_token_ids: Dict[str, List[int]],
):
    token = normalize_answer(raw_pred)
    if not single_token_mode:
        return token, False

    allowed = ["a", "b", "c"] if options is not None else ["yes", "no"]
    if token in allowed:
        return token, False

    if generated_ids is not None and getattr(generated_ids, "scores", None):
        first_scores = generated_ids.scores[0]
        if first_scores.dim() == 2:
            first_scores = first_scores[0]
        best_token = None
        best_score = -1e30
        for cand in allowed:
            for tid in candidate_token_ids.get(cand, []):
                if tid < first_scores.shape[-1]:
                    score = float(first_scores[tid].item())
                    if score > best_score:
                        best_score = score
                        best_token = cand
        if best_token is not None:
            return best_token, True

    text = str(raw_pred).lower()
    for cand in allowed:
        if cand in text:
            return cand, True
    return allowed[0], True


def prepare_tool_matrix(tools_embeddings: Dict[str, torch.Tensor], device):
    tool_names = list(tools_embeddings.keys())
    vectors = []
    for name in tool_names:
        emb = tools_embeddings[name].to(device=device, dtype=torch.float32)
        vec = emb if emb.dim() == 1 else emb.mean(dim=0)
        vec = F.normalize(vec, p=2, dim=0)
        vectors.append(vec)
    tool_matrix = torch.stack(vectors, dim=0)
    return tool_names, tool_matrix


def _finalize_selected_indices(
    sorted_indices: torch.Tensor,
    selected_pos: torch.Tensor,
    probs: torch.Tensor,
    threshold: float,
    min_selected_tools: int,
    max_selected_tools: int,
):
    if selected_pos.numel() == 0:
        selected_pos = torch.tensor([0], device=sorted_indices.device, dtype=torch.long)
    else:
        last = int(selected_pos[-1].item())
        if last + 1 < sorted_indices.numel():
            # "超过阈值"语义：若当前还没超过阈值，则补下一个。
            if float(torch.cumsum(probs[sorted_indices], dim=0)[last].item()) < float(threshold):
                selected_pos = torch.cat(
                    [selected_pos, torch.tensor([last + 1], device=selected_pos.device, dtype=selected_pos.dtype)],
                    dim=0,
                )

    selected_indices = sorted_indices[selected_pos]

    if max_selected_tools > 0 and selected_indices.numel() > max_selected_tools:
        selected_indices = selected_indices[:max_selected_tools]

    if selected_indices.numel() < max(1, min_selected_tools):
        need = max(1, min_selected_tools) - int(selected_indices.numel())
        existing = set(selected_indices.tolist())
        extras = []
        for idx in sorted_indices.tolist():
            if idx not in existing:
                extras.append(idx)
                if len(extras) >= need:
                    break
        if extras:
            selected_indices = torch.cat(
                [selected_indices, torch.tensor(extras, device=selected_indices.device, dtype=selected_indices.dtype)],
                dim=0,
            )

    return selected_indices


def cumulative_threshold_hard_mask(
    probs: torch.Tensor,
    threshold: float,
    min_selected_tools: int = 1,
    max_selected_tools: int = -1,
):
    probs = probs.reshape(-1)
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_probs = probs[sorted_indices]
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    selected_pos = torch.nonzero(cumsum_probs <= float(threshold), as_tuple=True)[0]
    selected_indices = _finalize_selected_indices(
        sorted_indices=sorted_indices,
        selected_pos=selected_pos,
        probs=probs,
        threshold=threshold,
        min_selected_tools=min_selected_tools,
        max_selected_tools=max_selected_tools,
    )
    hard_mask = torch.zeros_like(probs)
    hard_mask[selected_indices] = 1.0
    return hard_mask, selected_indices, sorted_indices


def ste_hard_mask(probs: torch.Tensor, hard_mask: torch.Tensor):
    return hard_mask + probs - probs.detach()


def cosine_teacher_mask(
    state_embeddings: torch.Tensor,
    tool_matrix: torch.Tensor,
    threshold: float,
    min_selected_tools: int = 1,
    max_selected_tools: int = -1,
):
    if state_embeddings.dim() == 1:
        query_vec = state_embeddings
    else:
        query_vec = state_embeddings.mean(dim=0)
    query_vec = query_vec.to(device=tool_matrix.device, dtype=torch.float32)
    query_vec = F.normalize(query_vec, p=2, dim=0)
    cosine_scores = torch.matmul(tool_matrix, query_vec)
    probs = (cosine_scores + 1.0) / 2.0
    probs = torch.clamp(probs, 0.0, 1.0)
    hard_mask, selected_indices, _ = cumulative_threshold_hard_mask(
        probs=probs,
        threshold=threshold,
        min_selected_tools=min_selected_tools,
        max_selected_tools=max_selected_tools,
    )
    return hard_mask, probs, selected_indices


def tool_names_from_mask(tool_names: List[str], hard_mask: torch.Tensor):
    selected = torch.nonzero(hard_mask > 0.5, as_tuple=True)[0].tolist()
    return [tool_names[i] for i in selected]


def first_step_target_confidence(generated_ids, target_token: str, candidate_token_ids: Dict[str, List[int]]):
    if generated_ids is None or getattr(generated_ids, "scores", None) is None or len(generated_ids.scores) == 0:
        return 0.5
    scores = generated_ids.scores[0]
    if scores.dim() == 2:
        scores = scores[0]
    probs = torch.softmax(scores.float(), dim=-1)
    token_ids = candidate_token_ids.get(str(target_token).lower(), [])
    if not token_ids:
        return 0.5
    valid_ids = [tid for tid in token_ids if tid < probs.shape[-1]]
    if not valid_ids:
        return 0.5
    conf = torch.max(probs[torch.tensor(valid_ids, device=probs.device)]).item()
    return float(conf)


def confidence_to_beta_target(conf_motion: float, conf_visual: float):
    diff = float(conf_motion) - float(conf_visual)
    val = (diff + 1.0) / 2.0
    return float(max(0.0, min(1.0, val)))


def summarize_counts(correct_dict: Dict[str, int], total_dict: Dict[str, int]):
    summary = {}
    macro_acc_sum = 0.0
    valid_types = 0
    micro_correct = 0
    micro_total = 0
    for q_type in QUESTION_TYPES:
        c = int(correct_dict.get(q_type, 0))
        t = int(total_dict.get(q_type, 0))
        micro_correct += c
        micro_total += t
        if t > 0:
            acc = c / t
            summary[q_type] = acc
            macro_acc_sum += acc
            valid_types += 1
        else:
            summary[q_type] = 0.0
    summary["macro_average_acc"] = macro_acc_sum / valid_types if valid_types > 0 else 0.0
    summary["micro_overall_acc"] = (micro_correct / micro_total) if micro_total > 0 else 0.0
    summary["counts"] = {
        q: {"correct": int(correct_dict.get(q, 0)), "total": int(total_dict.get(q, 0))}
        for q in QUESTION_TYPES
    }
    summary["micro_counts"] = {"correct": int(micro_correct), "total": int(micro_total)}
    return summary


def collect_summary_from_logs(logs: List[dict]):
    correct = Counter()
    total = Counter()
    fallback = 0
    errors = 0
    timeouts = 0
    skipped = 0
    timing_sums = {k: 0.0 for k in TIMING_KEYS}
    timing_count = 0

    for row in logs:
        status = row.get("status", "ok")
        if status != "ok":
            errors += 1
            if row.get("error") == "timeout":
                timeouts += 1
            else:
                skipped += 1
            continue

        q_type = row.get("type")
        if q_type in QUESTION_TYPES:
            total[q_type] += 1
            if bool(row.get("is_correct", False)):
                correct[q_type] += 1

        fallback += int(row.get("single_token_fallback", False))
        timing = row.get("timing_sec") or {}
        timing_count += 1
        for key in TIMING_KEYS:
            timing_sums[key] += float(timing.get(key, 0.0))

    summary = summarize_counts(correct, total)
    summary["single_token_fallback_count"] = int(fallback)
    summary["num_errors"] = int(errors)
    summary["num_timeouts"] = int(timeouts)
    summary["num_skipped"] = int(skipped)
    summary["timing"] = {
        "count": int(timing_count),
        "sum_sec": timing_sums,
        "mean_sec_per_query": {
            k: (timing_sums[k] / timing_count if timing_count > 0 else 0.0) for k in TIMING_KEYS
        },
    }
    return summary


def _sample_timeout_handler(signum, frame):
    raise TimeoutError("sample watchdog timeout")


@contextmanager
def sample_watchdog(timeout_sec: float):
    if timeout_sec is None or float(timeout_sec) <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _sample_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


__all__ = [
    "CACHE_STATS_KEYS",
    "QUESTION_TYPES",
    "TIMING_KEYS",
    "LRUCache",
    "PatchProcessor",
    "DINOv3SaliencyExtractor",
    "MotionSaliencyExtractor",
    "apply_limit",
    "build_candidate_token_ids",
    "build_tools",
    "build_video_index",
    "collect_summary_from_logs",
    "compute_saliency_and_cache",
    "confidence_to_beta_target",
    "constrained_pred_token",
    "cosine_teacher_mask",
    "cumulative_threshold_hard_mask",
    "ensure_dir",
    "first_step_target_confidence",
    "get_cached_saliency",
    "load_embeddings",
    "load_qa_data",
    "load_train_subset_manifest",
    "normalize_answer",
    "parse_qa_item",
    "prepare_tool_matrix",
    "qa_uid",
    "resolve_video_path",
    "sample_watchdog",
    "save_json",
    "seed_everything",
    "ste_hard_mask",
    "str2bool",
    "subset_by_type",
    "tool_names_from_mask",
]

