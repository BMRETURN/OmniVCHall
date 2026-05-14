import argparse
import io
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vcd_new.models.gate import QueryVisualFusionGater
from vcd_new.models.selector import QFormerToolRouter
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
from vcd_new.train.optimizer import VCDPolicy, VCDTrainer
from vcd_new.utils_internvl import (
    PatchProcessor,
    answer_question_negative,
    answer_question_original,
    answer_question_positive,
    load_embeddings,
    load_qa_data,
    read_video,
    transform_pixel_to_patch,
)


VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
QUESTION_TYPES = ("s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa")
TIMING_KEYS = ("orig_gen", "policy", "neg_branch", "saliency", "pos_gen", "total")
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

INTERNVL_RUNTIME_CFG = {
    "num_segments": 8,
    "max_num_tiles": 1,
    "image_size": 448,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def configure_internvl_runtime(args):
    INTERNVL_RUNTIME_CFG["num_segments"] = max(1, int(args.internvl_num_segments))
    INTERNVL_RUNTIME_CFG["max_num_tiles"] = max(1, int(args.internvl_max_num_tiles))
    INTERNVL_RUNTIME_CFG["image_size"] = max(64, int(args.internvl_image_size))


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


def unwrap_policy(policy):
    return policy.module if hasattr(policy, "module") else policy


def log(msg: str, rank: int = 0):
    if rank == 0:
        print(msg, flush=True)


def maybe_quiet_context(rank: int):
    if rank == 0:
        return nullcontext()
    return redirect_stdout(io.StringIO())


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


class TimingAccumulator:
    def __init__(self):
        self.count = 0
        self.sums = {k: 0.0 for k in TIMING_KEYS}

    def add(self, tdict: dict):
        self.count += 1
        for k in TIMING_KEYS:
            self.sums[k] += float(tdict.get(k, 0.0))

    def to_dict(self):
        means = {
            k: (self.sums[k] / self.count if self.count > 0 else 0.0)
            for k in TIMING_KEYS
        }
        return {"count": self.count, "sum_sec": self.sums, "mean_sec_per_query": means}


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


def normalize_answer(text):
    m = re.search(r"\b(a|b|c|d|yes|no)\b", str(text).lower())
    return m.group(1) if m else str(text).lower().strip()


def apply_limit(data, limit):
    if limit is None or int(limit) < 0:
        return data
    return data[: int(limit)]


def build_video_index(video_dir: Path):
    index = {}
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            index[p.stem] = str(p)
    return index


def resolve_video_path(video_id, video_dir: Path, video_index, use_video_index: bool):
    if use_video_index and video_index is not None:
        return video_index.get(video_id)
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    return None


def probe_video_metadata(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if total_frames <= 0 or fps <= 0:
        return None
    duration_sec = float(total_frames) / float(max(fps, 1e-6))
    return {
        "total_frames": int(total_frames),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "duration_sec": float(duration_sec),
    }


def run_subset_video_health_check(
    train_data: List[dict],
    video_dir: Path,
    video_index,
    use_video_index: bool,
    probe_frames: int,
    max_probe_sec: float,
    max_total_frames: int,
    max_duration_sec: float,
    drop_slow: bool,
):
    unique_video_ids = sorted({str(x["video_id"]) for x in train_data})
    valid_video_ids = set()
    video_meta_map = {}
    bad_videos = []
    slow_videos = []
    missing_videos = []

    iterator = tqdm(unique_video_ids, desc="HealthCheck")
    for video_id in iterator:
        video_path = resolve_video_path(video_id, video_dir, video_index, use_video_index)
        if video_path is None:
            missing_videos.append(video_id)
            bad_videos.append({"video_id": video_id, "reason": "missing"})
            continue

        meta = probe_video_metadata(video_path)
        if meta is None:
            bad_videos.append({"video_id": video_id, "reason": "invalid_metadata"})
            continue

        is_slow = False
        slow_reason = None
        probe_decode_sec = 0.0
        if max_total_frames > 0 and meta["total_frames"] > max_total_frames:
            is_slow = True
            slow_reason = f"too_many_frames:{meta['total_frames']}"
        elif max_duration_sec > 0 and meta["duration_sec"] > max_duration_sec:
            is_slow = True
            slow_reason = f"too_long_duration:{meta['duration_sec']:.2f}"

        if not is_slow and probe_frames > 0:
            try:
                end_frame = min(meta["total_frames"] - 1, max(0, probe_frames - 1))
                t0 = time.perf_counter()
                _ = read_video(video_path, frame_range=(0, end_frame))
                probe_decode_sec = time.perf_counter() - t0
                if max_probe_sec > 0 and probe_decode_sec > max_probe_sec:
                    is_slow = True
                    slow_reason = f"probe_decode_sec:{probe_decode_sec:.3f}"
            except Exception:
                bad_videos.append({"video_id": video_id, "reason": "decode_probe_failed"})
                continue

        meta["probe_decode_sec"] = float(probe_decode_sec)
        if is_slow and drop_slow:
            slow_videos.append({"video_id": video_id, "reason": slow_reason, "meta": meta})
            continue

        valid_video_ids.add(video_id)
        video_meta_map[video_id] = meta

    report = {
        "total_unique_videos": len(unique_video_ids),
        "valid_videos": len(valid_video_ids),
        "removed_videos": len(unique_video_ids) - len(valid_video_ids),
        "missing_videos_count": len(missing_videos),
        "bad_videos_count": len(bad_videos),
        "slow_videos_count": len(slow_videos),
        "drop_slow": bool(drop_slow),
        "probe_frames": int(probe_frames),
        "max_probe_sec": float(max_probe_sec),
        "max_total_frames": int(max_total_frames),
        "max_duration_sec": float(max_duration_sec),
        "bad_videos": bad_videos,
        "slow_videos": slow_videos,
        "missing_videos": missing_videos,
    }
    return valid_video_ids, video_meta_map, report


def collect_video_metadata_for_dataset(
    dataset: List[dict],
    video_dir: Path,
    video_index,
    use_video_index: bool,
):
    video_ids = sorted({str(x["video_id"]) for x in dataset})
    out = {}
    for video_id in video_ids:
        video_path = resolve_video_path(video_id, video_dir, video_index, use_video_index)
        if video_path is None:
            continue
        meta = probe_video_metadata(video_path)
        if meta is None:
            continue
        out[video_id] = meta
    return out


def build_cache_key(split_name, q_type, qa_id, video_id):
    return f"{split_name}_{q_type}_{qa_id}_{video_id}"


def _load_cache_tensor(path: Path, device, expected_video_id):
    if not path.exists():
        return None, "missing"
    try:
        data = torch.load(path, map_location="cpu")
        cached_video_id = data.get("video_id")
        if expected_video_id is not None and cached_video_id != expected_video_id:
            return None, "video_mismatch"
        return {
            "w_m": data["w_m"].to(device, dtype=torch.float32),
            "w_v": data["w_v"].to(device, dtype=torch.float32),
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
    return {"w_m": w_m.to(device, dtype=torch.float32), "w_v": w_v.to(device, dtype=torch.float32)}


def build_tools():
    tools_dict = {
        "ReverseVideo": ReverseVideo(),
        "SampleVideo": SampleVideo(),
        "ShuffleVideo": ShuffleVideo(),
        "BlurVideo": BlurVideo(),
        "NoiseVideo": NoiseVideo(),
        "HorizontalMirrorVideo": HorizontalMirrorVideo(),
        "VerticalMirrorVideo": VerticalMirrorVideo(),
        "GrayscaleVideo": GrayscaleVideo(),
    }
    return tools_dict, list(tools_dict.keys())


def build_candidate_token_ids(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "encode"):
        tokenizer = processor
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
    for k, vals in variants.items():
        ids = set()
        for v in vals:
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        out[k] = sorted(ids)
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
            token_ids = candidate_token_ids.get(cand, [])
            for tid in token_ids:
                if tid < first_scores.shape[-1]:
                    score = float(first_scores[tid].item())
                    if score > best_score:
                        best_score = score
                        best_token = cand
        if best_token is not None:
            return best_token, True

    text = str(raw_pred).lower()
    if "yes" in text:
        return "yes", True
    if "no" in text:
        return "no", True
    if "a" in text and options is not None:
        return "a", True
    if "b" in text and options is not None:
        return "b", True
    if "c" in text and options is not None:
        return "c", True
    return allowed[0], True


def infer_single_item(
    item,
    split_name: str,
    model,
    processor,
    policy,
    tools_embeddings,
    tools_dict,
    patch_processor,
    motion_sal_extractor,
    visual_sal_extractor,
    video_dir: Path,
    video_index,
    use_video_index: bool,
    frame_cache: LRUCache,
    meta_cache: LRUCache,
    cache_dir: Path,
    allow_legacy_cache: bool,
    max_new_tokens: int,
    single_token_mode: bool,
    candidate_token_ids: Dict[str, List[int]],
    device,
    is_train: bool,
    train_std_dev: float,
    cache_stats: dict,
):
    parsed = parse_qa_item(item)
    if parsed is None:
        return None, "invalid_qa", None

    q_type, qa_id, question, gt_answer, options = parsed
    video_id = str(item["video_id"])
    video_path = resolve_video_path(video_id, video_dir, video_index, use_video_index)
    if video_path is None:
        return None, "missing_video", None

    stage_times = {}
    t_item_start = time.perf_counter()

    frames = frame_cache.get(video_path)
    if frames is None:
        frames = read_video(video_path)
        frame_cache.put(video_path, frames)

    t0 = time.perf_counter()
    orig_logits, last_hidden_states, _ = answer_question_original(
        model=model,
        processor=processor,
        video_path=video_path,
        question=question,
        options=options,
        max_new_tokens=max_new_tokens,
        video_frames=frames,
        num_segments=INTERNVL_RUNTIME_CFG["num_segments"],
        max_num_tiles=INTERNVL_RUNTIME_CFG["max_num_tiles"],
        image_size=INTERNVL_RUNTIME_CFG["image_size"],
    )
    stage_times["orig_gen"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = last_hidden_states.detach().to(device=device, dtype=torch.float32)
    policy_impl = unwrap_policy(policy)
    selected_tools, beta_tensor, log_p_router, log_p_gater = policy_impl.get_action_and_log_prob(
        state,
        tools_embeddings,
        std_dev=(train_std_dev if is_train else 0.0),
    )
    beta_value = float(beta_tensor.item())
    stage_times["policy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    neg_frames = frames
    for tool_name in selected_tools:
        neg_frames = tools_dict[tool_name].process(neg_frames)
    neg_logits = answer_question_negative(
        model=model,
        processor=processor,
        negative_video_path=video_path,
        question=question,
        options=options,
        max_new_tokens=max_new_tokens,
        video_frames=neg_frames,
        num_segments=INTERNVL_RUNTIME_CFG["num_segments"],
        max_num_tiles=INTERNVL_RUNTIME_CFG["max_num_tiles"],
        image_size=INTERNVL_RUNTIME_CFG["image_size"],
    )
    stage_times["neg_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache_data, save_path = get_cached_saliency(
        split_name=split_name,
        q_type=q_type,
        qa_id=qa_id,
        video_id=video_id,
        cache_dir=cache_dir,
        device=device,
        allow_legacy_cache=allow_legacy_cache,
        cache_stats=cache_stats,
    )
    if cache_data is None:
        cache_data = compute_saliency_and_cache(
            video_path=video_path,
            video_id=video_id,
            cache_save_path=save_path,
            patch_processor=patch_processor,
            motion_sal_extractor=motion_sal_extractor,
            visual_sal_extractor=visual_sal_extractor,
            frame_cache=frame_cache,
            meta_cache=meta_cache,
            device=device,
            cache_stats=cache_stats,
        )

    combined_raw = beta_value * cache_data["w_m"] + (1.0 - beta_value) * cache_data["w_v"]
    final_weights = torch.sigmoid(combined_raw)
    stage_times["saliency"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_answer, generated_ids = answer_question_positive(
        model=model,
        processor=processor,
        video_path=video_path,
        patch_weights=final_weights,
        question=question,
        original_logits=orig_logits,
        negative_logits=neg_logits,
        options=options,
        max_new_tokens=max_new_tokens,
        video_frames=frames,
        num_segments=INTERNVL_RUNTIME_CFG["num_segments"],
        max_num_tiles=INTERNVL_RUNTIME_CFG["max_num_tiles"],
        image_size=INTERNVL_RUNTIME_CFG["image_size"],
    )
    stage_times["pos_gen"] = time.perf_counter() - t0
    stage_times["total"] = time.perf_counter() - t_item_start

    pred_token, fallback_used = constrained_pred_token(
        raw_pred=pred_answer,
        generated_ids=generated_ids,
        options=options,
        single_token_mode=single_token_mode,
        candidate_token_ids=candidate_token_ids,
    )
    gt_token = normalize_answer(gt_answer)

    result = {
        "qa_id": int(qa_id),
        "video_id": video_id,
        "type": q_type,
        "gt": str(gt_answer),
        "pred": str(pred_token if single_token_mode else pred_answer),
        "gt_token": gt_token,
        "pred_token": pred_token,
        "is_correct": gt_token == pred_token,
        "single_token_fallback": bool(fallback_used),
        "beta": round(beta_value, 6),
        "tools": selected_tools,
        "timing_sec": {k: round(v, 6) for k, v in stage_times.items()},
    }
    return result, None, (log_p_router, log_p_gater)


def summarize_counts(correct_dict: Dict[str, int], total_dict: Dict[str, int]):
    summary = {}
    total_acc_sum = 0.0
    valid_types = 0
    for q_type in QUESTION_TYPES:
        c = int(correct_dict.get(q_type, 0))
        t = int(total_dict.get(q_type, 0))
        if t > 0:
            acc = c / t
            summary[q_type] = acc
            total_acc_sum += acc
            valid_types += 1
        else:
            summary[q_type] = 0.0
    summary["average_acc"] = total_acc_sum / valid_types if valid_types > 0 else 0.0
    summary["counts"] = {
        q: {"correct": int(correct_dict.get(q, 0)), "total": int(total_dict.get(q, 0))}
        for q in QUESTION_TYPES
    }
    return summary


def init_distributed(args):
    distributed = bool(args.distributed)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed and env_world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(seconds=int(args.dist_timeout_sec)),
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        distributed = False
        rank = 0
        world_size = 1
        local_rank = 0 if args.local_rank < 0 else int(args.local_rank)
    return distributed, rank, world_size, local_rank


def barrier(distributed):
    if distributed and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def broadcast_object(distributed, obj, src=0):
    if not distributed:
        return obj
    data = [obj]
    dist.broadcast_object_list(data, src=src)
    return data[0]


def all_reduce_tensor(distributed, tensor: torch.Tensor):
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_results(distributed, local_results: List[dict]):
    if not distributed:
        return local_results
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_results)
    merged = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def build_rank_indices(length: int, rank: int, world_size: int, shuffle: bool, seed: int, epoch: int):
    indices = list(range(length))
    if shuffle:
        rnd = random.Random(seed + epoch)
        rnd.shuffle(indices)
    return indices[rank::world_size]


def build_rank_indices_equal_steps(length: int, rank: int, world_size: int, seed: int, epoch: int):
    indices = list(range(length))
    rnd = random.Random(seed + epoch)
    rnd.shuffle(indices)
    steps = int(math.ceil(float(length) / float(world_size))) if length > 0 else 0
    out = []
    for step in range(steps):
        gid = step * world_size + rank
        out.append(indices[gid] if gid < length else None)
    return out


def build_dataset_item_costs(dataset: List[dict], video_meta_map: Dict[str, dict], default_cost: float = 1.0):
    costs = []
    for item in dataset:
        video_id = str(item["video_id"])
        meta = video_meta_map.get(video_id, {})
        c = float(meta.get("total_frames", default_cost))
        if c <= 0:
            c = float(default_cost)
        costs.append(c)
    return costs


def build_rank_indices_cost_balanced(
    item_costs: List[float],
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
    equal_steps: bool,
):
    n = len(item_costs)
    indices = list(range(n))
    rnd = random.Random(seed + epoch)
    rnd.shuffle(indices)
    indices.sort(key=lambda i: (float(item_costs[i]), i), reverse=True)

    bins = [[] for _ in range(world_size)]
    loads = [0.0 for _ in range(world_size)]
    for idx in indices:
        min_load = min(loads)
        candidate_ranks = [r for r in range(world_size) if loads[r] == min_load]
        chosen_rank = candidate_ranks[rnd.randrange(len(candidate_ranks))]
        bins[chosen_rank].append(idx)
        loads[chosen_rank] += float(item_costs[idx])

    if equal_steps:
        max_len = max((len(x) for x in bins), default=0)
        for r in range(world_size):
            while len(bins[r]) < max_len:
                bins[r].append(None)
    else:
        max_len = max((len(x) for x in bins), default=0)

    return bins[rank], {"rank_loads": loads, "max_steps": int(max_len)}


def save_training_state(
    path: Path,
    policy,
    optimizer,
    trainer: VCDTrainer,
    epoch: int,
    best_val_acc: float,
    best_epoch: int,
    no_improve: int,
    train_logs: List[dict],
):
    ensure_dir(path.parent)
    payload = {
        "epoch": int(epoch),
        "policy_state": unwrap_policy(policy).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "trainer_state": {
            "baseline": float(trainer.baseline),
            "alpha": float(trainer.alpha),
            "current_step": int(trainer.current_step),
            "accumulation_steps": int(trainer.accumulation_steps),
        },
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "no_improve": int(no_improve),
        "train_logs": train_logs,
    }
    torch.save(payload, path)


def load_training_state(path: Path, policy, optimizer, trainer: VCDTrainer, device):
    if not path.exists():
        return None
    state = torch.load(path, map_location=device)
    unwrap_policy(policy).load_state_dict(state["policy_state"])
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    trainer_state = state.get("trainer_state", {})
    trainer.baseline = float(trainer_state.get("baseline", trainer.baseline))
    trainer.alpha = float(trainer_state.get("alpha", trainer.alpha))
    trainer.current_step = int(trainer_state.get("current_step", trainer.current_step))
    trainer.accumulation_steps = int(trainer_state.get("accumulation_steps", trainer.accumulation_steps))

    return {
        "epoch": int(state.get("epoch", 0)),
        "best_val_acc": float(state.get("best_val_acc", -1.0)),
        "best_epoch": int(state.get("best_epoch", -1)),
        "no_improve": int(state.get("no_improve", 0)),
        "train_logs": state.get("train_logs", []),
    }


def trainer_noop_step(policy, trainer: VCDTrainer):
    base = unwrap_policy(policy)
    params = [p for p in base.parameters() if p.requires_grad]
    if not params:
        return
    dummy = torch.zeros([], device=params[0].device, dtype=params[0].dtype)
    for p in params:
        dummy = dummy + (p.sum() * 0.0)
    dummy = dummy / float(trainer.accumulation_steps)
    dummy.backward()
    trainer.current_step += 1
    if trainer.current_step % trainer.accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()


def trainer_flush(policy, trainer: VCDTrainer):
    if trainer.current_step % trainer.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()


def run_train_epoch_distributed(
    dataset,
    item_costs: Optional[List[float]],
    epoch: int,
    model,
    processor,
    policy,
    trainer,
    tools_embeddings,
    tools_dict,
    patch_processor,
    motion_sal_extractor,
    visual_sal_extractor,
    video_dir: Path,
    video_index,
    use_video_index: bool,
    frame_cache: LRUCache,
    meta_cache: LRUCache,
    cache_dir: Path,
    allow_legacy_cache: bool,
    max_new_tokens: int,
    single_token_mode: bool,
    candidate_token_ids: Dict[str, List[int]],
    profile_timing: bool,
    train_std_dev: float,
    device,
    rank: int,
    world_size: int,
    distributed: bool,
    seed: int,
    balance_by_cost: bool,
    sample_timeout_sec: float,
):
    policy.train()
    assignment_info = {}
    if balance_by_cost and item_costs is not None and len(item_costs) == len(dataset):
        local_indices, assignment_info = build_rank_indices_cost_balanced(
            item_costs=item_costs,
            rank=rank,
            world_size=world_size,
            seed=seed,
            epoch=epoch,
            equal_steps=True,
        )
    else:
        local_indices = build_rank_indices_equal_steps(len(dataset), rank, world_size, seed=seed, epoch=epoch)
        assignment_info = {"rank_loads": [], "max_steps": len(local_indices)}
    iterator = tqdm(local_indices, desc=f"Training-E{epoch}-R{rank}") if rank == 0 else local_indices

    rewards = []
    losses = []
    timing = TimingAccumulator()
    fallback_count = 0
    cache_stats = {k: 0 for k in CACHE_STATS_KEYS}
    skipped = 0
    timeouts = 0
    dummy_steps = 0

    for idx in iterator:
        if idx is None:
            trainer_noop_step(policy, trainer)
            dummy_steps += 1
            continue

        item = dataset[idx]
        try:
            with sample_watchdog(sample_timeout_sec):
                result, _, log_probs = infer_single_item(
                    item=item,
                    split_name="train",
                    model=model,
                    processor=processor,
                    policy=policy,
                    tools_embeddings=tools_embeddings,
                    tools_dict=tools_dict,
                    patch_processor=patch_processor,
                    motion_sal_extractor=motion_sal_extractor,
                    visual_sal_extractor=visual_sal_extractor,
                    video_dir=video_dir,
                    video_index=video_index,
                    use_video_index=use_video_index,
                    frame_cache=frame_cache,
                    meta_cache=meta_cache,
                    cache_dir=cache_dir,
                    allow_legacy_cache=allow_legacy_cache,
                    max_new_tokens=max_new_tokens,
                    single_token_mode=single_token_mode,
                    candidate_token_ids=candidate_token_ids,
                    device=device,
                    is_train=True,
                    train_std_dev=train_std_dev,
                    cache_stats=cache_stats,
                )
            if result is None or log_probs is None:
                skipped += 1
                trainer_noop_step(policy, trainer)
                continue

            reward = trainer.compute_reward(result["pred"], result["gt"])
            router_log_prob, gater_log_prob = log_probs
            loss = trainer.step(reward, router_log_prob, gater_log_prob)

            rewards.append(float(reward))
            if loss != 0:
                losses.append(float(loss))
            fallback_count += int(result.get("single_token_fallback", False))
            if profile_timing:
                timing.add(result["timing_sec"])
        except TimeoutError:
            skipped += 1
            timeouts += 1
            trainer_noop_step(policy, trainer)
        except Exception as e:
            skipped += 1
            trainer_noop_step(policy, trainer)
            if rank == 0:
                print(f"[Training] error: {e}")

    trainer_flush(policy, trainer)

    stats_tensor = torch.tensor(
        [
            float(sum(rewards)),
            float(len(rewards)),
            float(sum(losses)),
            float(len(losses)),
            float(skipped),
            float(dummy_steps),
            float(fallback_count),
            float(timeouts),
        ],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_tensor(distributed, stats_tensor)

    cache_tensor = torch.tensor(
        [float(cache_stats[k]) for k in CACHE_STATS_KEYS],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_tensor(distributed, cache_tensor)

    timing_tensor = torch.tensor(
        [float(timing.sums[k]) for k in TIMING_KEYS] + [float(timing.count)],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_tensor(distributed, timing_tensor)

    avg_reward = (stats_tensor[0] / max(stats_tensor[1], 1.0)).item()
    avg_loss = (stats_tensor[2] / max(stats_tensor[3], 1.0)).item()
    timing_count = float(timing_tensor[-1].item())
    timing_sum = {k: float(timing_tensor[i].item()) for i, k in enumerate(TIMING_KEYS)}
    timing_mean = {k: (timing_sum[k] / timing_count if timing_count > 0 else 0.0) for k in TIMING_KEYS}

    return {
        "epoch": epoch,
        "num_samples": int(stats_tensor[1].item()),
        "avg_reward": float(avg_reward),
        "avg_loss": float(avg_loss),
        "num_skipped": int(stats_tensor[4].item()),
        "num_dummy_steps": int(stats_tensor[5].item()),
        "single_token_fallback_count": int(stats_tensor[6].item()),
        "num_timeouts": int(stats_tensor[7].item()),
        "assignment_info": assignment_info,
        "cache_stats": {k: int(cache_tensor[i].item()) for i, k in enumerate(CACHE_STATS_KEYS)},
        "timing": {
            "count": int(timing_count),
            "sum_sec": timing_sum,
            "mean_sec_per_query": timing_mean,
        },
    }


def evaluate_dataset_distributed(
    dataset,
    item_costs: Optional[List[float]],
    split_name: str,
    stage_name: str,
    epoch: Optional[int],
    model,
    processor,
    policy,
    tools_embeddings,
    tools_dict,
    patch_processor,
    motion_sal_extractor,
    visual_sal_extractor,
    video_dir: Path,
    video_index,
    use_video_index: bool,
    frame_cache: LRUCache,
    meta_cache: LRUCache,
    cache_dir: Path,
    allow_legacy_cache: bool,
    max_new_tokens: int,
    single_token_mode: bool,
    candidate_token_ids: Dict[str, List[int]],
    profile_timing: bool,
    output_dir: Path,
    device,
    rank: int,
    world_size: int,
    distributed: bool,
    save_logs: bool,
    balance_by_cost: bool,
    sample_timeout_sec: float,
):
    policy.eval()
    if balance_by_cost and item_costs is not None and len(item_costs) == len(dataset):
        local_indices, _ = build_rank_indices_cost_balanced(
            item_costs=item_costs,
            rank=rank,
            world_size=world_size,
            seed=0,
            epoch=0,
            equal_steps=False,
        )
    else:
        local_indices = build_rank_indices(len(dataset), rank, world_size, shuffle=False, seed=0, epoch=0)
    iterator = tqdm(local_indices, desc=f"{stage_name}-R{rank}") if rank == 0 else local_indices

    local_results = []
    local_correct = Counter()
    local_total = Counter()
    local_skipped = 0
    local_timeouts = 0
    local_fallback = 0
    timing = TimingAccumulator()
    cache_stats = {k: 0 for k in CACHE_STATS_KEYS}

    for idx in iterator:
        item = dataset[idx]
        try:
            with sample_watchdog(sample_timeout_sec):
                result, _, _ = infer_single_item(
                    item=item,
                    split_name=split_name,
                    model=model,
                    processor=processor,
                    policy=policy,
                    tools_embeddings=tools_embeddings,
                    tools_dict=tools_dict,
                    patch_processor=patch_processor,
                    motion_sal_extractor=motion_sal_extractor,
                    visual_sal_extractor=visual_sal_extractor,
                    video_dir=video_dir,
                    video_index=video_index,
                    use_video_index=use_video_index,
                    frame_cache=frame_cache,
                    meta_cache=meta_cache,
                    cache_dir=cache_dir,
                    allow_legacy_cache=allow_legacy_cache,
                    max_new_tokens=max_new_tokens,
                    single_token_mode=single_token_mode,
                    candidate_token_ids=candidate_token_ids,
                    device=device,
                    is_train=False,
                    train_std_dev=0.0,
                    cache_stats=cache_stats,
                )
            if result is None:
                local_skipped += 1
                continue
            local_results.append(result)
            q_type = result["type"]
            local_total[q_type] += 1
            if result["is_correct"]:
                local_correct[q_type] += 1
            local_fallback += int(result.get("single_token_fallback", False))
            if profile_timing:
                timing.add(result["timing_sec"])
        except TimeoutError:
            local_skipped += 1
            local_timeouts += 1
        except Exception as e:
            local_skipped += 1
            if rank == 0:
                print(f"[{stage_name}] error: {e}")

    count_tensor = torch.tensor(
        [
            float(local_correct["s_ynqa"]),
            float(local_total["s_ynqa"]),
            float(local_correct["m_ynqa"]),
            float(local_total["m_ynqa"]),
            float(local_correct["s_mcqa"]),
            float(local_total["s_mcqa"]),
            float(local_correct["m_mcqa"]),
            float(local_total["m_mcqa"]),
            float(local_skipped),
            float(local_timeouts),
            float(local_fallback),
            float(len(local_results)),
        ],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_tensor(distributed, count_tensor)

    cache_tensor = torch.tensor(
        [float(cache_stats[k]) for k in CACHE_STATS_KEYS],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_tensor(distributed, cache_tensor)

    timing_tensor = torch.tensor(
        [float(timing.sums[k]) for k in TIMING_KEYS] + [float(timing.count)],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_tensor(distributed, timing_tensor)

    merged_results = gather_results(distributed, local_results) if save_logs else []

    if rank != 0:
        return None

    correct_dict = {
        "s_ynqa": int(count_tensor[0].item()),
        "m_ynqa": int(count_tensor[2].item()),
        "s_mcqa": int(count_tensor[4].item()),
        "m_mcqa": int(count_tensor[6].item()),
    }
    total_dict = {
        "s_ynqa": int(count_tensor[1].item()),
        "m_ynqa": int(count_tensor[3].item()),
        "s_mcqa": int(count_tensor[5].item()),
        "m_mcqa": int(count_tensor[7].item()),
    }
    summary = summarize_counts(correct_dict, total_dict)
    summary["stage_name"] = stage_name
    summary["epoch"] = epoch
    summary["num_samples"] = int(count_tensor[11].item())
    summary["num_skipped"] = int(count_tensor[8].item())
    summary["num_timeouts"] = int(count_tensor[9].item())
    summary["single_token_fallback_count"] = int(count_tensor[10].item())
    summary["cache_stats"] = {k: int(cache_tensor[i].item()) for i, k in enumerate(CACHE_STATS_KEYS)}

    timing_count = float(timing_tensor[-1].item())
    timing_sum = {k: float(timing_tensor[i].item()) for i, k in enumerate(TIMING_KEYS)}
    summary["timing"] = {
        "count": int(timing_count),
        "sum_sec": timing_sum,
        "mean_sec_per_query": {
            k: (timing_sum[k] / timing_count if timing_count > 0 else 0.0) for k in TIMING_KEYS
        },
    }

    prefix = stage_name.lower()
    if epoch is not None:
        prefix = f"{prefix}_epoch_{epoch}"
    save_json(summary, output_dir / f"{prefix}_metrics.json")
    if save_logs:
        save_json(merged_results, output_dir / f"{prefix}_logs.json")
    return summary


def paper_reference():
    return {
        "table5_qwen3_8b": {
            "s_ynqa": 0.79,
            "m_ynqa": 0.78,
            "s_mcqa": 0.66,
            "m_mcqa": 0.64,
            "average_acc": 0.73,
        },
        "table14_latency_sec_per_query": 9.12,
    }


def parse_q_type(item):
    parsed = parse_qa_item(item)
    if parsed is None:
        return None
    return parsed[0]


def fair_sample_train_data(train_data: List[dict], sample_size: int, seed: int):
    if sample_size <= 0 or sample_size >= len(train_data):
        return train_data

    rng = random.Random(seed)
    by_video = defaultdict(list)
    for item in train_data:
        by_video[str(item["video_id"])].append(item)

    videos = sorted(by_video.keys())
    selected = []
    if sample_size >= len(videos):
        for vid in videos:
            selected.append(rng.choice(by_video[vid]))
    else:
        chosen_videos = rng.sample(videos, sample_size)
        for vid in chosen_videos:
            selected.append(rng.choice(by_video[vid]))
        return selected

    def item_uid(x):
        parsed = parse_qa_item(x)
        if parsed is None:
            return ("unknown", str(x.get("video_id", "")), str(id(x)))
        q_type, qa_id, _, _, _ = parsed
        return (q_type, str(x["video_id"]), int(qa_id))

    selected_uid = {item_uid(x) for x in selected}
    remaining = [x for x in train_data if item_uid(x) not in selected_uid]

    total = len(train_data)
    type_counts = Counter(parse_q_type(x) for x in train_data if parse_q_type(x) is not None)
    target = {}
    floor_sum = 0
    for q in QUESTION_TYPES:
        raw = type_counts[q] * float(sample_size) / float(total)
        target[q] = int(math.floor(raw))
        floor_sum += target[q]
    remain_slots = sample_size - floor_sum
    if remain_slots > 0:
        order = sorted(
            QUESTION_TYPES,
            key=lambda q: ((type_counts[q] * float(sample_size) / float(total)) - target[q], type_counts[q]),
            reverse=True,
        )
        for i in range(remain_slots):
            target[order[i % len(order)]] += 1

    now = Counter(parse_q_type(x) for x in selected if parse_q_type(x) is not None)
    quota = {q: max(0, target[q] - now[q]) for q in QUESTION_TYPES}

    pool_by_type = defaultdict(list)
    for x in remaining:
        q = parse_q_type(x)
        if q is not None:
            pool_by_type[q].append(x)

    for q in QUESTION_TYPES:
        need = min(quota[q], len(pool_by_type[q]))
        if need > 0:
            chosen = rng.sample(pool_by_type[q], need)
            selected.extend(chosen)
            chosen_uid = {item_uid(x) for x in chosen}
            pool_by_type[q] = [x for x in pool_by_type[q] if item_uid(x) not in chosen_uid]

    if len(selected) < sample_size:
        selected_uid = {item_uid(x) for x in selected}
        residual = [x for x in train_data if item_uid(x) not in selected_uid]
        fill = sample_size - len(selected)
        selected.extend(rng.sample(residual, fill))
    return selected


def load_train_subset_manifest(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid manifest format: {path}")
    subset = []
    for x in items:
        if isinstance(x, dict) and "item" in x:
            subset.append(x["item"])
        else:
            subset.append(x)
    return subset, manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Fast distributed training entry (5-GPU) for vcd_new + InternVL")
    parser.add_argument("--distributed", type=str2bool, default=True)
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_timeout_sec", type=int, default=7200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "InternVL3_5-8B"),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "MyBench"),
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "MyBench" / "all_video"),
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "saliency_cache"),
    )
    parser.add_argument(
        "--tools_embeddings",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "tools" / "tools_embeddings_internvl.pkl"),
    )
    parser.add_argument(
        "--dino_dir",
        type=str,
        default=str(
            PROJECT_ROOT / "checkpoints" / "DINOv3" / "dinov3-vitl16-pretrain-lvd1689m"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")

    parser.add_argument("--epochs_max", type=int, default=5)
    parser.add_argument("--epochs_min", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--val_limit", type=int, default=256)
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--train_subset_manifest", type=str, default="")
    parser.add_argument("--train_sample_size", type=int, default=1800)
    parser.add_argument("--fair_sample_mode", type=str, default="video_cover_stratified")

    parser.add_argument("--single_token_mode", type=str2bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1)

    parser.add_argument("--resume_policy_path", type=str, default="")
    parser.add_argument("--allow_legacy_cache", type=str2bool, default=True)
    parser.add_argument("--profile_timing", type=str2bool, default=True)
    parser.add_argument("--use_video_index", type=str2bool, default=True)
    parser.add_argument("--lru_cache_size", type=int, default=16)
    parser.add_argument("--train_std_dev", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--router_threshold", type=float, default=0.4)
    parser.add_argument("--embed_dim", type=int, default=4096)
    parser.add_argument("--final_full_eval", type=str2bool, default=True)
    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--sample_timeout_sec", type=float, default=180.0)
    parser.add_argument("--balance_by_cost", type=str2bool, default=True)
    parser.add_argument("--health_check_enable", type=str2bool, default=True)
    parser.add_argument("--health_check_probe_frames", type=int, default=12)
    parser.add_argument("--health_check_max_probe_sec", type=float, default=15.0)
    parser.add_argument("--health_check_max_total_frames", type=int, default=4000)
    parser.add_argument("--health_check_max_duration_sec", type=float, default=300.0)
    parser.add_argument("--health_check_drop_slow", type=str2bool, default=True)
    parser.add_argument("--internvl_num_segments", type=int, default=8)
    parser.add_argument("--internvl_max_num_tiles", type=int, default=1)
    parser.add_argument("--internvl_image_size", type=int, default=448)
    return parser


def main():
    args = build_parser().parse_args()
    configure_internvl_runtime(args)
    distributed, rank, world_size, local_rank = init_distributed(args)

    if torch.cuda.is_available():
        if args.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = args.device
    else:
        device = "cpu"

    if args.single_token_mode and args.max_new_tokens != 1 and rank == 0:
        print("[Info] single_token_mode enabled, forcing max_new_tokens=1")
        args.max_new_tokens = 1

    seed_everything(args.seed + rank)

    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S_fast5")
    run_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "vcd_new" / "runs" / run_id)
    if rank == 0:
        ensure_dir(run_dir)
        ensure_dir(run_dir / "checkpoints")
        ensure_dir(run_dir / "val_output")
        ensure_dir(run_dir / "test_output")
        save_json(vars(args), run_dir / "run_config.json")
    barrier(distributed)

    log("Loading dataset...", rank)
    dataset_root = Path(args.dataset_root)
    with maybe_quiet_context(rank):
        val_data_full = load_qa_data(str(dataset_root / "val"), shuffle=False)
        test_data_full = load_qa_data(str(dataset_root / "test"), shuffle=False)

    train_data_full = []
    manifest_meta = {}
    if args.train_subset_manifest:
        subset_path = Path(args.train_subset_manifest)
        train_data, manifest_meta = load_train_subset_manifest(subset_path)
        train_full_size = int(
            manifest_meta.get("full_train_stats", {}).get("total_qas", len(train_data))
        )
        log(f"Loaded train subset manifest: {subset_path} ({len(train_data)} samples)", rank)
    else:
        with maybe_quiet_context(rank):
            train_data_full = load_qa_data(str(dataset_root / "train"), shuffle=True)
        train_data = train_data_full
        train_full_size = len(train_data_full)
        if args.train_sample_size > 0 and args.train_sample_size < len(train_data):
            if args.fair_sample_mode != "video_cover_stratified":
                raise ValueError(f"Unsupported fair_sample_mode: {args.fair_sample_mode}")
            train_data = fair_sample_train_data(train_data, args.train_sample_size, args.seed)
            log(f"On-the-fly fair sampled train subset: {len(train_data)}", rank)

    train_data = apply_limit(train_data, args.train_limit)
    val_data_small = apply_limit(val_data_full, args.val_limit)
    test_data_limited = apply_limit(test_data_full, args.test_limit)

    video_dir = Path(args.video_dir)
    video_index = build_video_index(video_dir) if args.use_video_index else None
    if rank == 0 and args.use_video_index:
        log(f"Video index size: {len(video_index)}", rank)

    health_report = {}
    video_meta_map = {}
    if args.health_check_enable:
        sync_payload = None
        if rank == 0:
            before_items = len(train_data)
            valid_video_ids, video_meta_map, health_report = run_subset_video_health_check(
                train_data=train_data,
                video_dir=video_dir,
                video_index=video_index,
                use_video_index=args.use_video_index,
                probe_frames=args.health_check_probe_frames,
                max_probe_sec=args.health_check_max_probe_sec,
                max_total_frames=args.health_check_max_total_frames,
                max_duration_sec=args.health_check_max_duration_sec,
                drop_slow=args.health_check_drop_slow,
            )
            train_data = [x for x in train_data if str(x["video_id"]) in valid_video_ids]
            health_report["train_items_before"] = before_items
            health_report["train_items_after"] = len(train_data)
            health_report["removed_items"] = before_items - len(train_data)
            save_json(health_report, run_dir / "health_check_report.json")
            log(
                f"Health check done: valid_videos={len(valid_video_ids)}, "
                f"train_items {before_items}->{len(train_data)}",
                rank,
            )
            sync_payload = {
                "valid_video_ids": sorted(valid_video_ids),
                "video_meta_map": video_meta_map,
            }
        sync_payload = broadcast_object(distributed, sync_payload, src=0)
        if rank != 0:
            valid_video_ids = set(sync_payload["valid_video_ids"])
            video_meta_map = sync_payload["video_meta_map"]
            train_data = [x for x in train_data if str(x["video_id"]) in valid_video_ids]
    elif args.balance_by_cost:
        sync_payload = None
        if rank == 0:
            video_meta_map = collect_video_metadata_for_dataset(
                dataset=train_data,
                video_dir=video_dir,
                video_index=video_index,
                use_video_index=args.use_video_index,
            )
            sync_payload = {"video_meta_map": video_meta_map}
        sync_payload = broadcast_object(distributed, sync_payload, src=0)
        if rank != 0:
            video_meta_map = sync_payload["video_meta_map"]

    train_item_costs = (
        build_dataset_item_costs(train_data, video_meta_map, default_cost=1.0)
        if args.balance_by_cost
        else None
    )
    val_item_costs_small = (
        build_dataset_item_costs(val_data_small, video_meta_map, default_cost=1.0)
        if args.balance_by_cost
        else None
    )
    val_item_costs_full = (
        build_dataset_item_costs(val_data_full, video_meta_map, default_cost=1.0)
        if args.balance_by_cost
        else None
    )
    test_item_costs_limited = (
        build_dataset_item_costs(test_data_limited, video_meta_map, default_cost=1.0)
        if args.balance_by_cost
        else None
    )
    test_item_costs_full = (
        build_dataset_item_costs(test_data_full, video_meta_map, default_cost=1.0)
        if args.balance_by_cost
        else None
    )

    if rank == 0:
        manifest_meta_compact = dict(manifest_meta) if isinstance(manifest_meta, dict) else {}
        if "items" in manifest_meta_compact:
            manifest_meta_compact["items_count"] = len(manifest_meta_compact.get("items", []))
            manifest_meta_compact.pop("items", None)
        sampling_report = {
            "train_full_size": int(train_full_size),
            "train_used_size": len(train_data),
            "val_full_size": len(val_data_full),
            "val_small_size": len(val_data_small),
            "test_full_size": len(test_data_full),
            "test_limited_size": len(test_data_limited),
            "manifest_meta": manifest_meta_compact,
            "health_check_enable": bool(args.health_check_enable),
            "balance_by_cost": bool(args.balance_by_cost),
            "sample_timeout_sec": float(args.sample_timeout_sec),
        }
        save_json(sampling_report, run_dir / "sampling_report.json")
        log(
            f"Data loaded: train={len(train_data)}, val_small={len(val_data_small)}, "
            f"val_full={len(val_data_full)}, test_full={len(test_data_full)}",
            rank,
        )

    log(f"Loading InternVL model and tokenizer on rank={rank}, device={device} ...", rank)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        use_flash_attn=False,
    )
    if str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    candidate_token_ids = build_candidate_token_ids(processor)

    log("Loading tools and policy...", rank)
    tools_embeddings = load_embeddings(args.tools_embeddings, device)
    tools_dict, tool_names = build_tools()
    selector = QFormerToolRouter(
        num_tools=len(tool_names),
        d_in=args.embed_dim,
        d_model=1024,
        threshold=args.router_threshold,
        device=device,
    )
    gater = QueryVisualFusionGater(embed_dim=args.embed_dim).to(device)
    policy = VCDPolicy(selector, gater, tool_names).to(device)

    if args.resume_policy_path:
        state = torch.load(args.resume_policy_path, map_location=device)
        policy.load_state_dict(state)
        log(f"Loaded policy checkpoint: {args.resume_policy_path}", rank)

    if distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = VCDTrainer(policy, optimizer, accumulation_steps=args.batch_size)

    log("Initializing saliency extractors...", rank)
    motion_sal_extractor = MotionSaliencyExtractor()
    visual_sal_extractor = DINOv3SaliencyExtractor(
        checkpoints=args.dino_dir,
        device=device,
    )
    patch_processor = PatchProcessor()

    frame_cache = LRUCache(args.lru_cache_size)
    meta_cache = LRUCache(args.lru_cache_size)

    cache_root = Path(args.cache_root)
    train_cache = cache_root / "train"
    val_cache = cache_root / "val"
    test_cache = cache_root / "test"
    if rank == 0:
        ensure_dir(train_cache)
        ensure_dir(val_cache)
        ensure_dir(test_cache)
    barrier(distributed)

    train_logs = []
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    start_epoch = 1

    state_ckpt_path = run_dir / "checkpoints" / "training_state_latest.pt"
    if args.auto_resume and state_ckpt_path.exists():
        resumed = load_training_state(
            path=state_ckpt_path,
            policy=policy,
            optimizer=optimizer,
            trainer=trainer,
            device=device,
        )
        if resumed is not None:
            train_logs = resumed.get("train_logs", [])
            best_val_acc = float(resumed.get("best_val_acc", -1.0))
            best_epoch = int(resumed.get("best_epoch", -1))
            no_improve = int(resumed.get("no_improve", 0))
            start_epoch = int(resumed.get("epoch", 0)) + 1
            if rank == 0:
                log(
                    f"Auto-resume from epoch={start_epoch - 1}, "
                    f"best_epoch={best_epoch}, best_val={best_val_acc:.4f}",
                    rank,
                )
    barrier(distributed)

    log("Start training...", rank)
    for epoch in range(start_epoch, args.epochs_max + 1):
        epoch_train = run_train_epoch_distributed(
            dataset=train_data,
            item_costs=train_item_costs,
            epoch=epoch,
            model=model,
            processor=processor,
            policy=policy,
            trainer=trainer,
            tools_embeddings=tools_embeddings,
            tools_dict=tools_dict,
            patch_processor=patch_processor,
            motion_sal_extractor=motion_sal_extractor,
            visual_sal_extractor=visual_sal_extractor,
            video_dir=video_dir,
            video_index=video_index,
            use_video_index=args.use_video_index,
            frame_cache=frame_cache,
            meta_cache=meta_cache,
            cache_dir=train_cache,
            allow_legacy_cache=args.allow_legacy_cache,
            max_new_tokens=args.max_new_tokens,
            single_token_mode=args.single_token_mode,
            candidate_token_ids=candidate_token_ids,
            profile_timing=args.profile_timing,
            train_std_dev=args.train_std_dev,
            device=device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            seed=args.seed,
            balance_by_cost=args.balance_by_cost,
            sample_timeout_sec=args.sample_timeout_sec,
        )
        if rank == 0:
            train_logs.append(epoch_train)

        val_summary = evaluate_dataset_distributed(
            dataset=val_data_small,
            item_costs=val_item_costs_small,
            split_name="val",
            stage_name="validation",
            epoch=epoch,
            model=model,
            processor=processor,
            policy=policy,
            tools_embeddings=tools_embeddings,
            tools_dict=tools_dict,
            patch_processor=patch_processor,
            motion_sal_extractor=motion_sal_extractor,
            visual_sal_extractor=visual_sal_extractor,
            video_dir=video_dir,
            video_index=video_index,
            use_video_index=args.use_video_index,
            frame_cache=frame_cache,
            meta_cache=meta_cache,
            cache_dir=val_cache,
            allow_legacy_cache=args.allow_legacy_cache,
            max_new_tokens=args.max_new_tokens,
            single_token_mode=args.single_token_mode,
            candidate_token_ids=candidate_token_ids,
            profile_timing=args.profile_timing,
            output_dir=run_dir / "val_output",
            device=device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            save_logs=False,
            balance_by_cost=args.balance_by_cost,
            sample_timeout_sec=args.sample_timeout_sec,
        )

        state = {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "stop": False,
        }
        if rank == 0:
            val_acc = float(val_summary["average_acc"])
            log(f"[Epoch {epoch}] val_small average_acc={val_acc:.4f}", rank)
            epoch_ckpt = run_dir / "checkpoints" / f"vcd_policy_epoch_{epoch}.pth"
            torch.save(unwrap_policy(policy).state_dict(), epoch_ckpt)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
                torch.save(unwrap_policy(policy).state_dict(), run_dir / "checkpoints" / "vcd_policy_best.pth")
                save_json(val_summary, run_dir / "val_metrics_best.json")
                log(f"New best at epoch {epoch}: {best_val_acc:.4f}", rank)
            else:
                no_improve += 1

            save_json(train_logs, run_dir / "train_log.json")
            save_training_state(
                path=state_ckpt_path,
                policy=policy,
                optimizer=optimizer,
                trainer=trainer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                no_improve=no_improve,
                train_logs=train_logs,
            )
            if epoch >= args.epochs_min and no_improve >= args.early_stop_patience:
                state["stop"] = True

            state["best_val_acc"] = best_val_acc
            state["best_epoch"] = best_epoch
            state["no_improve"] = no_improve

        state = broadcast_object(distributed, state, src=0)
        best_val_acc = state["best_val_acc"]
        best_epoch = state["best_epoch"]
        no_improve = state["no_improve"]
        if state["stop"]:
            if rank == 0:
                log(
                    f"Early stopping at epoch {epoch} (best epoch={best_epoch}, "
                    f"best_val={best_val_acc:.4f}, no_improve={no_improve})",
                    rank,
                )
            break

    barrier(distributed)
    best_ckpt = run_dir / "checkpoints" / "vcd_policy_best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        unwrap_policy(policy).load_state_dict(state)
        if rank == 0:
            log(f"Loaded best checkpoint from epoch {best_epoch}", rank)
    else:
        if rank == 0:
            log("Best checkpoint not found; using last epoch weights.", rank)

    barrier(distributed)
    if args.final_full_eval:
        val_eval_data = val_data_full
        test_eval_data = test_data_full
        val_eval_item_costs = val_item_costs_full
        test_eval_item_costs = test_item_costs_full
        val_stage = "validation_full"
        test_stage = "testing"
        save_test_logs = True
    else:
        val_eval_data = val_data_small
        test_eval_data = test_data_limited
        val_eval_item_costs = val_item_costs_small
        test_eval_item_costs = test_item_costs_limited
        val_stage = "validation"
        test_stage = "testing"
        save_test_logs = True

    val_full_summary = evaluate_dataset_distributed(
        dataset=val_eval_data,
        item_costs=val_eval_item_costs,
        split_name="val",
        stage_name=val_stage,
        epoch=None,
        model=model,
        processor=processor,
        policy=policy,
        tools_embeddings=tools_embeddings,
        tools_dict=tools_dict,
        patch_processor=patch_processor,
        motion_sal_extractor=motion_sal_extractor,
        visual_sal_extractor=visual_sal_extractor,
        video_dir=video_dir,
        video_index=video_index,
        use_video_index=args.use_video_index,
        frame_cache=frame_cache,
        meta_cache=meta_cache,
        cache_dir=val_cache,
        allow_legacy_cache=args.allow_legacy_cache,
        max_new_tokens=args.max_new_tokens,
        single_token_mode=args.single_token_mode,
        candidate_token_ids=candidate_token_ids,
        profile_timing=args.profile_timing,
        output_dir=run_dir / "val_output",
        device=device,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        save_logs=False,
        balance_by_cost=args.balance_by_cost,
        sample_timeout_sec=args.sample_timeout_sec,
    )

    test_summary = evaluate_dataset_distributed(
        dataset=test_eval_data,
        item_costs=test_eval_item_costs,
        split_name="test",
        stage_name=test_stage,
        epoch=None,
        model=model,
        processor=processor,
        policy=policy,
        tools_embeddings=tools_embeddings,
        tools_dict=tools_dict,
        patch_processor=patch_processor,
        motion_sal_extractor=motion_sal_extractor,
        visual_sal_extractor=visual_sal_extractor,
        video_dir=video_dir,
        video_index=video_index,
        use_video_index=args.use_video_index,
        frame_cache=frame_cache,
        meta_cache=meta_cache,
        cache_dir=test_cache,
        allow_legacy_cache=args.allow_legacy_cache,
        max_new_tokens=args.max_new_tokens,
        single_token_mode=args.single_token_mode,
        candidate_token_ids=candidate_token_ids,
        profile_timing=args.profile_timing,
        output_dir=run_dir / "test_output",
        device=device,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        save_logs=save_test_logs,
        balance_by_cost=args.balance_by_cost,
        sample_timeout_sec=args.sample_timeout_sec,
    )

    if rank == 0:
        if val_full_summary is not None:
            save_json(val_full_summary, run_dir / "val_metrics_final.json")
        save_json(test_summary, run_dir / "test_metrics.json")

        full_timing = {
            "train_epochs": train_logs,
            "val_best_epoch": best_epoch,
            "val_best_acc_small": best_val_acc,
            "val_final_timing": (val_full_summary or {}).get("timing", {}),
            "test_timing": test_summary.get("timing", {}),
        }
        save_json(full_timing, run_dir / "timing_breakdown_full.json")

        paper_ref = paper_reference()
        report = {
            "run_id": run_id,
            "test_metrics": test_summary,
            "paper_ref": paper_ref,
            "diff_vs_paper": {
                "s_ynqa": test_summary.get("s_ynqa", 0.0) - paper_ref["table5_qwen3_8b"]["s_ynqa"],
                "m_ynqa": test_summary.get("m_ynqa", 0.0) - paper_ref["table5_qwen3_8b"]["m_ynqa"],
                "s_mcqa": test_summary.get("s_mcqa", 0.0) - paper_ref["table5_qwen3_8b"]["s_mcqa"],
                "m_mcqa": test_summary.get("m_mcqa", 0.0) - paper_ref["table5_qwen3_8b"]["m_mcqa"],
                "average_acc": test_summary.get("average_acc", 0.0)
                - paper_ref["table5_qwen3_8b"]["average_acc"],
                "latency_sec_per_query": test_summary.get("timing", {})
                .get("mean_sec_per_query", {})
                .get("total", 0.0)
                - paper_ref["table14_latency_sec_per_query"],
            },
        }
        save_json(report, run_dir / "paper_comparison.json")
        log(f"Finished. Outputs saved to: {run_dir}", rank)

    barrier(distributed)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
