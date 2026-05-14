import argparse
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
from vcd_new.utils import (
    PatchProcessor,
    answer_question_negative,
    answer_question_original,
    answer_question_positive,
    load_embeddings,
    load_qa_data,
    read_video,
    save_video_to_temp,
    transform_pixel_to_patch,
)


VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
QUESTION_TYPES = ("s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa")
TIMING_KEYS = ("orig_gen", "policy", "neg_branch", "saliency", "pos_gen", "total")


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def seed_everything(seed):
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

    def add(self, tdict):
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
        return "s_ynqa", item["s_ynqa_id"], item["yn_question"], item["yn_answer"], None
    if "m_ynqa_id" in item:
        return "m_ynqa", item["m_ynqa_id"], item["yn_question"], item["yn_answer"], None
    if "s_mcqa_id" in item:
        return "s_mcqa", item["s_mcqa_id"], item["mc_question"], item["mc_answer"], item.get("mc_option")
    if "m_mcqa_id" in item:
        return "m_mcqa", item["m_mcqa_id"], item["mc_question"], item["mc_answer"], item.get("mc_option")
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


def build_cache_key(split_name, q_type, qa_id, video_id):
    return f"{split_name}_{q_type}_{qa_id}_{video_id}"


def _load_cache_tensor(path, device, expected_video_id):
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
    split_name,
    q_type,
    qa_id,
    video_id,
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
    video_path,
    video_id,
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
    grid_t, grid_h, grid_w, h_bar, w_bar = patch_processor.get_smart_resize_grid(
        len(indices), metadata["height"], metadata["width"]
    )

    with torch.no_grad():
        motion_np = np.asarray(
            motion_sal_extractor.extract_motion_saliency(frames, indices=indices.tolist()),
            dtype=np.float32,
        )
        visual_np = np.asarray(
            visual_sal_extractor.extract_dino_video_pixel_last(frames, indices=indices.tolist()),
            dtype=np.float32,
        )
        motion_sal = torch.tensor(
            motion_np,
            dtype=torch.float32,
            device=device,
        )
        visual_sal = torch.tensor(
            visual_np,
            dtype=torch.float32,
            device=device,
        )
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


def infer_single_item(
    item,
    split_name,
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
    use_video_index,
    frame_cache,
    meta_cache,
    cache_dir: Path,
    allow_legacy_cache: bool,
    max_new_tokens: int,
    device,
    is_train: bool,
    train_std_dev: float,
    cache_stats: dict,
):
    parsed = parse_qa_item(item)
    if parsed is None:
        return None, "invalid_qa", None
    q_type, qa_id, question, gt_answer, options = parsed
    video_id = item["video_id"]

    video_path = resolve_video_path(video_id, video_dir, video_index, use_video_index)
    if video_path is None:
        return None, "missing_video", None

    stage_times = {}
    t_item_start = time.perf_counter()

    t0 = time.perf_counter()
    orig_logits, last_hidden_states, _ = answer_question_original(
        model=model,
        processor=processor,
        video_path=video_path,
        question=question,
        options=options,
        max_new_tokens=max_new_tokens,
    )
    stage_times["orig_gen"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = last_hidden_states.detach().to(device=device, dtype=torch.float32)
    selected_tools, beta_tensor, log_p_router, log_p_gater = policy.get_action_and_log_prob(
        state,
        tools_embeddings,
        std_dev=(train_std_dev if is_train else 0.0),
    )
    beta_value = float(beta_tensor.item())
    stage_times["policy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    frames = frame_cache.get(video_path)
    if frames is None:
        frames = read_video(video_path)
        frame_cache.put(video_path, frames)

    neg_frames = frames
    for tool_name in selected_tools:
        neg_frames = tools_dict[tool_name].process(neg_frames)

    neg_path = save_video_to_temp(neg_frames, video_path)
    try:
        neg_logits = answer_question_negative(
            model=model,
            processor=processor,
            negative_video_path=neg_path,
            question=question,
            options=options,
            max_new_tokens=max_new_tokens,
        )
    finally:
        if os.path.exists(neg_path):
            os.remove(neg_path)
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
    pred_answer, _ = answer_question_positive(
        model=model,
        processor=processor,
        video_path=video_path,
        patch_weights=final_weights,
        question=question,
        original_logits=orig_logits,
        negative_logits=neg_logits,
        options=options,
        max_new_tokens=max_new_tokens,
    )
    stage_times["pos_gen"] = time.perf_counter() - t0

    stage_times["total"] = time.perf_counter() - t_item_start

    result = {
        "qa_id": int(qa_id),
        "video_id": video_id,
        "type": q_type,
        "gt": str(gt_answer),
        "pred": str(pred_answer),
        "gt_token": normalize_answer(gt_answer),
        "pred_token": normalize_answer(pred_answer),
        "is_correct": normalize_answer(gt_answer) == normalize_answer(pred_answer),
        "beta": round(beta_value, 6),
        "tools": selected_tools,
        "timing_sec": {k: round(v, 6) for k, v in stage_times.items()},
    }
    return result, None, (log_p_router, log_p_gater)


def summarize_metrics(results):
    metrics = {k: {"correct": 0, "total": 0} for k in QUESTION_TYPES}
    for x in results:
        q_type = x["type"]
        metrics[q_type]["total"] += 1
        if x["is_correct"]:
            metrics[q_type]["correct"] += 1

    summary = {}
    total_acc_sum = 0.0
    valid_types = 0
    for q_type in QUESTION_TYPES:
        total = metrics[q_type]["total"]
        if total > 0:
            acc = metrics[q_type]["correct"] / total
            summary[q_type] = acc
            total_acc_sum += acc
            valid_types += 1
        else:
            summary[q_type] = 0.0
    summary["average_acc"] = total_acc_sum / valid_types if valid_types > 0 else 0.0
    summary["counts"] = metrics
    return summary


def evaluate_dataset(
    dataset,
    split_name,
    stage_name,
    epoch,
    model,
    processor,
    policy,
    tools_embeddings,
    tools_dict,
    patch_processor,
    motion_sal_extractor,
    visual_sal_extractor,
    video_dir,
    video_index,
    use_video_index,
    frame_cache,
    meta_cache,
    cache_dir,
    allow_legacy_cache,
    max_new_tokens,
    profile_timing,
    output_dir,
    device,
):
    policy.eval()
    results = []
    timing = TimingAccumulator()
    cache_stats = {
        "hit_v2": 0,
        "hit_legacy": 0,
        "miss": 0,
        "saved_v2": 0,
        "legacy_mismatch": 0,
        "legacy_invalid": 0,
        "v2_mismatch": 0,
        "v2_invalid": 0,
    }
    skipped = 0

    pbar = tqdm(dataset, desc=stage_name)
    for item in pbar:
        try:
            result, err, _ = infer_single_item(
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
                device=device,
                is_train=False,
                train_std_dev=0.0,
                cache_stats=cache_stats,
            )
            if result is None:
                skipped += 1
                continue
            results.append(result)
            if profile_timing:
                timing.add(result["timing_sec"])
            if len(results) > 0:
                pbar.set_postfix({"avg_acc": f"{np.mean([r['is_correct'] for r in results]):.3f}"})
        except Exception as e:
            skipped += 1
            print(f"[{stage_name}] error: {e}")

    summary = summarize_metrics(results)
    summary["stage_name"] = stage_name
    summary["epoch"] = epoch
    summary["num_samples"] = len(results)
    summary["num_skipped"] = skipped
    summary["cache_stats"] = cache_stats
    summary["timing"] = timing.to_dict()

    prefix = stage_name.lower()
    if epoch is not None:
        prefix = f"{prefix}_epoch_{epoch}"
    save_json(summary, output_dir / f"{prefix}_metrics.json")
    save_json(results, output_dir / f"{prefix}_logs.json")
    return summary


def run_train_epoch(
    dataset,
    epoch,
    model,
    processor,
    policy,
    trainer,
    tools_embeddings,
    tools_dict,
    patch_processor,
    motion_sal_extractor,
    visual_sal_extractor,
    video_dir,
    video_index,
    use_video_index,
    frame_cache,
    meta_cache,
    cache_dir,
    allow_legacy_cache,
    max_new_tokens,
    profile_timing,
    train_std_dev,
    device,
):
    policy.train()
    rewards = []
    losses = []
    timing = TimingAccumulator()
    cache_stats = {
        "hit_v2": 0,
        "hit_legacy": 0,
        "miss": 0,
        "saved_v2": 0,
        "legacy_mismatch": 0,
        "legacy_invalid": 0,
        "v2_mismatch": 0,
        "v2_invalid": 0,
    }
    skipped = 0

    pbar = tqdm(dataset, desc=f"Training-E{epoch}")
    for item in pbar:
        try:
            result, err, log_probs = infer_single_item(
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
                device=device,
                is_train=True,
                train_std_dev=train_std_dev,
                cache_stats=cache_stats,
            )
            if result is None:
                skipped += 1
                continue

            reward = trainer.compute_reward(result["pred"], result["gt"])
            router_log_prob, gater_log_prob = log_probs
            loss = trainer.step(reward, router_log_prob, gater_log_prob)

            rewards.append(float(reward))
            if loss != 0:
                losses.append(float(loss))
            if profile_timing:
                timing.add(result["timing_sec"])

            avg_r = np.mean(rewards) if rewards else 0.0
            pbar.set_postfix({"avg_reward": f"{avg_r:.3f}", "loss": f"{loss:.4f}"})
        except Exception as e:
            skipped += 1
            print(f"[Training] error: {e}")

    return {
        "epoch": epoch,
        "num_samples": len(rewards),
        "num_skipped": skipped,
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "avg_loss": float(np.mean(losses) if losses else 0.0),
        "cache_stats": cache_stats,
        "timing": timing.to_dict(),
    }


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


def build_parser():
    parser = argparse.ArgumentParser(description="TriCD training entry for vcd_new")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "Qwen3-VL-8B-Instruct"),
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
        default=str(Path(__file__).resolve().parents[1] / "tools" / "tools_embeddings_qwen3vl.pkl"),
    )
    parser.add_argument(
        "--dino_dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "checkpoints"
            / "DINOv3"
            / "dinov3-vitl16-pretrain-lvd1689m"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")

    parser.add_argument("--epochs_max", type=int, default=5)
    parser.add_argument("--epochs_min", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--val_limit", type=int, default=-1)
    parser.add_argument("--test_limit", type=int, default=-1)
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
    return parser


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "vcd_new" / "runs" / run_id)
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "val_output")
    ensure_dir(run_dir / "test_output")

    save_json(vars(args), run_dir / "run_config.json")

    print("Loading dataset...")
    dataset_root = Path(args.dataset_root)
    train_data = load_qa_data(str(dataset_root / "train"), shuffle=True)
    val_data = load_qa_data(str(dataset_root / "val"), shuffle=False)
    test_data = load_qa_data(str(dataset_root / "test"), shuffle=False)
    train_data = apply_limit(train_data, args.train_limit)
    val_data = apply_limit(val_data, args.val_limit)
    test_data = apply_limit(test_data, args.test_limit)
    print(f"Data loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print("Loading model and processor...")
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=model_dtype,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained(args.model_dir)

    print("Loading tools and policy...")
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
        policy.load_state_dict(torch.load(args.resume_policy_path, map_location=device))
        print(f"Loaded policy checkpoint: {args.resume_policy_path}")

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = VCDTrainer(policy, optimizer, accumulation_steps=args.batch_size)

    print("Initializing saliency extractors...")
    motion_sal_extractor = MotionSaliencyExtractor()
    visual_sal_extractor = DINOv3SaliencyExtractor(
        checkpoints=args.dino_dir,
        device=device,
    )
    patch_processor = PatchProcessor()

    video_dir = Path(args.video_dir)
    video_index = build_video_index(video_dir) if args.use_video_index else None
    if args.use_video_index:
        print(f"Video index size: {len(video_index)}")
    frame_cache = LRUCache(args.lru_cache_size)
    meta_cache = LRUCache(args.lru_cache_size)

    cache_root = Path(args.cache_root)
    train_cache = cache_root / "train"
    val_cache = cache_root / "val"
    test_cache = cache_root / "test"
    ensure_dir(train_cache)
    ensure_dir(val_cache)
    ensure_dir(test_cache)

    train_logs = []
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    print("Start training...")
    for epoch in range(1, args.epochs_max + 1):
        epoch_train = run_train_epoch(
            dataset=train_data,
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
            profile_timing=args.profile_timing,
            train_std_dev=args.train_std_dev,
            device=device,
        )
        train_logs.append(epoch_train)

        val_summary = evaluate_dataset(
            dataset=val_data,
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
            profile_timing=args.profile_timing,
            output_dir=run_dir / "val_output",
            device=device,
        )
        val_acc = float(val_summary["average_acc"])
        print(f"[Epoch {epoch}] val average_acc={val_acc:.4f}")

        epoch_ckpt = run_dir / "checkpoints" / f"vcd_policy_epoch_{epoch}.pth"
        torch.save(policy.state_dict(), epoch_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(policy.state_dict(), run_dir / "checkpoints" / "vcd_policy_best.pth")
            save_json(val_summary, run_dir / "val_metrics_best.json")
            print(f"New best at epoch {epoch}: {best_val_acc:.4f}")
        else:
            no_improve += 1

        save_json(train_logs, run_dir / "train_log.json")

        if epoch >= args.epochs_min and no_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} (best epoch={best_epoch}, "
                f"best_val={best_val_acc:.4f}, no_improve={no_improve})"
            )
            break

    best_ckpt = run_dir / "checkpoints" / "vcd_policy_best.pth"
    if best_ckpt.exists():
        policy.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best checkpoint from epoch {best_epoch}")
    else:
        print("Best checkpoint not found; evaluating with last epoch weights.")

    test_summary = evaluate_dataset(
        dataset=test_data,
        split_name="test",
        stage_name="testing",
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
        profile_timing=args.profile_timing,
        output_dir=run_dir / "test_output",
        device=device,
    )
    save_json(test_summary, run_dir / "test_metrics.json")

    full_timing = {
        "train_epochs": train_logs,
        "val_best_epoch": best_epoch,
        "val_best_acc": best_val_acc,
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
            "average_acc": test_summary.get("average_acc", 0.0) - paper_ref["table5_qwen3_8b"]["average_acc"],
            "latency_sec_per_query": test_summary.get("timing", {})
            .get("mean_sec_per_query", {})
            .get("total", 0.0)
            - paper_ref["table14_latency_sec_per_query"],
        },
    }
    save_json(report, run_dir / "paper_comparison.json")

    print(f"Finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
