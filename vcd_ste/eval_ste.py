import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vcd_new.utils import (
    answer_question_negative,
    answer_question_original,
    answer_question_positive,
    read_video,
    save_video_to_temp,
)
from vcd_ste.core import (
    CACHE_STATS_KEYS,
    DINOv3SaliencyExtractor,
    LRUCache,
    MotionSaliencyExtractor,
    PatchProcessor,
    apply_limit,
    build_candidate_token_ids,
    build_tools,
    build_video_index,
    collect_summary_from_logs,
    compute_saliency_and_cache,
    constrained_pred_token,
    cumulative_threshold_hard_mask,
    ensure_dir,
    get_cached_saliency,
    load_embeddings,
    load_qa_data,
    normalize_answer,
    parse_qa_item,
    prepare_tool_matrix,
    qa_uid,
    resolve_video_path,
    sample_watchdog,
    save_json,
    seed_everything,
    str2bool,
    subset_by_type,
    tool_names_from_mask,
)
from vcd_ste.models import STESelectorMLP, SampleBetaGater


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate STE-based TriCD ablation on MyBench.")
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
        "--tools_embeddings",
        type=str,
        default=str(PROJECT_ROOT / "vcd_new" / "tools" / "tools_embeddings_qwen3vl.pkl"),
    )
    parser.add_argument(
        "--dino_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "DINOv3" / "dinov3-vitl16-pretrain-lvd1689m"),
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(PROJECT_ROOT / "vcd_new" / "train" / "saliency_cache"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "vcd_ste" / "runs"),
    )
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--smoke_per_type", type=int, default=0)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument(
        "--skip_logs_path",
        type=str,
        default="",
        help="Optional log json path for skip-only uid filtering (not loaded into current logs).",
    )

    parser.add_argument("--sample_timeout_sec", type=float, default=300.0)
    parser.add_argument("--use_video_index", type=str2bool, default=True)
    parser.add_argument("--allow_legacy_cache", type=str2bool, default=True)
    parser.add_argument("--lru_cache_size", type=int, default=16)

    parser.add_argument("--tool_threshold", type=float, default=0.4)
    parser.add_argument("--min_selected_tools", type=int, default=1)
    parser.add_argument("--max_selected_tools", type=int, default=-1)
    parser.add_argument("--ste_temperature", type=float, default=1.0)

    parser.add_argument("--single_token_mode", type=str2bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1)

    parser.add_argument("--selector_hidden_dim", type=int, default=1024)
    parser.add_argument("--gater_hidden_dim", type=int, default=256)
    return parser


def load_ste_checkpoint(ckpt_path: str, selector, gater, device):
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    selector.load_state_dict(state["selector_state"])
    gater.load_state_dict(state["gater_state"])
    return state


def infer_single_item(
    item,
    split_name: str,
    model,
    processor,
    selector,
    gater,
    tools_dict,
    tool_names,
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
    tool_threshold: float,
    min_selected_tools: int,
    max_selected_tools: int,
    device,
    cache_stats: dict,
):
    parsed = parse_qa_item(item)
    if parsed is None:
        return None, "invalid_qa"
    q_type, qa_id, question, gt_answer, options = parsed
    video_id = str(item["video_id"])
    video_path = resolve_video_path(video_id, video_dir, video_index, use_video_index)
    if video_path is None:
        return None, "missing_video"

    stage_times = {}
    t_item_start = time.perf_counter()

    t0 = time.perf_counter()
    with torch.inference_mode():
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
    with torch.no_grad():
        router_logits = selector(state)
        probs = torch.sigmoid(router_logits)
        hard_mask, _, sorted_indices = cumulative_threshold_hard_mask(
            probs=probs,
            threshold=tool_threshold,
            min_selected_tools=min_selected_tools,
            max_selected_tools=max_selected_tools,
        )
        beta_pred = gater(state).view(())
    selected_tools = tool_names_from_mask(tool_names, hard_mask)
    ranked_scores = [(tool_names[int(i)], float(probs[int(i)].item())) for i in sorted_indices.tolist()]
    stage_times["tool_select"] = time.perf_counter() - t0

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
        with torch.inference_mode():
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

    alpha = beta_pred.float().clamp(0.0, 1.0)
    combined_raw = alpha * cache_data["w_m"] + (1.0 - alpha) * cache_data["w_v"]
    final_weights = torch.sigmoid(combined_raw)
    stage_times["saliency"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.inference_mode():
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
        "status": "ok",
        "qa_uid": f"{q_type}:{qa_id}:{video_id}",
        "qa_id": int(qa_id),
        "video_id": video_id,
        "type": q_type,
        "gt": str(gt_answer),
        "pred": str(pred_token if single_token_mode else pred_answer),
        "gt_token": gt_token,
        "pred_token": pred_token,
        "is_correct": bool(gt_token == pred_token),
        "single_token_fallback": bool(fallback_used),
        "tool_threshold": round(float(tool_threshold), 6),
        "selected_tools": selected_tools,
        "tool_scores_ranked": [[name, round(score, 6)] for name, score in ranked_scores],
        "beta_pred": round(float(alpha.item()), 6),
        "timing_sec": {k: round(v, 6) for k, v in stage_times.items()},
    }
    return result, None


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    if args.single_token_mode and args.max_new_tokens != 1:
        print("[Info] single_token_mode enabled, forcing max_new_tokens=1")
        args.max_new_tokens = 1

    args.num_shards = int(args.num_shards)
    args.shard_id = int(args.shard_id)
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards - 1}], got {args.shard_id}")

    output_suffix = str(args.output_suffix or "").strip()
    suffix_tag = f"_{output_suffix}" if output_suffix else ""

    run_id = args.run_id if args.run_id else datetime.now().strftime("vcd_ste_eval_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    ensure_dir(run_dir)
    save_json(vars(args), run_dir / f"eval_config{suffix_tag}.json")

    print("Loading dataset...")
    dataset_root = Path(args.dataset_root)
    data = load_qa_data(str(dataset_root / args.split), shuffle=False)
    data = subset_by_type(data, args.smoke_per_type)
    data = apply_limit(data, args.limit)
    if args.num_shards > 1:
        before = len(data)
        data = [item for i, item in enumerate(data) if (i % args.num_shards) == args.shard_id]
        print(f"Shard split: shard_id={args.shard_id}/{args.num_shards}, shard_size={len(data)} (from {before})")
    print(f"Dataset split={args.split}, size={len(data)}")

    print("Loading model and processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained(args.model_dir)
    candidate_token_ids = build_candidate_token_ids(processor)
    primary_device = next(model.parameters()).device
    print(f"Primary device: {primary_device}")

    print("Loading tools, model heads and saliency extractors...")
    tools_embeddings = load_embeddings(args.tools_embeddings, primary_device)
    tools_dict = build_tools()
    tool_names, _ = prepare_tool_matrix(tools_embeddings, primary_device)

    selector = STESelectorMLP(
        embed_dim=4096,
        hidden_dim=args.selector_hidden_dim,
        num_tools=len(tool_names),
    ).to(primary_device)
    gater = SampleBetaGater(
        embed_dim=4096,
        hidden_dim=args.gater_hidden_dim,
    ).to(primary_device)
    ckpt_meta = load_ste_checkpoint(args.checkpoint_path, selector, gater, primary_device)
    selector.eval()
    gater.eval()
    print(f"Loaded checkpoint epoch={ckpt_meta.get('epoch')}")

    motion_sal_extractor = MotionSaliencyExtractor()
    visual_sal_extractor = DINOv3SaliencyExtractor(
        checkpoints=args.dino_dir,
        device=str(primary_device),
    )
    patch_processor = PatchProcessor()

    video_dir = Path(args.video_dir)
    video_index = build_video_index(video_dir) if args.use_video_index else None
    if video_index is not None:
        print(f"Video index size: {len(video_index)}")

    cache_dir = Path(args.cache_root) / args.split
    ensure_dir(cache_dir)

    logs_path = run_dir / f"{args.split}_logs{suffix_tag}.json"
    metrics_path = run_dir / f"{args.split}_metrics{suffix_tag}.json"

    logs = []
    processed = set()
    skip_processed = set()
    if args.resume and logs_path.exists():
        with open(logs_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        for row in logs:
            uid = row.get("qa_uid")
            if uid:
                processed.add(uid)
        print(f"Resumed logs: {len(logs)} entries")

    if args.skip_logs_path:
        skip_path = Path(args.skip_logs_path)
        if skip_path.exists():
            with open(skip_path, "r", encoding="utf-8") as f:
                skip_rows = json.load(f)
            for row in skip_rows:
                uid = row.get("qa_uid")
                if uid and uid not in processed:
                    skip_processed.add(uid)
            print(f"Skip-only processed uids loaded: {len(skip_processed)} from {skip_path}")
        else:
            print(f"[Warn] skip_logs_path does not exist: {skip_path}")

    pending_items = []
    skip_hits = 0
    for item in data:
        uid = qa_uid(item)
        if uid in skip_processed:
            skip_hits += 1
            continue
        if uid is None or uid not in processed:
            pending_items.append(item)
    print(f"Pending items: {len(pending_items)} (skip_hits={skip_hits})")

    frame_cache = LRUCache(args.lru_cache_size)
    meta_cache = LRUCache(args.lru_cache_size)
    cache_stats = {k: 0 for k in CACHE_STATS_KEYS}
    extra_errors = 0
    extra_timeouts = 0

    pbar = tqdm(pending_items, desc=f"Eval-{args.split}")
    for idx, item in enumerate(pbar, start=1):
        uid = qa_uid(item)
        try:
            with sample_watchdog(args.sample_timeout_sec):
                result, err = infer_single_item(
                    item=item,
                    split_name=args.split,
                    model=model,
                    processor=processor,
                    selector=selector,
                    gater=gater,
                    tools_dict=tools_dict,
                    tool_names=tool_names,
                    patch_processor=patch_processor,
                    motion_sal_extractor=motion_sal_extractor,
                    visual_sal_extractor=visual_sal_extractor,
                    video_dir=video_dir,
                    video_index=video_index,
                    use_video_index=args.use_video_index,
                    frame_cache=frame_cache,
                    meta_cache=meta_cache,
                    cache_dir=cache_dir,
                    allow_legacy_cache=args.allow_legacy_cache,
                    max_new_tokens=args.max_new_tokens,
                    single_token_mode=args.single_token_mode,
                    candidate_token_ids=candidate_token_ids,
                    tool_threshold=args.tool_threshold,
                    min_selected_tools=args.min_selected_tools,
                    max_selected_tools=args.max_selected_tools,
                    device=primary_device,
                    cache_stats=cache_stats,
                )
            if result is None:
                logs.append({"status": "error", "qa_uid": uid, "error": err or "unknown"})
                extra_errors += 1
            else:
                logs.append(result)
        except TimeoutError:
            logs.append({"status": "error", "qa_uid": uid, "error": "timeout"})
            extra_errors += 1
            extra_timeouts += 1
        except Exception as e:
            logs.append({"status": "error", "qa_uid": uid, "error": str(e)})
            extra_errors += 1

        if args.save_every > 0 and idx % args.save_every == 0:
            summary = collect_summary_from_logs(logs)
            summary["num_errors"] = int(summary.get("num_errors", 0) + extra_errors)
            summary["num_timeouts"] = int(summary.get("num_timeouts", 0) + extra_timeouts)
            summary["cache_stats"] = {k: int(cache_stats[k]) for k in CACHE_STATS_KEYS}
            summary["progress"] = {"done": len(logs) + skip_hits, "total": len(data)}
            save_json(logs, logs_path)
            save_json(summary, metrics_path)

    summary = collect_summary_from_logs(logs)
    summary["num_errors"] = int(summary.get("num_errors", 0) + extra_errors)
    summary["num_timeouts"] = int(summary.get("num_timeouts", 0) + extra_timeouts)
    summary["cache_stats"] = {k: int(cache_stats[k]) for k in CACHE_STATS_KEYS}
    summary["num_samples"] = len(logs)
    summary["run_id"] = run_id
    summary["split"] = args.split
    summary["checkpoint_path"] = args.checkpoint_path
    summary["config"] = {
        "tool_threshold": float(args.tool_threshold),
        "min_selected_tools": int(args.min_selected_tools),
        "max_selected_tools": int(args.max_selected_tools),
        "single_token_mode": bool(args.single_token_mode),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "output_suffix": output_suffix,
    }
    summary["progress"] = {"done": len(logs) + skip_hits, "total": len(data)}

    save_json(logs, logs_path)
    save_json(summary, metrics_path)

    print("\n=== Final Summary ===")
    for q_type in ("s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"):
        c = summary["counts"][q_type]["correct"]
        t = summary["counts"][q_type]["total"]
        print(f"{q_type}: {summary[q_type]:.4f} ({c}/{t})")
    print(
        f"micro_overall_acc: {summary['micro_overall_acc']:.4f} "
        f"({summary['micro_counts']['correct']}/{summary['micro_counts']['total']})"
    )
    print(f"macro_average_acc: {summary['macro_average_acc']:.4f}")
    print(f"metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
