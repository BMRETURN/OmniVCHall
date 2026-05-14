import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
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
    compute_saliency_and_cache,
    confidence_to_beta_target,
    constrained_pred_token,
    cosine_teacher_mask,
    cumulative_threshold_hard_mask,
    ensure_dir,
    first_step_target_confidence,
    get_cached_saliency,
    load_embeddings,
    load_train_subset_manifest,
    normalize_answer,
    parse_qa_item,
    prepare_tool_matrix,
    resolve_video_path,
    sample_watchdog,
    save_json,
    seed_everything,
    ste_hard_mask,
    str2bool,
    tool_names_from_mask,
)
from vcd_ste.models import STESelectorMLP, SampleBetaGater


def build_parser():
    parser = argparse.ArgumentParser(description="Train STE router + sample gater for TriCD ablation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "MyBench" / "all_video"),
    )
    parser.add_argument(
        "--train_subset_manifest",
        type=str,
        default=str(PROJECT_ROOT / "vcd_new" / "runs" / "splits" / "train_subset_1800_seed2025.json"),
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

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--sample_timeout_sec", type=float, default=300.0)
    parser.add_argument("--save_every", type=int, default=20)

    parser.add_argument("--tool_threshold", type=float, default=0.4)
    parser.add_argument("--min_selected_tools", type=int, default=1)
    parser.add_argument("--max_selected_tools", type=int, default=-1)
    parser.add_argument("--ste_temperature", type=float, default=1.0)

    parser.add_argument("--single_token_mode", type=str2bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--allow_legacy_cache", type=str2bool, default=True)
    parser.add_argument("--use_video_index", type=str2bool, default=True)
    parser.add_argument("--lru_cache_size", type=int, default=16)

    parser.add_argument("--selector_hidden_dim", type=int, default=1024)
    parser.add_argument("--gater_hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_router_weight", type=float, default=1.0)
    parser.add_argument("--loss_beta_weight", type=float, default=1.0)
    parser.add_argument("--loss_ste_weight", type=float, default=0.0)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    return parser


def maybe_load_checkpoint(path: str, selector, gater, optimizer, device):
    if not path:
        return 1, []
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return 1, []
    state = torch.load(ckpt_path, map_location=device)
    selector.load_state_dict(state["selector_state"])
    gater.load_state_dict(state["gater_state"])
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    start_epoch = int(state.get("epoch", 0)) + 1
    history = state.get("history", [])
    return start_epoch, history


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    if args.single_token_mode and args.max_new_tokens != 1:
        print("[Info] single_token_mode enabled, forcing max_new_tokens=1")
        args.max_new_tokens = 1

    run_id = args.run_id if args.run_id else datetime.now().strftime("vcd_ste_train_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    save_json(vars(args), run_dir / "train_config.json")

    print("Loading training subset manifest...")
    train_data, manifest = load_train_subset_manifest(args.train_subset_manifest)
    train_data = apply_limit(train_data, args.train_limit)
    print(f"Train samples: {len(train_data)}")
    save_json({"manifest_keys": list(manifest.keys()), "train_size": len(train_data)}, run_dir / "train_manifest_meta.json")

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

    print("Loading tools and saliency extractors...")
    tools_embeddings = load_embeddings(args.tools_embeddings, primary_device)
    tools_dict = build_tools()
    tool_names, tool_matrix = prepare_tool_matrix(tools_embeddings, primary_device)
    motion_sal_extractor = MotionSaliencyExtractor()
    visual_sal_extractor = DINOv3SaliencyExtractor(
        checkpoints=args.dino_dir,
        device=str(primary_device),
    )
    patch_processor = PatchProcessor()

    selector = STESelectorMLP(
        embed_dim=4096,
        hidden_dim=args.selector_hidden_dim,
        num_tools=len(tool_names),
    ).to(primary_device)
    gater = SampleBetaGater(
        embed_dim=4096,
        hidden_dim=args.gater_hidden_dim,
    ).to(primary_device)
    optimizer = torch.optim.AdamW(
        list(selector.parameters()) + list(gater.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch, history = maybe_load_checkpoint(
        path=args.resume_checkpoint,
        selector=selector,
        gater=gater,
        optimizer=optimizer,
        device=primary_device,
    )

    video_dir = Path(args.video_dir)
    video_index = build_video_index(video_dir) if args.use_video_index else None
    if video_index is not None:
        print(f"Video index size: {len(video_index)}")

    cache_dir = Path(args.cache_root) / "train"
    ensure_dir(cache_dir)
    frame_cache = LRUCache(args.lru_cache_size)
    meta_cache = LRUCache(args.lru_cache_size)

    for epoch in range(start_epoch, args.epochs + 1):
        selector.train()
        gater.train()

        epoch_logs = []
        cache_stats = {k: 0 for k in CACHE_STATS_KEYS}
        num_errors = 0
        num_timeouts = 0
        total_loss = 0.0
        total_router_loss = 0.0
        total_beta_loss = 0.0
        total_correct = 0
        total_seen = 0

        pbar = tqdm(train_data, desc=f"Train-E{epoch}")
        for step_idx, item in enumerate(pbar, start=1):
            try:
                with sample_watchdog(args.sample_timeout_sec):
                    parsed = parse_qa_item(item)
                    if parsed is None:
                        continue
                    q_type, qa_id, question, gt_answer, options = parsed
                    video_id = str(item["video_id"])
                    video_path = resolve_video_path(video_id, video_dir, video_index, args.use_video_index)
                    if video_path is None:
                        continue

                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        orig_logits, last_hidden_states, _ = answer_question_original(
                            model=model,
                            processor=processor,
                            video_path=video_path,
                            question=question,
                            options=options,
                            max_new_tokens=args.max_new_tokens,
                        )
                    state = last_hidden_states.detach().to(device=primary_device, dtype=torch.float32)

                    router_logits = selector(state)
                    probs = torch.sigmoid(router_logits)
                    hard_mask, _, _ = cumulative_threshold_hard_mask(
                        probs=probs,
                        threshold=args.tool_threshold,
                        min_selected_tools=args.min_selected_tools,
                        max_selected_tools=args.max_selected_tools,
                    )
                    ste_mask = ste_hard_mask(probs, hard_mask)

                    teacher_mask, _, _ = cosine_teacher_mask(
                        state_embeddings=state,
                        tool_matrix=tool_matrix,
                        threshold=args.tool_threshold,
                        min_selected_tools=args.min_selected_tools,
                        max_selected_tools=args.max_selected_tools,
                    )
                    teacher_mask = teacher_mask.to(device=primary_device, dtype=torch.float32)

                    frames = frame_cache.get(video_path)
                    if frames is None:
                        frames = read_video(video_path)
                        frame_cache.put(video_path, frames)

                    selected_tools = tool_names_from_mask(tool_names, hard_mask.detach())
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
                                max_new_tokens=args.max_new_tokens,
                            )
                    finally:
                        if os.path.exists(neg_path):
                            os.remove(neg_path)

                    cache_data, save_path = get_cached_saliency(
                        split_name="train",
                        q_type=q_type,
                        qa_id=qa_id,
                        video_id=video_id,
                        cache_dir=cache_dir,
                        device=primary_device,
                        allow_legacy_cache=args.allow_legacy_cache,
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
                            device=primary_device,
                            cache_stats=cache_stats,
                        )

                    beta_pred = gater(state).view(())
                    combined_raw = beta_pred * cache_data["w_m"] + (1.0 - beta_pred) * cache_data["w_v"]
                    weights = torch.sigmoid(combined_raw)

                    with torch.inference_mode():
                        pred_answer, generated_ids = answer_question_positive(
                            model=model,
                            processor=processor,
                            video_path=video_path,
                            patch_weights=weights,
                            question=question,
                            original_logits=orig_logits,
                            negative_logits=neg_logits,
                            options=options,
                            max_new_tokens=args.max_new_tokens,
                        )

                    gt_token = normalize_answer(gt_answer)
                    pred_token, fallback_used = constrained_pred_token(
                        raw_pred=pred_answer,
                        generated_ids=generated_ids,
                        options=options,
                        single_token_mode=args.single_token_mode,
                        candidate_token_ids=candidate_token_ids,
                    )

                    with torch.inference_mode():
                        motion_weights = torch.sigmoid(cache_data["w_m"])
                        visual_weights = torch.sigmoid(cache_data["w_v"])
                        _, gen_motion = answer_question_positive(
                            model=model,
                            processor=processor,
                            video_path=video_path,
                            patch_weights=motion_weights,
                            question=question,
                            original_logits=orig_logits,
                            negative_logits=neg_logits,
                            options=options,
                            max_new_tokens=args.max_new_tokens,
                        )
                        _, gen_visual = answer_question_positive(
                            model=model,
                            processor=processor,
                            video_path=video_path,
                            patch_weights=visual_weights,
                            question=question,
                            original_logits=orig_logits,
                            negative_logits=neg_logits,
                            options=options,
                            max_new_tokens=args.max_new_tokens,
                        )

                    conf_motion = first_step_target_confidence(gen_motion, gt_token, candidate_token_ids)
                    conf_visual = first_step_target_confidence(gen_visual, gt_token, candidate_token_ids)
                    beta_target = torch.tensor(
                        confidence_to_beta_target(conf_motion, conf_visual),
                        device=primary_device,
                        dtype=torch.float32,
                    )

                    loss_router = F.binary_cross_entropy_with_logits(router_logits, teacher_mask)
                    loss_beta = F.mse_loss(beta_pred.float(), beta_target)
                    loss = args.loss_router_weight * loss_router + args.loss_beta_weight * loss_beta
                    if args.loss_ste_weight > 0:
                        loss = loss + args.loss_ste_weight * F.mse_loss(ste_mask.float(), teacher_mask.float())

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(selector.parameters()) + list(gater.parameters()),
                            max_norm=args.grad_clip,
                        )
                    optimizer.step()

                    total_loss += float(loss.item())
                    total_router_loss += float(loss_router.item())
                    total_beta_loss += float(loss_beta.item())
                    total_seen += 1
                    total_correct += int(pred_token == gt_token)

                    epoch_logs.append(
                        {
                            "status": "ok",
                            "epoch": epoch,
                            "step": step_idx,
                            "qa_id": int(qa_id),
                            "video_id": video_id,
                            "type": q_type,
                            "gt_token": gt_token,
                            "pred_token": pred_token,
                            "is_correct": bool(pred_token == gt_token),
                            "single_token_fallback": bool(fallback_used),
                            "selected_tools": selected_tools,
                            "beta_pred": float(beta_pred.detach().item()),
                            "beta_target": float(beta_target.detach().item()),
                            "loss": float(loss.item()),
                            "loss_router": float(loss_router.item()),
                            "loss_beta": float(loss_beta.item()),
                            "cost_sec": round(time.perf_counter() - t0, 4),
                        }
                    )

                    avg_loss = total_loss / max(total_seen, 1)
                    avg_acc = total_correct / max(total_seen, 1)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            except TimeoutError:
                num_timeouts += 1
                num_errors += 1
                epoch_logs.append({"status": "error", "epoch": epoch, "step": step_idx, "error": "timeout"})
            except Exception as e:
                num_errors += 1
                epoch_logs.append({"status": "error", "epoch": epoch, "step": step_idx, "error": str(e)})

            if args.save_every > 0 and step_idx % args.save_every == 0:
                interim = {
                    "epoch": epoch,
                    "step": step_idx,
                    "seen": total_seen,
                    "accuracy": (total_correct / max(total_seen, 1)),
                    "avg_loss": (total_loss / max(total_seen, 1)),
                    "avg_loss_router": (total_router_loss / max(total_seen, 1)),
                    "avg_loss_beta": (total_beta_loss / max(total_seen, 1)),
                    "num_errors": num_errors,
                    "num_timeouts": num_timeouts,
                    "cache_stats": {k: int(cache_stats[k]) for k in CACHE_STATS_KEYS},
                }
                save_json(interim, run_dir / "train_interim_metrics.json")
                save_json(epoch_logs, run_dir / f"train_epoch_{epoch}_logs.json")

        epoch_summary = {
            "epoch": epoch,
            "num_seen": total_seen,
            "accuracy": (total_correct / max(total_seen, 1)),
            "avg_loss": (total_loss / max(total_seen, 1)),
            "avg_loss_router": (total_router_loss / max(total_seen, 1)),
            "avg_loss_beta": (total_beta_loss / max(total_seen, 1)),
            "num_errors": num_errors,
            "num_timeouts": num_timeouts,
            "cache_stats": {k: int(cache_stats[k]) for k in CACHE_STATS_KEYS},
        }
        history.append(epoch_summary)

        save_json(epoch_summary, run_dir / f"train_epoch_{epoch}_metrics.json")
        save_json(epoch_logs, run_dir / f"train_epoch_{epoch}_logs.json")
        save_json(history, run_dir / "train_history.json")

        ckpt = {
            "epoch": epoch,
            "selector_state": selector.state_dict(),
            "gater_state": gater.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
        }
        ckpt_path = ckpt_dir / f"ste_epoch_{epoch}.pth"
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, ckpt_dir / "ste_latest.pth")
        print(f"[Epoch {epoch}] checkpoint saved: {ckpt_path}")

    final_summary = history[-1] if history else {}
    save_json(final_summary, run_dir / "train_final_metrics.json")
    print("\n=== Training Finished ===")
    print(final_summary)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
