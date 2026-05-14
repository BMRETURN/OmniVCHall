import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


QUESTION_TYPES = ("s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa")


def parse_q_type(item: dict) -> str:
    if "s_ynqa_id" in item:
        return "s_ynqa"
    if "m_ynqa_id" in item:
        return "m_ynqa"
    if "s_mcqa_id" in item:
        return "s_mcqa"
    if "m_mcqa_id" in item:
        return "m_mcqa"
    raise ValueError(f"Unsupported QA item keys: {list(item.keys())[:8]}")


def parse_qa_id(item: dict) -> int:
    for key in ("s_ynqa_id", "m_ynqa_id", "s_mcqa_id", "m_mcqa_id"):
        if key in item:
            return int(item[key])
    raise ValueError("No qa_id field found.")


def load_train_records(dataset_root: Path) -> Tuple[List[dict], Dict[str, int]]:
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    qa_files = sorted(train_dir.glob("*qa*.json"))
    if not qa_files:
        raise FileNotFoundError(f"No *qa*.json files under: {train_dir}")

    records: List[dict] = []
    file_sizes: Dict[str, int] = {}
    for qa_file in qa_files:
        with open(qa_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        file_sizes[qa_file.name] = len(items)
        for idx, item in enumerate(items):
            q_type = parse_q_type(item)
            qa_id = parse_qa_id(item)
            records.append(
                {
                    "source_file": qa_file.name,
                    "source_index": idx,
                    "video_id": str(item["video_id"]),
                    "q_type": q_type,
                    "qa_id": qa_id,
                    "item": item,
                }
            )
    return records, file_sizes


def compute_type_targets(records: List[dict], sample_size: int) -> Dict[str, int]:
    counts = Counter(rec["q_type"] for rec in records)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Empty train records.")

    raw_targets = {
        q_type: (counts[q_type] * float(sample_size) / float(total))
        for q_type in QUESTION_TYPES
    }
    targets = {q_type: int(math.floor(raw_targets[q_type])) for q_type in QUESTION_TYPES}
    remaining = sample_size - sum(targets.values())
    if remaining > 0:
        # Largest remainder method
        order = sorted(
            QUESTION_TYPES,
            key=lambda q: (raw_targets[q] - targets[q], counts[q]),
            reverse=True,
        )
        for i in range(remaining):
            targets[order[i % len(order)]] += 1
    return targets


def sample_video_cover_stratified(records: List[dict], sample_size: int, seed: int) -> List[dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size > len(records):
        raise ValueError(f"sample_size={sample_size} exceeds train size={len(records)}")

    rng = random.Random(seed)
    by_video = defaultdict(list)
    for rec in records:
        by_video[rec["video_id"]].append(rec)

    all_videos = sorted(by_video.keys())
    selected: List[dict] = []

    # Step 1: guarantee coverage as much as possible.
    if sample_size >= len(all_videos):
        for vid in all_videos:
            selected.append(rng.choice(by_video[vid]))
    else:
        picked_videos = rng.sample(all_videos, sample_size)
        for vid in picked_videos:
            selected.append(rng.choice(by_video[vid]))
        return selected

    uid = lambda r: (r["source_file"], int(r["source_index"]))
    selected_uids = {uid(r) for r in selected}
    remaining_pool = [r for r in records if uid(r) not in selected_uids]

    need = sample_size - len(selected)
    if need <= 0:
        return selected

    # Step 2: stratify by global q_type ratio.
    target_counts = compute_type_targets(records, sample_size)
    current_counts = Counter(r["q_type"] for r in selected)
    quota = {q: max(0, target_counts[q] - current_counts[q]) for q in QUESTION_TYPES}

    chosen_from_strata: List[dict] = []
    pool_by_type = defaultdict(list)
    for rec in remaining_pool:
        pool_by_type[rec["q_type"]].append(rec)

    for q_type in QUESTION_TYPES:
        candidates = pool_by_type[q_type]
        take = min(len(candidates), quota[q_type])
        if take > 0:
            chosen = rng.sample(candidates, take)
            chosen_from_strata.extend(chosen)
            chosen_ids = {uid(r) for r in chosen}
            pool_by_type[q_type] = [r for r in candidates if uid(r) not in chosen_ids]

    selected.extend(chosen_from_strata)
    selected_uids = {uid(r) for r in selected}

    # Step 3: fill remaining slots from train-only residual pool.
    if len(selected) < sample_size:
        residual = [r for r in records if uid(r) not in selected_uids]
        fill = sample_size - len(selected)
        selected.extend(rng.sample(residual, fill))

    # Stable ordering for reproducibility and debug.
    selected = sorted(
        selected,
        key=lambda r: (
            r["video_id"],
            r["q_type"],
            int(r["qa_id"]),
            r["source_file"],
            int(r["source_index"]),
        ),
    )
    return selected


def ratio_dict(counter: Counter, total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in QUESTION_TYPES}
    return {k: round(float(counter.get(k, 0)) / float(total), 6) for k in QUESTION_TYPES}


def build_manifest(records: List[dict], selected: List[dict], args) -> dict:
    full_type_counts = Counter(r["q_type"] for r in records)
    sample_type_counts = Counter(r["q_type"] for r in selected)
    full_total = len(records)
    sample_total = len(selected)

    full_video_count = len({r["video_id"] for r in records})
    sample_video_count = len({r["video_id"] for r in selected})

    full_ratio = ratio_dict(full_type_counts, full_total)
    sample_ratio = ratio_dict(sample_type_counts, sample_total)
    abs_diff = {k: round(abs(sample_ratio[k] - full_ratio[k]), 6) for k in QUESTION_TYPES}

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "strategy": "video_cover_stratified",
        "full_train_stats": {
            "total_qas": full_total,
            "total_videos": full_video_count,
            "type_counts": {k: int(full_type_counts.get(k, 0)) for k in QUESTION_TYPES},
            "type_ratio": full_ratio,
        },
        "sample_stats": {
            "total_qas": sample_total,
            "covered_videos": sample_video_count,
            "video_coverage_ratio": round(sample_video_count / max(full_video_count, 1), 6),
            "type_counts": {k: int(sample_type_counts.get(k, 0)) for k in QUESTION_TYPES},
            "type_ratio": sample_ratio,
            "type_abs_diff_vs_full": abs_diff,
        },
        "items": selected,
    }


def save_manifest(manifest: dict, output_manifest: Path):
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Build train subset manifest with fair sampling.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/storage/wenbinxing/Mybenchmark/dataset/MyBench",
    )
    parser.add_argument("--sample_size", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="/home/storage/wenbinxing/Mybenchmark/vcd_new/runs/splits/train_subset_1800_seed2025.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    records, file_sizes = load_train_records(dataset_root)
    selected = sample_video_cover_stratified(records, args.sample_size, args.seed)
    manifest = build_manifest(records, selected, args)
    manifest["source_files"] = file_sizes

    output_manifest = Path(args.output_manifest)
    save_manifest(manifest, output_manifest)

    sample_stats = manifest["sample_stats"]
    print(f"Saved subset manifest to: {output_manifest}")
    print(
        f"Subset size={sample_stats['total_qas']}, "
        f"covered_videos={sample_stats['covered_videos']}, "
        f"video_coverage_ratio={sample_stats['video_coverage_ratio']}"
    )
    print(f"Type ratio abs diff vs full: {sample_stats['type_abs_diff_vs_full']}")


if __name__ == "__main__":
    main()

