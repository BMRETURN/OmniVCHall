import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vcd_ste.core import (  # noqa: E402
    CACHE_STATS_KEYS,
    apply_limit,
    collect_summary_from_logs,
    load_qa_data,
    qa_uid,
    save_json,
    subset_by_type,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Merge sharded eval_ste logs into canonical metrics.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "vcd_ste" / "runs"),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "MyBench"),
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--smoke_per_type", type=int, default=0)
    parser.add_argument(
        "--shard_suffixes",
        type=str,
        required=True,
        help="Comma-separated suffix tags, e.g. shard0,shard1,shard2,shard3",
    )
    parser.add_argument(
        "--include_base_logs",
        type=str,
        default="true",
        help="Whether to include canonical split logs as base source (true/false).",
    )
    return parser


def str2bool(v):
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def load_json_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list json at {path}, got {type(data)}")
    return data


def merge_row(prev: dict, curr: dict) -> Tuple[dict, bool]:
    prev_ok = prev.get("status", "ok") == "ok"
    curr_ok = curr.get("status", "ok") == "ok"
    if prev_ok and not curr_ok:
        return prev, False
    if (not prev_ok) and curr_ok:
        return curr, True
    return curr, True


def main():
    args = build_parser().parse_args()
    include_base_logs = str2bool(args.include_base_logs)

    run_dir = Path(args.output_root) / args.run_id
    split = args.split
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    suffixes = [s.strip() for s in str(args.shard_suffixes).split(",") if s.strip()]
    if len(suffixes) == 0:
        raise ValueError("No shard_suffixes provided")

    data = load_qa_data(str(Path(args.dataset_root) / split), shuffle=False)
    data = subset_by_type(data, args.smoke_per_type)
    data = apply_limit(data, args.limit)
    ordered_uids = [uid for uid in (qa_uid(item) for item in data) if uid]

    sources: List[Tuple[str, Path]] = []
    if include_base_logs:
        sources.append(("base", run_dir / f"{split}_logs.json"))
    for suffix in suffixes:
        sources.append((suffix, run_dir / f"{split}_logs_{suffix}.json"))

    metric_sources = []
    if include_base_logs:
        metric_sources.append(run_dir / f"{split}_metrics.json")
    for suffix in suffixes:
        metric_sources.append(run_dir / f"{split}_metrics_{suffix}.json")

    merged_by_uid: Dict[str, dict] = {}
    unkeyed_rows: List[dict] = []
    seen_rows = 0
    replaced_rows = 0
    loaded_sources = []

    for name, path in sources:
        rows = load_json_list(path)
        if len(rows) == 0:
            continue
        loaded_sources.append(str(path))
        for row in rows:
            seen_rows += 1
            uid = row.get("qa_uid")
            if not uid:
                unkeyed_rows.append(row)
                continue
            prev = merged_by_uid.get(uid)
            if prev is None:
                merged_by_uid[uid] = row
                continue
            merged, replaced = merge_row(prev, row)
            merged_by_uid[uid] = merged
            if replaced:
                replaced_rows += 1

    ordered_logs = []
    seen_uids = set()
    for uid in ordered_uids:
        row = merged_by_uid.get(uid)
        if row is None:
            continue
        ordered_logs.append(row)
        seen_uids.add(uid)

    for uid, row in merged_by_uid.items():
        if uid not in seen_uids:
            ordered_logs.append(row)
    ordered_logs.extend(unkeyed_rows)

    summary = collect_summary_from_logs(ordered_logs)
    summary["num_samples"] = len(ordered_logs)
    summary["run_id"] = args.run_id
    summary["split"] = split
    summary["progress"] = {"done": len(ordered_logs), "total": len(ordered_uids)}
    summary["merge"] = {
        "sources": loaded_sources,
        "raw_rows_seen": int(seen_rows),
        "unique_uid_rows": int(len(merged_by_uid)),
        "unkeyed_rows": int(len(unkeyed_rows)),
        "rows_replaced": int(replaced_rows),
    }

    cache_stats = {k: 0 for k in CACHE_STATS_KEYS}
    for metric_path in metric_sources:
        if not metric_path.exists():
            continue
        with open(metric_path, "r", encoding="utf-8") as f:
            metric_data = json.load(f)
        source_cache_stats = metric_data.get("cache_stats") or {}
        for k in CACHE_STATS_KEYS:
            cache_stats[k] += int(source_cache_stats.get(k, 0))
    summary["cache_stats"] = cache_stats

    out_logs = run_dir / f"{split}_logs.json"
    out_metrics = run_dir / f"{split}_metrics.json"
    save_json(ordered_logs, out_logs)
    save_json(summary, out_metrics)

    print(f"Merged logs saved: {out_logs}")
    print(f"Merged metrics saved: {out_metrics}")
    print(
        f"done={summary['progress']['done']}/{summary['progress']['total']}, "
        f"micro_acc={summary['micro_overall_acc']:.4f}, "
        f"macro_acc={summary['macro_average_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
