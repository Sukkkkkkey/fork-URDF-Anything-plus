#!/usr/bin/env python3
"""Compare oriented first-version local metrics with paper Table 1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PAPER_TABLE1_URDF = {
    "parts_iou": 0.879,
    "parts_fscore": 0.721,
    "parts_cd": 0.033,
    "whole_iou": 0.930,
    "whole_fscore": 0.742,
    "whole_cd": 0.009,
    "axis_error_rad": 0.129,
    "origin_error_m": 0.062,
    "limit_error_rad": 0.225,
}


DIRECTIONS = {
    "parts_iou": "higher",
    "parts_fscore": "higher",
    "parts_cd": "lower",
    "whole_iou": "higher",
    "whole_fscore": "higher",
    "whole_cd": "lower",
    "axis_error_rad": "lower",
    "origin_error_m": "lower",
    "limit_error_rad": "lower",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def summary_values(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "parts_iou": summary["geometry"]["parts"]["matched_only"]["iou"]["mean"],
        "parts_fscore": summary["geometry"]["parts"]["matched_only"]["fscore"]["mean"],
        "parts_cd": summary["geometry"]["parts"]["matched_only"]["cd"]["mean"],
        "whole_iou": summary["geometry"]["whole"]["iou"]["mean"],
        "whole_fscore": summary["geometry"]["whole"]["fscore"]["mean"],
        "whole_cd": summary["geometry"]["whole"]["cd"]["mean"],
        "axis_error_rad": summary["joints"]["axis_error_rad"]["mean"],
        # The legacy evaluator reports normalized object coordinates, not metres.
        "origin_error_m": summary["joints"]["origin_error"]["mean"],
        "limit_error_rad": summary["joints"]["limit_error_rad"]["mean"],
    }


def comparison_rows(
    oriented: dict[str, Any], unoriented: dict[str, Any]
) -> list[dict[str, Any]]:
    oriented_values = summary_values(oriented)
    unoriented_values = summary_values(unoriented)
    rows = []
    for metric, paper_value in PAPER_TABLE1_URDF.items():
        new_value = float(oriented_values[metric])
        old_value = float(unoriented_values[metric])
        direction = DIRECTIONS[metric]
        exact_comparison = metric not in {"parts_iou", "whole_iou", "origin_error_m"}
        if metric in {"parts_iou", "whole_iou"}:
            comparability = "proxy_only: local surface voxels vs paper volume/occupancy"
        elif metric == "origin_error_m":
            comparability = "no: local normalized coordinates vs paper metres"
        else:
            comparability = "yes: same published formula; hidden protocol choices remain"
        improvement = new_value - old_value if direction == "higher" else old_value - new_value
        rows.append(
            {
                "metric": metric,
                "better": direction,
                "paper_table1": paper_value,
                "unoriented_local": old_value,
                "oriented_local": new_value,
                "improvement_vs_unoriented": improvement,
                "comparability": comparability,
                "oriented_minus_paper": new_value - paper_value if exact_comparison else None,
                "oriented_over_paper": new_value / paper_value if exact_comparison else None,
            }
        )
    return rows


def format_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    oriented: dict[str, Any], unoriented: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    header = (
        "| Metric | Paper Table 1 | Old orientation | Corrected orientation | "
        "Improvement | Comparability |\n"
        "|---|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {metric} | {paper} | {old} | {new} | {improvement} | {note} |".format(
                metric=row["metric"],
                paper=format_number(row["paper_table1"]),
                old=format_number(row["unoriented_local"]),
                new=format_number(row["oriented_local"]),
                improvement=format_number(row["improvement_vs_unoriented"]),
                note=row["comparability"],
            )
        )
    counts = oriented["counts"]
    joints = oriented["joints"]
    return "\n".join(
        [
            "# Corrected-orientation URDF-Anything+ vs paper Table 1",
            "",
            "The local run uses the first-version `evaluate_author_metrics.py` "
            "protocol with the generated URDF fixed root transform applied before "
            "evaluation (`--urdf-coordinate world`). Metric formulas, 100k surface "
            "samples, 128-cubed surface voxels, CD-Hungarian matching, and object "
            "aggregation are unchanged.",
            "",
            *lines,
            "",
            "Coverage and caveats:",
            "",
            f"- Objects: {oriented['object_count']}; predicted/GT/matched parts: "
            f"{counts['pred_parts']}/{counts['gt_parts']}/{counts['matched_parts']}.",
            f"- Conditional revolute coverage: {joints['correct_type_count']}/"
            f"{joints['gt_revolute_count']} = {joints['coverage']:.6f}; the paper "
            "does not publish the corresponding denominator.",
            "- Part rows are matched-only diagnostics; paper handling of missing or "
            "extra parts is unpublished.",
            "- IoU is not an exact reproduction because this evaluator measures "
            "surface-voxel overlap rather than a uniquely defined solid volume.",
            "- The local origin value is normalized axis-line distance, while the "
            "paper reports metres, so those two numbers must not be ratio-compared.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oriented-summary", type=Path, required=True)
    parser.add_argument("--unoriented-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    oriented = load_json(args.oriented_summary)
    unoriented = load_json(args.unoriented_summary)
    rows = comparison_rows(oriented, unoriented)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paper_comparison.csv", rows)
    payload = {
        "paper_reference": {
            "url": "https://arxiv.org/abs/2603.14010",
            "table": "Table 1, URDF-Anything+ row",
            "values": PAPER_TABLE1_URDF,
        },
        "oriented_summary": str(args.oriented_summary),
        "unoriented_summary": str(args.unoriented_summary),
        "oriented_counts": oriented["counts"],
        "oriented_joints": oriented["joints"],
        "rows": rows,
    }
    (args.output_dir / "paper_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        build_report(oriented, unoriented, rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
