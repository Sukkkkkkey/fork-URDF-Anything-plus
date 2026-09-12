#!/usr/bin/env python3
"""Compare two local metric summaries with URDF-Anything+ paper Table 1."""

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


def geometry_metric_mean(
    summary: dict[str, Any],
    scope: str,
    metric: str,
    part_variant: str = "matched_only",
) -> float | None:
    if scope == "parts":
        values = summary["geometry"]["parts"][part_variant]
    elif scope == "whole":
        values = summary["geometry"]["whole"]
    else:
        raise ValueError(f"unknown geometry scope: {scope}")
    if metric in values:
        return values[metric]["mean"]

    # Read legacy single-IoU summaries when building historical reports.
    if metric in {"iou_surface", "iou_volume"} and "iou" in values:
        solid_fill = summary.get("metadata", {}).get("voxel_iou", {}).get(
            "solid_fill"
        )
        expected_solid_fill = metric == "iou_volume"
        if solid_fill is expected_solid_fill:
            return values["iou"]["mean"]
    return None


def summary_values(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "parts_iou": geometry_metric_mean(summary, "parts", "iou_volume"),
        "parts_fscore": summary["geometry"]["parts"]["matched_only"]["fscore"]["mean"],
        "parts_cd": summary["geometry"]["parts"]["matched_only"]["cd"]["mean"],
        "whole_iou": geometry_metric_mean(summary, "whole", "iou_volume"),
        "whole_fscore": summary["geometry"]["whole"]["fscore"]["mean"],
        "whole_cd": summary["geometry"]["whole"]["cd"]["mean"],
        "axis_error_rad": summary["joints"]["axis_error_rad"]["mean"],
        # The legacy evaluator reports normalized object coordinates, not metres.
        "origin_error_m": summary["joints"]["origin_error"]["mean"],
        "limit_error_rad": summary["joints"]["limit_error_rad"]["mean"],
    }


def iou_variant_rows(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for scope in ("parts", "whole"):
        for variant in ("iou_surface", "iou_volume"):
            rows.append(
                {
                    "metric": f"{scope}_{variant}",
                    "baseline_local": geometry_metric_mean(
                        baseline, scope, variant
                    ),
                    "candidate_local": geometry_metric_mean(
                        candidate, scope, variant
                    ),
                }
            )
    return rows


def comparison_rows(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    candidate_values = summary_values(candidate)
    baseline_values = summary_values(baseline)
    rows = []
    for metric, paper_value in PAPER_TABLE1_URDF.items():
        if candidate_values[metric] is None:
            raise KeyError(f"candidate summary has no value for {metric}")
        candidate_value = float(candidate_values[metric])
        baseline_value = (
            None
            if baseline_values[metric] is None
            else float(baseline_values[metric])
        )
        direction = DIRECTIONS[metric]
        exact_comparison = metric not in {"parts_iou", "whole_iou", "origin_error_m"}
        if metric in {"parts_iou", "whole_iou"}:
            comparability = "protocol-dependent: local 128^3 solid voxels; paper voxelizer unpublished"
        elif metric == "origin_error_m":
            comparability = "no: local normalized coordinates vs paper metres"
        else:
            comparability = "yes: same published formula; hidden protocol choices remain"
        improvement = None
        if baseline_value is not None:
            improvement = (
                candidate_value - baseline_value
                if direction == "higher"
                else baseline_value - candidate_value
            )
        rows.append(
            {
                "metric": metric,
                "better": direction,
                "paper_table1": paper_value,
                "baseline_local": baseline_value,
                "candidate_local": candidate_value,
                "improvement_vs_baseline": improvement,
                "comparability": comparability,
                "candidate_minus_paper": (
                    candidate_value - paper_value if exact_comparison else None
                ),
                "candidate_over_paper": (
                    candidate_value / paper_value if exact_comparison else None
                ),
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
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    title: str,
) -> str:
    header = (
        f"| Metric | Paper Table 1 | {baseline_label} | {candidate_label} | "
        f"{candidate_label} improvement | Comparability |\n"
        "|---|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {metric} | {paper} | {old} | {new} | {improvement} | {note} |".format(
                metric=row["metric"],
                paper=format_number(row["paper_table1"]),
                old=format_number(row["baseline_local"]),
                new=format_number(row["candidate_local"]),
                improvement=format_number(row["improvement_vs_baseline"]),
                note=row["comparability"],
            )
        )
    iou_rows = iou_variant_rows(candidate, baseline)
    iou_lines = [
        f"| IoU representation | {baseline_label} | {candidate_label} |",
        "|---|---:|---:|",
    ]
    for row in iou_rows:
        iou_lines.append(
            f"| {row['metric']} | {format_number(row['baseline_local'])} | "
            f"{format_number(row['candidate_local'])} |"
        )
    counts = candidate["counts"]
    joints = candidate["joints"]
    baseline_counts = baseline["counts"]
    baseline_joints = baseline["joints"]
    metadata = candidate.get("metadata", {})
    protocol_version = metadata.get("protocol_version", "unknown")
    stress_lines = [
        "| Part completeness diagnostic | "
        f"{baseline_label} | {candidate_label} |",
        "|---|---:|---:|",
    ]
    for metric in ("iou_surface", "iou_volume", "fscore", "cd"):
        stress_lines.append(
            f"| Penalized {metric} | "
            f"{format_number(geometry_metric_mean(baseline, 'parts', metric, 'penalized'))} | "
            f"{format_number(geometry_metric_mean(candidate, 'parts', metric, 'penalized'))} |"
        )
    method_caveat = []
    if candidate.get("method") != baseline.get("method"):
        method_caveat = [
            "- Cross-method geometry is input-sensitive: Particulate preserves and "
            "segments the provided whole mesh, whereas URDF-Anything+ decodes new "
            "per-link geometry while conditioned on that GT mesh.",
        ]
    return "\n".join(
        [
            f"# {title}",
            "",
            "Both local summaries use `evaluate_author_metrics.py`; the generated "
            "URDF fixed root transform is applied to URDF-Anything+ before evaluation "
            "(`--urdf-coordinate world`). Metric formulas, 100k surface "
            "samples, one shared 128-cubed rasterization for surface and volume IoU, "
            "CD-Hungarian matching, and object aggregation are unchanged. "
            f"Protocol version: `{protocol_version}`.",
            "",
            *lines,
            "",
            "Explicit IoU variants (the paper row above uses volume IoU):",
            "",
            *iou_lines,
            "",
            "Coverage and caveats:",
            "",
            f"- {candidate_label}: {candidate['object_count']} objects; "
            "predicted/GT/matched parts: "
            f"{counts['pred_parts']}/{counts['gt_parts']}/{counts['matched_parts']}.",
            f"- {candidate_label} conditional revolute coverage: "
            f"{joints['correct_type_count']}/"
            f"{joints['gt_revolute_count']} = {joints['coverage']:.6f}; the paper "
            "does not publish the corresponding denominator.",
            f"- {baseline_label}: {baseline['object_count']} objects; "
            "predicted/GT/matched parts: "
            f"{baseline_counts['pred_parts']}/{baseline_counts['gt_parts']}/"
            f"{baseline_counts['matched_parts']}.",
            f"- {baseline_label} conditional revolute coverage: "
            f"{baseline_joints['correct_type_count']}/"
            f"{baseline_joints['gt_revolute_count']} = "
            f"{baseline_joints['coverage']:.6f}.",
            "- Part rows are matched-only diagnostics; paper handling of missing or "
            "extra parts is unpublished.",
            *method_caveat,
            "- Both surface and volume IoU are emitted by default; paper comparison "
            "uses volume IoU, whose exact voxelizer and fill rule remain unpublished.",
            "- The local origin value is normalized axis-line distance, while the "
            "paper reports metres, so those two numbers must not be ratio-compared.",
            "",
            "Completeness stress test (unmatched parts use IoU/F=0 and CD=12; "
            "this is not a paper metric):",
            "",
            *stress_lines,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-summary", "--oriented-summary", dest="candidate_summary",
        type=Path, required=True
    )
    parser.add_argument(
        "--baseline-summary", "--unoriented-summary", dest="baseline_summary",
        type=Path, required=True
    )
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument(
        "--title", default="URDF-Anything+ local metrics vs paper Table 1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate_summary)
    baseline = load_json(args.baseline_summary)
    rows = comparison_rows(candidate, baseline)
    iou_rows = iou_variant_rows(candidate, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paper_comparison.csv", rows)
    write_csv(args.output_dir / "iou_comparison.csv", iou_rows)
    payload = {
        "paper_reference": {
            "url": "https://arxiv.org/abs/2603.14010",
            "table": "Table 1, URDF-Anything+ row",
            "values": PAPER_TABLE1_URDF,
        },
        "candidate_summary": str(args.candidate_summary),
        "baseline_summary": str(args.baseline_summary),
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "candidate_counts": candidate["counts"],
        "candidate_joints": candidate["joints"],
        "baseline_counts": baseline["counts"],
        "baseline_joints": baseline["joints"],
        "rows": rows,
        "iou_variants": iou_rows,
    }
    (args.output_dir / "paper_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        build_report(
            candidate,
            baseline,
            rows,
            args.baseline_label,
            args.candidate_label,
            args.title,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
