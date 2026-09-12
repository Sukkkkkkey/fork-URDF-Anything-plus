import unittest

from comparison.build_metric_comparison import (
    comparison_rows,
    iou_variant_rows,
)


def metric(mean):
    return {"mean": mean, "std": 0.0, "count": 1}


def summary(value):
    return {
        "geometry": {
            "parts": {
                "matched_only": {
                    "iou_surface": metric(value + 0.1),
                    "iou_volume": metric(value),
                    "fscore": metric(value),
                    "cd": metric(value),
                },
                "penalized": {
                    "iou_surface": metric(value + 0.1),
                    "iou_volume": metric(value),
                    "fscore": metric(value),
                    "cd": metric(value),
                },
            },
            "whole": {
                "iou_surface": metric(value + 0.1),
                "iou_volume": metric(value),
                "fscore": metric(value),
                "cd": metric(value),
            },
        },
        "joints": {
            "axis_error_rad": metric(value),
            "origin_error": metric(value),
            "limit_error_rad": metric(value),
        },
    }


class MetricComparisonTests(unittest.TestCase):
    def test_improvement_respects_metric_direction(self):
        rows = {row["metric"]: row for row in comparison_rows(summary(0.2), summary(0.4))}
        self.assertAlmostEqual(rows["parts_fscore"]["improvement_vs_baseline"], -0.2)
        self.assertAlmostEqual(rows["parts_cd"]["improvement_vs_baseline"], 0.2)

    def test_origin_and_iou_are_not_exactly_compared(self):
        rows = {row["metric"]: row for row in comparison_rows(summary(0.2), summary(0.4))}
        self.assertIsNone(rows["origin_error_m"]["candidate_minus_paper"])
        self.assertIsNone(rows["parts_iou"]["candidate_over_paper"])
        self.assertIn("solid voxels", rows["parts_iou"]["comparability"])

    def test_paper_uses_volume_but_report_keeps_both_iou_variants(self):
        candidate = summary(0.2)
        baseline = summary(0.4)
        rows = {row["metric"]: row for row in comparison_rows(candidate, baseline)}
        self.assertAlmostEqual(rows["parts_iou"]["candidate_local"], 0.2)
        iou_rows = {
            row["metric"]: row for row in iou_variant_rows(candidate, baseline)
        }
        self.assertAlmostEqual(iou_rows["parts_iou_surface"]["candidate_local"], 0.3)
        self.assertAlmostEqual(iou_rows["parts_iou_volume"]["candidate_local"], 0.2)


if __name__ == "__main__":
    unittest.main()
