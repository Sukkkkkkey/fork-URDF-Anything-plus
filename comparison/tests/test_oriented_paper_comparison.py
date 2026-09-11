import unittest

from comparison.build_oriented_paper_comparison import comparison_rows


def metric(mean):
    return {"mean": mean, "std": 0.0, "count": 1}


def summary(value):
    return {
        "geometry": {
            "parts": {
                "matched_only": {
                    "iou": metric(value),
                    "fscore": metric(value),
                    "cd": metric(value),
                }
            },
            "whole": {
                "iou": metric(value),
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


class OrientedPaperComparisonTests(unittest.TestCase):
    def test_improvement_respects_metric_direction(self):
        rows = {row["metric"]: row for row in comparison_rows(summary(0.2), summary(0.4))}
        self.assertAlmostEqual(rows["parts_fscore"]["improvement_vs_unoriented"], -0.2)
        self.assertAlmostEqual(rows["parts_cd"]["improvement_vs_unoriented"], 0.2)

    def test_origin_and_iou_are_not_exactly_compared(self):
        rows = {row["metric"]: row for row in comparison_rows(summary(0.2), summary(0.4))}
        self.assertIsNone(rows["origin_error_m"]["oriented_minus_paper"])
        self.assertIsNone(rows["parts_iou"]["oriented_over_paper"])


if __name__ == "__main__":
    unittest.main()
