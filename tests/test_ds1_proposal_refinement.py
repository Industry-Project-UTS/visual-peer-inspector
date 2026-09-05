from __future__ import annotations

import unittest

from visual_peer_inspector.workflows.proposal_refinement_ds1 import (
    POLICIES,
    ProposalPolicy,
    apply_proposal_policy,
)


class DS1ProposalRefinementTests(unittest.TestCase):
    def test_policy_registry_is_fixed_and_contains_the_baseline(self) -> None:
        self.assertEqual(
            [policy.key for policy in POLICIES],
            [
                "baseline",
                "area_only",
                "nms_090",
                "nms_075",
                "area_nms_075",
            ],
        )

    def test_area_and_overlap_filter_preserves_original_order(self) -> None:
        policy = ProposalPolicy("test", 0.0015, 0.70, 0.75)
        boxes = [
            [10, 10, 30, 30],
            [11, 11, 30, 30],
            [100, 100, 102, 102],
            [40, 40, 70, 70],
        ]

        kept, indices, removed = apply_proposal_policy(boxes, policy)

        self.assertEqual(kept, [[10, 10, 30, 30], [40, 40, 70, 70]])
        self.assertEqual(indices, [0, 3])
        self.assertEqual(removed["overlap_suppression"], 1)
        self.assertEqual(removed["below_area_floor"], 1)

    def test_invalid_box_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid box"):
            apply_proposal_policy([[20, 20, 10, 30]], POLICIES[0])

    def test_proposal_cap_keeps_the_largest_boxes_in_model_order(self) -> None:
        policy = ProposalPolicy("cap", 0.0, 1.0, None, 2)
        boxes = [
            [0, 0, 10, 10],
            [0, 0, 30, 30],
            [50, 50, 70, 70],
            [100, 100, 105, 105],
        ]

        kept, indices, removed = apply_proposal_policy(boxes, policy)

        self.assertEqual(kept, [[0, 0, 30, 30], [50, 50, 70, 70]])
        self.assertEqual(indices, [1, 2])
        self.assertEqual(removed["proposal_cap"], 2)


if __name__ == "__main__":
    unittest.main()
