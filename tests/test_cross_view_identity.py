import json
import unittest

from visual_peer_inspector.representation.cross_view_identity import (
    INPUT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    CandidateObservation,
    IdentityConfiguration,
    NormalizedGeometry,
    ViewObservation,
    cosine_cost,
    match_cross_view_identities,
    match_view_pair,
    parse_model_facing_input,
)


def candidate(candidate_id, embedding, box=None):
    geometry = None
    if box is not None:
        geometry = NormalizedGeometry(box_xyxy_normalized=box)
    return CandidateObservation(candidate_id, tuple(embedding), geometry)


def view(view_id, *candidates):
    return ViewObservation(view_id, tuple(candidates))


class InputContractTests(unittest.TestCase):
    def valid_payload(self):
        return {
            "schema_version": INPUT_SCHEMA_VERSION,
            "views": [
                {
                    "view_id": "upload-01",
                    "candidates": [
                        {
                            "candidate_id": "P000",
                            "embedding": [1.0, 0.0, 0.0],
                            "geometry": {
                                "box_xyxy_normalized": [0.1, 0.2, 0.3, 0.5],
                                "mask_centroid_xy_normalized": [0.2, 0.35],
                                "mask_area_fraction": 0.04,
                            },
                        }
                    ],
                },
                {
                    "view_id": "upload-03",
                    "candidates": [
                        {
                            "candidate_id": "P100",
                            "embedding": [0.99, 0.01, 0.0],
                        }
                    ],
                },
            ],
        }

    def test_strict_payload_parses_nonconsecutive_view_ids(self):
        views = parse_model_facing_input(self.valid_payload())

        self.assertEqual([item.view_id for item in views], ["upload-01", "upload-03"])
        self.assertEqual(views[0].candidates[0].candidate_id, "P000")
        self.assertEqual(
            views[0].candidates[0].geometry.mask_centroid_xy_normalized,
            (0.2, 0.35),
        )

    def test_forbidden_dataset_and_evaluator_fields_are_rejected(self):
        forbidden_fields = [
            "source_scene_id",
            "source_object_index",
            "source_path",
            "evaluator_id",
            "anomaly_label",
            "anomaly_count",
            "base_asset_id",
            "role",
            "category",
            "camera_metadata",
            "filename",
            "object_order",
        ]
        for field in forbidden_fields:
            with self.subTest(field=field):
                payload = self.valid_payload()
                payload["views"][0]["candidates"][0][field] = "blocked"
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    parse_model_facing_input(payload)

    def test_unknown_metadata_is_rejected(self):
        payload = self.valid_payload()
        payload["views"][0]["candidates"][0]["metadata"] = {}

        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_model_facing_input(payload)

    def test_duplicate_views_and_candidates_are_rejected(self):
        payload = self.valid_payload()
        payload["views"][1]["view_id"] = "upload-01"
        with self.assertRaisesRegex(ValueError, "duplicate view"):
            parse_model_facing_input(payload)

        payload = self.valid_payload()
        payload["views"][0]["candidates"].append(
            dict(payload["views"][0]["candidates"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            parse_model_facing_input(payload)

    def test_invalid_embedding_and_geometry_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            candidate("P0", [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "unit square"):
            candidate("P0", [1.0, 0.0], (-0.1, 0.1, 0.4, 0.4))
        with self.assertRaisesRegex(ValueError, "same dimension"):
            match_cross_view_identities(
                [
                    view("v1", candidate("a", [1.0, 0.0])),
                    view("v2", candidate("b", [1.0, 0.0, 0.0])),
                ]
            )


class PairwiseMatchingTests(unittest.TestCase):
    def test_cosine_cost_uses_global_embedding_direction(self):
        self.assertAlmostEqual(cosine_cost([1, 0], [2, 0]), 0.0)
        self.assertAlmostEqual(cosine_cost([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(cosine_cost([1, 0], [-1, 0]), 2.0)

    def test_hungarian_assignment_is_one_to_one_with_unmatched_option(self):
        left = view(
            "v1",
            candidate("a", [1.0, 0.0]),
            candidate("b", [0.99, 0.01]),
        )
        right = view("v2", candidate("x", [1.0, 0.0]))

        result = match_view_pair(
            left,
            right,
            configuration=IdentityConfiguration(unmatched_cost=0.2),
        )

        matched = [item for item in result.assignments if item.status == "matched"]
        unmatched = [
            item for item in result.assignments if item.status == "unmatched_left"
        ]
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(matched[0].left_candidate_id, "a")
        self.assertEqual(matched[0].right_candidate_id, "x")

    def test_far_candidate_is_explicitly_unmatched(self):
        left = view("v1", candidate("a", [1.0, 0.0]))
        right = view("v2", candidate("x", [0.0, 1.0]))

        result = match_view_pair(
            left,
            right,
            configuration=IdentityConfiguration(unmatched_cost=0.25),
        )

        statuses = [item.status for item in result.assignments]
        self.assertEqual(statuses, ["unmatched_left", "unmatched_right"])

    def test_ties_are_deterministic_and_reported_as_ambiguous(self):
        left = view(
            "v1",
            candidate("a", [1.0, 0.0]),
            candidate("b", [1.0, 0.0]),
        )
        right = view(
            "v2",
            candidate("x", [1.0, 0.0]),
            candidate("y", [1.0, 0.0]),
        )

        first = match_view_pair(left, right)
        second = match_view_pair(
            view("v1", *reversed(left.candidates)),
            view("v2", *reversed(right.candidates)),
        )

        self.assertEqual(first.to_record(), second.to_record())
        links = [
            (item.left_candidate_id, item.right_candidate_id)
            for item in first.assignments
            if item.status == "matched"
        ]
        self.assertEqual(links, [("a", "x"), ("b", "y")])
        self.assertTrue(all(item.ambiguous for item in first.assignments))

    def test_optional_geometry_uses_normalized_upload_measurements(self):
        left = view(
            "v1",
            candidate("a", [1.0, 0.0], (0.05, 0.1, 0.25, 0.4)),
            candidate("b", [1.0, 0.0], (0.75, 0.1, 0.95, 0.4)),
        )
        right = view(
            "v2",
            candidate("x", [1.0, 0.0], (0.74, 0.1, 0.94, 0.4)),
            candidate("y", [1.0, 0.0], (0.06, 0.1, 0.26, 0.4)),
        )

        result = match_view_pair(
            left,
            right,
            configuration=IdentityConfiguration(
                unmatched_cost=0.8,
                geometry_weight=0.5,
            ),
        )

        links = {
            item.left_candidate_id: item.right_candidate_id
            for item in result.assignments
            if item.status == "matched"
        }
        self.assertEqual(links, {"a": "y", "b": "x"})
        self.assertTrue(
            all(
                item.geometry_distance is not None
                for item in result.assignments
                if item.status == "matched"
            )
        )


class CrossViewTrackingTests(unittest.TestCase):
    def stable_views(self, count=5):
        values = []
        for index in range(count):
            values.append(
                view(
                    f"upload-{2 * index + 1:02d}",
                    candidate(f"left-{index}", [1.0, 0.02 * index, 0.0]),
                    candidate(f"right-{index}", [0.0, 0.02 * index, 1.0]),
                )
            )
        return values

    def test_five_view_tracks_are_cycle_consistent(self):
        result = match_cross_view_identities(self.stable_views())

        self.assertEqual(len(result.tracks), 2)
        self.assertEqual({track.status for track in result.tracks}, {"matched"})
        self.assertTrue(all(len(track.members) == 5 for track in result.tracks))
        self.assertTrue(all(len(track.cycle_checks) == 10 for track in result.tracks))
        self.assertTrue(
            all(check.consistent for track in result.tracks for check in track.cycle_checks)
        )

    def test_result_is_invariant_to_view_and_candidate_input_order(self):
        original = self.stable_views(3)
        reversed_input = [
            view(item.view_id, *reversed(item.candidates))
            for item in reversed(original)
        ]

        first = match_cross_view_identities(original).to_record()
        second = match_cross_view_identities(reversed_input).to_record()

        self.assertEqual(first, second)

    def test_missing_and_empty_views_are_explicit_without_forced_links(self):
        result = match_cross_view_identities(
            [
                view("upload-01", candidate("a", [1.0, 0.0])),
                view("upload-05"),
                view("upload-09", candidate("b", [1.0, 0.01])),
            ]
        )

        self.assertEqual(result.view_ids, ("upload-01", "upload-05", "upload-09"))
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(len(result.tracks[0].members), 2)
        empty_pairs = [
            assignment.status
            for pair in result.pairwise_results
            if "upload-05" in (pair.left_view_id, pair.right_view_id)
            for assignment in pair.assignments
        ]
        self.assertTrue(empty_pairs)
        self.assertTrue(all(status.startswith("unmatched") for status in empty_pairs))

    def test_ambiguous_pair_marks_the_identity_track(self):
        result = match_cross_view_identities(
            [
                view(
                    "v1",
                    candidate("a", [1.0, 0.0]),
                    candidate("b", [1.0, 0.0]),
                ),
                view(
                    "v2",
                    candidate("x", [1.0, 0.0]),
                    candidate("y", [1.0, 0.0]),
                ),
            ]
        )

        self.assertEqual({track.status for track in result.tracks}, {"ambiguous"})
        self.assertTrue(all(track.minimum_margin == 0.0 for track in result.tracks))

    def test_inconsistent_three_view_cycle_is_marked_ambiguous(self):
        angle_vectors = {
            0: (1.0, 0.0),
            60: (0.5, 0.8660254),
            120: (-0.5, 0.8660254),
            180: (-1.0, 0.0),
            240: (-0.5, -0.8660254),
            300: (0.5, -0.8660254),
        }
        result = match_cross_view_identities(
            [
                view(
                    "v1",
                    candidate("a0", angle_vectors[0]),
                    candidate("a1", angle_vectors[180]),
                ),
                view(
                    "v2",
                    candidate("b0", angle_vectors[60]),
                    candidate("b1", angle_vectors[240]),
                ),
                view(
                    "v3",
                    candidate("c0", angle_vectors[120]),
                    candidate("c1", angle_vectors[300]),
                ),
            ],
            configuration=IdentityConfiguration(unmatched_cost=1.0),
        )

        self.assertEqual({track.status for track in result.tracks}, {"ambiguous"})
        self.assertTrue(all(not track.cycle_consistent for track in result.tracks))
        self.assertTrue(
            all(
                not check.consistent
                for track in result.tracks
                for check in track.cycle_checks
            )
        )

    def test_versioned_result_is_json_serializable(self):
        result = match_cross_view_identities(self.stable_views(2))
        record = result.to_record()

        self.assertEqual(record["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(record["summary"]["view_count"], 2)
        self.assertEqual(len(record["configuration_sha256"]), 64)
        self.assertEqual(len(record["input_sha256"]), 64)
        json.dumps(record, allow_nan=False)

    def test_only_two_to_five_views_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "2 to 5"):
            match_cross_view_identities(self.stable_views(1))
        with self.assertRaisesRegex(ValueError, "2 to 5"):
            match_cross_view_identities(self.stable_views(6))


if __name__ == "__main__":
    unittest.main()
