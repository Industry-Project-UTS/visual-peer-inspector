import unittest

from visual_peer_inspector.data.sam_prompt_package import (
    _assert_model_safe,
    _geometric_prompts,
)


class SamPromptPackageTests(unittest.TestCase):
    def test_prompt_ids_follow_geometry_not_source_order(self):
        boxes = [[40, 20, 60, 50], [5, 5, 15, 15], [20, 10, 30, 30]]

        first = _geometric_prompts(boxes, "R")
        second = _geometric_prompts(list(reversed(boxes)), "R")

        self.assertEqual(first, second)
        self.assertEqual([row["prompt_key"] for row in first], ["R000", "R001", "R002"])

    def test_duplicate_prompts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Duplicate boxes"):
            _geometric_prompts([[0, 0, 10, 10], [0, 0, 10, 10]], "P")

    def test_protected_fields_are_rejected_recursively(self):
        with self.assertRaisesRegex(ValueError, "Forbidden model-facing field"):
            _assert_model_safe({"routes": [{"category": "toy"}]})

    def test_prompt_manifest_fields_are_allowed(self):
        _assert_model_safe(
            {
                "opaque_scene_id": "S1",
                "view_id": "V00",
                "routes": {
                    "reference_box": [
                        {"prompt_key": "R000", "box_xyxy": [0, 0, 10, 10]}
                    ]
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
