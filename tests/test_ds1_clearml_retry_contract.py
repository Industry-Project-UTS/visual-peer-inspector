from __future__ import annotations

from pathlib import Path
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
FINALIZERS = (
    "finalize_ds1_opaque_package_clearml.py",
    "finalize_locateanything_ds1_clearml.py",
    "finalize_ds1_sam_clearml.py",
    "finalize_ds1_mask_evaluation_clearml.py",
)


class Ds1ClearmlRetryContractTests(unittest.TestCase):
    def test_every_ds1_finalizer_uses_the_shared_bounded_retry_contract(self) -> None:
        for name in FINALIZERS:
            source = (CODE_ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(finalizer=name):
                self.assertIn("configure_official_clearml_hosts", source)
                self.assertIn("initialize_clearml_task_with_retry", source)
                self.assertIn("sanitized_tracking_failure_fields", source)
                self.assertIn("signal.alarm", source)
                self.assertNotIn('receipt["error_message"]', source)
                self.assertNotIn("os.environ.setdefault(\"CLEARML_API_HOST\"", source)

    def test_every_ds1_finalizer_records_safe_retry_diagnostics(self) -> None:
        required = (
            "clearml_hosts",
            "tracking_attempt_count",
            "tracking_retry_count",
            "tracking_transient_failure_count",
        )
        for name in FINALIZERS:
            source = (CODE_ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(finalizer=name):
                for field in required:
                    self.assertIn(f'"{field}"', source)


if __name__ == "__main__":
    unittest.main()
