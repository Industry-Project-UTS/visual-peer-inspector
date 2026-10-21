"""Caption regression checks without private media or the application."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import build
import narration


HERE = Path(__file__).resolve().parent


class CaptionTests(unittest.TestCase):
    def test_final_edit_matches_measured_speech_instead_of_planned_starts(self):
        cues = narration.read_timeline(HERE / "data/final_edit_captions.json")
        self.assertEqual(len(cues), 29)
        self.assertEqual(cues[0]["start"], 1.14)
        study = next(c for c in cues if c["segment"] == "V03")
        self.assertEqual(study["start"], 161.263)
        self.assertEqual(cues[-1]["end"], 229.314)
        # The final edit inserted a pause inside this cue; it is not a rigid shift.
        self.assertAlmostEqual(cues[1]["end"] - cues[1]["start"], 3.294)

    def test_build_captions_follow_a_shorter_middle_section(self):
        order = ["V01.mp4", "V02.mp4", "S01.mp4", "S02.mp4", "S03.mp4",
                 "S04.mp4", "S05.mp4", "V03.mp4", "V04.mp4", "V05.mp4", "C01.mp4"]
        lengths = dict(zip(order, [25, 25, 15, 15, 30, 20, 20, 40, 25, 15, 10]))
        with patch.object(build, "duration", side_effect=lengths.__getitem__):
            timeline = build.caption_timeline(order, 240)
        cues = narration.cues_for_segment_starts(timeline["segment_starts"])
        narration.validate_cues(cues, timeline["duration_seconds"])
        study = next(c for c in cues if c["segment"] == "V03")
        self.assertAlmostEqual(study["start"], 151.93)
        self.assertAlmostEqual(cues[-1]["end"], 226.57)

    def test_explicit_build_timeline_overrides_default_final_edit(self):
        timeline = {"schema_version": 1, "duration_seconds": 280,
                    "segment_starts": {"V01": 0, "V02": 25, "V03": 190,
                                       "V04": 230, "V05": 255}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            path.write_text(json.dumps(timeline), encoding="utf-8")
            result = subprocess.run([sys.executable, str(HERE / "narration.py"), tmp,
                                     "--timeline", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (Path(tmp) / "Vaibhav_narration_captions.srt").read_text(encoding="utf-8")
            self.assertIn("00:03:11,930 --> 00:03:15,670", text)

    def test_caption_export_needs_no_voice_or_video_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(HERE / "narration.py"), tmp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = (Path(tmp) / "Vaibhav_narration_captions.srt").read_text(encoding="utf-8")
            committed = (HERE / "captions/Vaibhav_narration_captions.srt").read_text(encoding="utf-8")
            self.assertEqual(generated, committed)

    def test_invalid_intervals_and_nonfinite_times_are_rejected(self):
        for start, end in [(-1, 2), (2, 1), (1, 241), (float("nan"), 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                narration.validate_cues([{"start": start, "end": end, "text": "Speech"}], 240)
        with self.assertRaises(ValueError):
            narration.validate_cues([{"start": 1, "end": 3, "text": "First"},
                                     {"start": 2, "end": 4, "text": "Second"}], 240)
        self.assertEqual(narration.srt_time(59.9996), "00:01:00,000")


if __name__ == "__main__":
    unittest.main()
