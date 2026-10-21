"""
Narration script and timings for Vaibhav's sections.

SEGMENTS holds the script text from the recording guide. ACTUAL holds the sentence start/end
times measured from each recording in voice/ (pause detection plus forced alignment, see
align.py); the animations in render.py are cued from these times. CUES holds the caption
lines. The default caption export uses the measured, published final edit in
data/final_edit_captions.json. SEGMENTS retains the original rendering schedule;
it is not the timeline of the final edited video.

    python narration.py            # writes out/Vaibhav_narration_captions.srt
    python narration.py --timeline out/build-timeline.json
"""

import argparse
import json
import math
from pathlib import Path
import textwrap

WPS = 2.35      # estimated speaking rate, words per second (~141 wpm)
GAP = 0.35      # pause after each sentence
LEAD = 0.5      # silence before first sentence

SEGMENTS = [
    dict(id="V01", title="Problem and project", start=0, dur=25, sentences=[
        "One of these bottles is missing its cap.",
        "From another angle, the same bottle appears in a different position.",
        "Our project, Visual Peer Inspector, explores how to identify an unusual item among similar products and show the evidence across several photographs.",
        "The intended setting is small-batch inspection.",
    ]),
    dict(id="V02", title="Technology and handover", start=25, dur=25, sentences=[
        "The system uses the other objects in the scene as its reference.",
        "Locate Anything finds candidate objects,",
        "Sam separates their foreground,",
        "and Dino version two extracts visual features.",
        "Our pipeline compares the objects and combines evidence across views.",
        "Sreekar will now show how this works in the web app.",
    ]),
    dict(id="V03", title="Main measured outcome", start=180, dur=40, sentences=[
        "Our main study used two thousand selected Toys A D eight K scenes: one thousand for development, six hundred for calibration and four hundred for internal evaluation.",
        "With five views, assisted top-one accuracy was sixty-one point seven two percent, compared with twenty-two percent for Automatic.",
        "These are tie-aware averages across predefined view combinations.",
        "The assisted benchmark used dataset-supplied object matches, rather than a human-user study.",
        "The routes also use different scoring rules, so matching alone does not explain the gap.",
    ]),
    dict(id="V04", title="Limits and potential value", start=220, dur=25, sentences=[
        "Automatic matching remains unreliable, and supplementary dataset results were lower.",
        "Historical exposure to related dataset objects also limits the evaluation.",
        "We have not established false-alarm rates for all-normal batches or measured operator time savings.",
        "The potential value is a clearer inspection review process, but the current prototype cannot certify that a batch is defect-free.",
    ]),
    dict(id="V05", title="Next steps and conclusion", start=245, dur=15, sentences=[
        "Next, we would improve matching and independently test normal and unusual groups.",
        "We delivered a working prototype with measured limits, turning photographs into evidence an operator can review.",
    ]),
]

# Caption text (how words should read on screen, vs. how the voice should say them)
CAPTION_FIX = {
    "Toys A D eight K": "ToysAD-8K",
    "two thousand": "2,000", "one thousand": "1,000", "six hundred": "600", "four hundred": "400",
    "sixty-one point seven two percent": "61.72%", "twenty-two percent": "22%",
    "top-one": "top-1", "Locate Anything": "LocateAnything", "Sam separates": "SAM separates",
    "Dino version two": "DINOv2",
}


# Measured from recorded voice files (seconds from the start of the file, file placed at clip start)
ACTUAL = {
    "V01": [(1.14, 3.56), (3.99, 7.18), (7.74, 17.20), (17.74, 20.32)],
    "V02": [(1.35, 4.89), (5.28, 7.42), (7.84, 9.42), (9.87, 12.26), (12.64, 16.38), (17.03, 19.90)],
    "V03": [(1.93, 11.32), (11.80, 19.88), (20.30, 24.25), (24.85, 29.97), (30.63, 34.65)],
    "V04": [(0.68, 4.87), (5.10, 8.94), (9.50, 15.01), (15.42, 22.15)],
    "V05": [(1.23, 5.45), (5.82, 11.57)],
}
# Caption cues measured from the recordings (overrides the automatic sentence split)
CUES = {
    "V01": [(1.14, 3.56, "One of these bottles is missing its cap."),
            (3.99, 7.18, "From another angle, the same bottle appears in a different position."),
            (7.74, 9.77, "Our project, Visual Peer Inspector,"),
            (10.13, 14.02, "explores how to identify an unusual item among similar products"),
            (14.48, 17.20, "and show the evidence across several photographs."),
            (17.74, 20.32, "The intended setting is small-batch inspection.")],
    "V02": [(1.35, 4.89, "The system uses the other objects in the scene as its reference."),
            (5.28, 7.42, "LocateAnything finds candidate objects,"),
            (7.84, 9.42, "SAM separates their foreground,"),
            (9.87, 12.26, "and DINOv2 extracts visual features."),
            (12.64, 16.38, "Our pipeline compares the objects and combines evidence across views."),
            (17.03, 19.90, "Sreekar will now show how this works in the web app.")],
    "V03": [(1.93, 5.67, "Our main study used 2,000 selected ToysAD-8K scenes:"),
            (5.98, 7.39, "1,000 for development,"),
            (7.59, 11.32, "600 for calibration and 400 for internal evaluation."),
            (11.80, 16.80, "With five views, assisted top-1 accuracy was 61.72%,"),
            (17.18, 19.88, "compared with 22% for Automatic."),
            (20.30, 24.25, "These are tie-aware averages across predefined view combinations."),
            (24.85, 29.97, "The assisted benchmark used dataset-supplied object matches, rather than a human-user study."),
            (30.63, 34.65, "The routes also use different scoring rules, so matching alone does not explain the gap.")],
    "V04": [(0.68, 4.87, "Automatic matching remains unreliable, and supplementary dataset results were lower."),
            (5.10, 8.94, "Historical exposure to related dataset objects also limits the evaluation."),
            (9.50, 12.81, "We have not established false-alarm rates for all-normal batches"),
            (12.81, 15.01, "or measured operator time savings."),
            (15.42, 18.43, "The potential value is a clearer inspection review process,"),
            (18.76, 22.15, "but the current prototype cannot certify that a batch is defect-free.")],
    "V05": [(1.23, 5.45, "Next, we would improve matching and independently test normal and unusual groups."),
            (5.82, 8.35, "We delivered a working prototype with measured limits,"),
            (8.74, 11.57, "turning photographs into evidence an operator can review.")],
}


def words(s):
    return len(s.replace("-", " ").split())


def timings(seg, wps=WPS):
    """Return list of (start, end) per sentence, local to the segment."""
    if seg["id"] in ACTUAL:
        return [list(x) for x in ACTUAL[seg["id"]]]
    t, out = LEAD, []
    for s in seg["sentences"]:
        d = words(s) / wps
        out.append([t, t + d])
        t += d + GAP
    limit = seg["dur"] - 0.6
    if out[-1][1] > limit:                       # compress to fit
        k = (limit - LEAD) / (out[-1][1] - LEAD)
        out = [[LEAD + (a - LEAD) * k, LEAD + (b - LEAD) * k] for a, b in out]
    return out


def caption(s):
    for a, b in CAPTION_FIX.items():
        s = s.replace(a, b)
    return s


def srt_time(x):
    if not math.isfinite(x) or x < 0:
        raise ValueError("Caption time must be finite and non-negative")
    ms = int(round(x * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def cues_for_segment_starts(starts):
    """Place the original recording cues at the actual assembled clip starts."""
    cues = []
    for seg in SEGMENTS:
        start = starts[seg["id"]]
        for a, b, text in CUES[seg["id"]]:
            cues.append({"segment": seg["id"], "start": start + a,
                         "end": start + b, "text": text})
    return cues


def read_timeline(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported caption timeline schema")
    duration = data["duration_seconds"]
    cues = data.get("cues")
    if cues is None:
        cues = cues_for_segment_starts(data["segment_starts"])
    validate_cues(cues, duration)
    return cues


def validate_cues(cues, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite")
    previous_end = 0
    for cue in cues:
        a, b = cue["start"], cue["end"]
        if not (math.isfinite(a) and math.isfinite(b) and previous_end <= a < b <= duration):
            raise ValueError(f"Invalid or overlapping caption interval: {a} to {b}")
        if not isinstance(cue["text"], str) or not cue["text"].strip():
            raise ValueError("Caption text must be non-empty")
        previous_end = b


def write_srt(cues, path):
    lines = []
    for index, cue in enumerate(cues, 1):
        lines.extend([str(index), f"{srt_time(cue['start'])} --> {srt_time(cue['end'])}",
                      "\n".join(textwrap.wrap(cue["text"], 48)), ""])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export captions for Vaibhav's spoken sections only.")
    parser.add_argument("out", nargs="?", type=Path, default=base / "out")
    parser.add_argument("--timeline", type=Path, default=base / "data/final_edit_captions.json",
                        help="Measured final-edit cues or a build-generated segment-start timeline")
    args = parser.parse_args()
    try:
        cues = read_timeline(args.timeline)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    target = args.out / "Vaibhav_narration_captions.srt"
    write_srt(cues, target)
    print(f"Wrote {len(cues)} cues to {target}")
