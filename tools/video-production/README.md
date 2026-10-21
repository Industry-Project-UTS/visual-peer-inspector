# Visual Peer Inspector video production

Source and supporting inputs for Vaibhav's sections of the UTS 32040 team video.
These offline tools are independent of the inspection web app and SageMaker.
They do not change models, ranking methods or the application.

The final video is at the [team-provided Google Drive link](https://drive.google.com/file/d/1LR95jNu69FzyNajPrL7KpzZ4HSMU7vIl/view).
The supplied local copy measures **4:00.367**, 1280 x 720, 30 fps, H.264/AAC.
Drive sharing permissions have not been independently checked. Audio recordings
and video files are excluded from this repository.

## Contents

| Path | Purpose |
| --- | --- |
| `render.py` | Draws V01-V05 and the C01 credit roll with Pillow; encodes with ffmpeg |
| `build.py` | Renders, mixes privately supplied narration, joins the app clips and exports captions |
| `narration.py` | Recording-guide text, clip-local speech cues and caption export |
| `masks.py` | Regenerates illustrative bottle cut-outs with OpenCV GrabCut |
| `align.py` | Optional offline word alignment with pocketsphinx |
| `assets/` | Two bottle photographs and a saved project-app screenshot |
| `masks/` | Six illustrative bottle cut-outs |
| `data/results_internal_lock400.json` | Compact chart values from the evaluation tables |
| `data/final_edit_captions.json` | Measured absolute caption boundaries for the final edit |
| `captions/Vaibhav_narration_captions.srt` | Matching final-edit captions for Vaibhav's speech |
| `fonts/` | Inter fonts and the supplied licence notice |

A complete build requires private inputs supplied locally:

- `voice/V01.m4a` through `voice/V05.m4a`: the original narration recordings.
- A separate folder containing Sreekar's `S01*.mp4` through `S05*.mp4` app clips.

The original archive did not contain Sreekar's recordings. The public repository
excludes all voice recordings, app videos and the finished MP4. Generated files
are written to ignored `out/`. Missing narration is reported before rendering.
Stills and caption export do not require these private recordings.

## What the visuals represent

Frames are drawn in code from the team's photos, screenshot and results. The
renderer calls no generative video or voice service. No stock footage was
supplied in the archive.

V02 is an illustration: its boxes are hand-placed, cut-outs use OpenCV GrabCut
rather than SAM 2.1, and feature bars use pixel colour rather than DINOv2. It
retains the label **Illustration of the steps, not live model output**. The app
screenshot is labelled as a previously recorded demonstration. Keep these labels
visible when editing.

Chart values come from
[`reports/remaining-cohort-workbench/internal-lock400/tables`](../../reports/remaining-cohort-workbench/internal-lock400/tables)
at commit `12cee36c2d0d78d8f9857b9150f65590ec628fe9`. All eight points, confidence
intervals and eight five-view schedule values match the CSVs. Assisted five-view
top-1 is 61.72%; Automatic is 22.00%. These are schedule-averaged, tie-aware
results on 400 internal-lock scenes. Assisted matches are dataset-supplied,
and historical asset exposure limits the study. This is not a human-user study.
Supplementary seen/unseen figures in V04 remain separate and follow the team's
final report.

## Corrected caption timing

The recording plan allocated 4:30; the final edit is 4:00.367 and includes edits
within some spoken cues. A single global offset would not align it correctly.

The default export reads `data/final_edit_captions.json`. Each cue's start and
end were located independently against the final video's audio using the private
recordings. All 58 boundary comparisons had normalized waveform correlations
above 0.97. The JSON records those scores, the final video's SHA-256 and the
alignment method. No audio is included.

| Section | Spoken cue span in the final edit |
| --- | --- |
| V01 Problem and project | 0:01.140 to 0:20.524 |
| V02 Technology and handover | 0:21.242 to 0:39.792 |
| V03 Main measured outcome | 2:41.263 to 3:14.168 |
| V04 Limits and potential value | 3:15.513 to 3:37.086 |
| V05 Next steps and conclusion | 3:38.863 to 3:49.314 |

These spans describe speech, not visual clip boundaries. The SRT covers
Vaibhav's narration only; Sreekar's narration and credits are not transcribed.
Check captions against any later edit before reusing them.

For fresh builds, `build.py` measures clips in join order and writes
`out/build-timeline.json`. It passes that timeline to `narration.py`, so closing
captions follow the actual duration of Sreekar's clips rather than assuming a
130-second middle section.

Original rendering durations and local animation cues are preserved. A fresh
build is not a byte-for-byte recreation of the manually edited final video.
Its generated captions apply to that build; the committed SRT applies to the
linked final edit.

## Requirements and use

Use Python 3.10 or newer. ffmpeg and ffprobe must be on `PATH` for video builds.

```bash
cd tools/video-production
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Final-edit captions, without private media.
python narration.py

# Visual previews, without audio or an app backend.
python render.py --stills V03 --times 4,14,30

# Full build after supplying the five private voice files locally.
python build.py --sreekar /path/to/private/sreekar_clips

# Without --sreekar, the original 130-second stand-in is used for previews.
# The private narration files are still required.
python build.py

# Captions for a fresh build's actual join timeline.
python narration.py --timeline out/build-timeline.json

# Optional helpers.
python masks.py
python align.py /path/to/local/recording.raw "the actual recording transcript"
```

OpenCV is needed to regenerate masks; pocketsphinx is needed for `align.py`.
Neither is required for caption export or ordinary visual rendering. The
renderer targets 1920 x 1080 at 30 fps, unlike the final edited 720p export.
The builder targets -16 LUFS audio and writes
`out/Visual_Peer_Inspector_video.mp4`.

## Provenance and assistance

Vaibhav supplied the original video-production archive. Its README attributes
rendering-code development to assistance from Claude (Anthropic). The recording
guide and narration draft were prepared with Codex assistance; the team supplied
photographs, app recordings and spoken narration. Repository integration, timing
corrections and documentation were prepared with Codex assistance. This records
provenance and does not establish permission under assessment AI-use rules.

Photographs: Vaibhav Bairathi. App screenshot and results: the Visual Peer
Inspector team. Fonts: Inter, with the supplied
[`fonts/LICENSE-Inter.txt`](fonts/LICENSE-Inter.txt) notice retained. Dataset:
ToysAD-8K, Bhunia et al. (2025). Presentation illustrations must not be presented
as fresh model measurements.

## Validation

Run from the repository root:

```bash
python -m unittest discover -s tools/video-production -p 'test_*.py' -v
```

Tests check final-edit captions and duration-dependent caption placement without
private recordings, GPU inference or the web app. Full video rebuilding requires
the excluded media and is separate from these checks.

Visual Peer Inspector: Sreekar Reddy Edulapalli and Vaibhav Bairathi.
Project supervisor: Dr Maoying Qiao. Subject coordinator: Rene Leveaux.
University of Technology Sydney, 32040 Industry Project.
