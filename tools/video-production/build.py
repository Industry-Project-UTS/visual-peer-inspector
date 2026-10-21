"""
Build the full presentation video in one step.

    python build.py                              # Sreekar's slot filled with a placeholder slate
    python build.py --sreekar path/to/clips      # folder containing S01*.mp4 ... S05*.mp4

Steps
  1. render.py draws V01-V05 (silent) and the credit roll C01.
  2. Each narration file in voice/ is loudness-normalised (two-pass EBU R128, -16 LUFS,
     -1.5 dBTP) and mixed onto its clip.
  3. Sreekar's five clips are conformed to 1920x1080, 30 fps, 48 kHz stereo at the same
     loudness and joined (or a 130 s placeholder slate is rendered).
  4. Everything is joined in order: V01, V02, Sreekar, V03, V04, V05, C01.
  5. narration.py writes captions using the assembled clips' actual start times.
"""
import os, sys, json, glob, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CLIPS = {"V01": 25, "V02": 25, "V03": 40, "V04": 25, "V05": 15}


def run(cmd):
    print("  $", " ".join(cmd) if len(" ".join(cmd)) < 160 else " ".join(cmd)[:160] + " ...")
    subprocess.run(cmd, check=True)


def loudnorm_filter(path, pre="aresample=48000,pan=stereo|c0=c0|c1=c0"):
    """First loudnorm pass: measure, then return a linear second-pass filter."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af",
                        f"{pre},loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1]
    m = json.loads(txt)
    return (f"{pre},loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")


def mix_voice(seg, dur):
    voice = os.path.join(HERE, "voice", f"{seg}.m4a")
    silent = os.path.join(OUT, f"{seg}_silent.mp4")
    f = loudnorm_filter(voice)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", voice, "-filter_complex",
         f"[1:a]{f},aresample=48000,apad=whole_dur={dur}[a]", "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", str(dur), "-movflags", "+faststart",
         os.path.join(OUT, f"{seg}.mp4")])


def conform(src, dst):
    """Bring an external clip to the same format as the rendered clips."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                            "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
    vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xFAF9F7,fps=30,format=yuv420p"
    if probe:
        f = loudnorm_filter(src, pre="aresample=48000,aformat=channel_layouts=stereo")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-af", f + ",aresample=48000"]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
               "-vf", vf, "-shortest", "-map", "0:v", "-map", "1:a"]
    run(cmd + ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "128k",
               "-ar", "48000", "-ac", "2", "-movflags", "+faststart", dst])


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout)


def caption_timeline(order, final_duration):
    """Use the joined clip durations, including any length of Sreekar's section."""
    starts, cursor = {}, 0.0
    for path in order:
        segment = os.path.splitext(os.path.basename(path))[0]
        if segment in CLIPS:
            starts[segment] = cursor
        cursor += duration(path)
    return {"schema_version": 1, "duration_seconds": final_duration,
            "segment_starts": starts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sreekar", help="folder with Sreekar's S01-S05 clips")
    ap.add_argument("--skip-render", action="store_true", help="reuse clips already in out/")
    args = ap.parse_args()
    missing = [os.path.join("voice", f"{segment}.m4a") for segment in CLIPS
               if not os.path.isfile(os.path.join(HERE, "voice", f"{segment}.m4a"))]
    if missing:
        ap.error("Private narration recordings are not committed. Supply locally: " + ", ".join(missing))
    os.makedirs(OUT, exist_ok=True)

    print("1. Rendering clips")
    if not args.skip_render:
        run([sys.executable, os.path.join(HERE, "render.py"), "--out", OUT])

    print("2. Mixing narration")
    for seg, dur in CLIPS.items():
        mix_voice(seg, dur)

    print("3. Sreekar's section")
    middle = []
    if args.sreekar:
        for i in range(1, 6):
            found = sorted(glob.glob(os.path.join(args.sreekar, f"S0{i}*")))
            if not found: raise SystemExit(f"missing S0{i} clip in {args.sreekar}")
            dst = os.path.join(OUT, f"S0{i}.mp4"); conform(found[0], dst); middle.append(dst)
        total = sum(duration(p) for p in middle)
        print(f"   Sreekar's section runs {total:.1f} s (planned 130 s)")
    else:
        run([sys.executable, os.path.join(HERE, "render.py"), "--out", OUT, "--only", "SLATE"])
        middle = [os.path.join(OUT, "SLATE.mp4")]

    print("4. Joining the final video")
    order = [os.path.join(OUT, f"{s}.mp4") for s in ("V01", "V02")] + middle + \
            [os.path.join(OUT, f"{s}.mp4") for s in ("V03", "V04", "V05", "C01")]
    lst = os.path.join(OUT, "concat.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p.replace(os.sep, '/')}'\n" for p in order)
    final = os.path.join(OUT, "Visual_Peer_Inspector_video.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
         "-movflags", "+faststart", final])

    print("5. Captions")
    d = duration(final)
    timeline_path = os.path.join(OUT, "build-timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as fh:
        json.dump(caption_timeline(order, d), fh, indent=2)
        fh.write("\n")
    run([sys.executable, os.path.join(HERE, "narration.py"), OUT, "--timeline", timeline_path])
    print(f"\nDone: {final}  ({int(d // 60)}:{d % 60:04.1f}; required range 3:00-5:00)")
