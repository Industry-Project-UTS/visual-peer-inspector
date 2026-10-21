"""
Word-level forced alignment of a narration recording to its script (pocketsphinx, offline).
Used to find the exact time each word is spoken so animations can be cued to it.

    ffmpeg -i voice/V04.m4a -ac 1 -ar 16000 -f s16le V04.raw
    python align.py V04.raw "Automatic matching remains unreliable, ..."
"""
import sys, re
from pocketsphinx import Decoder
raw, text = sys.argv[1], sys.argv[2]
words = re.sub(r"[^a-z' ]", " ", text.lower().replace("-", " ")).split()
d = Decoder(samprate=16000, bestpath=False)
d.set_align_text(" ".join(words))
data = open(raw, "rb").read()
d.start_utt(); d.process_raw(data, full_utt=True); d.end_utt()
d.set_alignment(); d.start_utt(); d.process_raw(data, full_utt=True); d.end_utt()
for w in d.get_alignment():
    print(f"{w.start/100:7.2f} {(w.start+w.duration)/100:7.2f} {w.name}")
