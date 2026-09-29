#!/usr/bin/env python3
"""Measure chapter JSONs and committed audio. Usage: measure.py <chapters_dir> [audio_dir]
Writes JSON to stdout. Needs ffprobe only if audio_dir is given."""
import json, sys, glob, os, subprocess
cd = sys.argv[1]; ad = sys.argv[2] if len(sys.argv) > 2 else None
out = {"chapters": {}, "totals": {}}
ts = tid = ten = tgl = 0; tdur = 0.0
for f in sorted(glob.glob(os.path.join(cd, "[0-9]*.json")), key=lambda p: int(os.path.basename(p)[:-5])):
    c = json.load(open(f)); n = os.path.basename(f)[:-5]
    ss = c["sentences"]
    idi = sum(len(s.get("idioms", [])) for s in ss)
    en = sum(1 for s in ss if s.get("en")); gl = sum(1 for s in ss if s.get("gloss"))
    out["chapters"][n] = {"sentences": len(ss), "idiom_tags": idi, "en_filled": en, "gloss_filled": gl,
                          "synthetic_duration_s": c.get("duration"), "hasAudio": c.get("hasAudio")}
    ts += len(ss); tid += idi; ten += en; tgl += gl; tdur += c.get("duration") or 0
out["totals"] = {"sentences": ts, "idiom_tags": tid, "en_filled": ten, "gloss_filled": tgl,
                 "gloss_empty": ts - tgl, "idiom_tags_per_sentence": round(tid / ts, 3),
                 "synthetic_duration_total_s": round(tdur, 2)}
if ad:
    for f in sorted(glob.glob(os.path.join(ad, "ch*.mp3"))):
        n = os.path.basename(f)[2:-4]
        d = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f]).decode())
        e = out["chapters"].get(n)
        e["real_audio_s"] = round(d, 2); e["audio_bytes"] = os.path.getsize(f)
        e["real_over_synthetic"] = round(d / e["synthetic_duration_s"], 2)
json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
