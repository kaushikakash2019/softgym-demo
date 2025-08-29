# save as merge_captions.py
import json, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--ai_json", required=True)
ap.add_argument("--motion_json", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--mode", choices=["motion_only","concat","prefer_motion"], default="concat")
args = ap.parse_args()

ai = json.load(open(args.ai_json))
mo = json.load(open(args.motion_json))
n = min(len(ai), len(mo))
out = []
for i in range(n):
    if args.mode == "motion_only":
        out.append(mo[i])
    elif args.mode == "prefer_motion":
        out.append(mo[i] if mo[i] else ai[i])
    else:
        # concat
        a = ai[i].strip()
        m = mo[i].strip()
        if not a: out.append(m)
        elif not m: out.append(a)
        else: out.append(f"{m} {a}")

os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
json.dump(out, open(args.out_json, "w"))
print(f"Wrote merged captions -> {args.out_json} (len={len(out)})")
