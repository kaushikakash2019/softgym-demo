# merge_captions.py
import json, argparse, os, re

TRIVIAL_PATTERNS = [
    r'^\s*$',  # empty
    r'^\s*cloth\s+stays\s+still.*$',   # your common string
    r'^\s*no\s+significant\s+motion.*$',
    r'^\s*wrinkles\s+stable.*$'
]
TRIVIAL_RE = re.compile("|".join(TRIVIAL_PATTERNS), re.I)

def is_trivial(s: str) -> bool:
    return TRIVIAL_RE.match(s or "") is not None

ap = argparse.ArgumentParser()
ap.add_argument("--ai_json", required=True)
ap.add_argument("--motion_json", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--mode", choices=["motion_only","concat","prefer_motion"], default="prefer_motion")
args = ap.parse_args()

ai = json.load(open(args.ai_json))
mo = json.load(open(args.motion_json))
n = min(len(ai), len(mo))
out = []
for i in range(n):
    a = (ai[i] or "").strip()
    m = (mo[i] or "").strip()
    if args.mode == "motion_only":
        out.append(m)
    elif args.mode == "concat":
        if not a: out.append(m)
        elif not m: out.append(a)
        else: out.append(f"{m} {a}")
    else:  # prefer_motion
        out.append(m if not is_trivial(m) else a)

os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
json.dump(out, open(args.out_json, "w"))
print(f"Wrote merged captions -> {args.out_json} (len={len(out)})")

