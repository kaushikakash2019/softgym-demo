# save as merge_captions.py
import json, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--ai_json", required=True)
ap.add_argument("--motion_json", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--mode", choices=["motion_only","concat","prefer_motion","smart"], default="concat")
args = ap.parse_args()

ai = json.load(open(args.ai_json))
mo = json.load(open(args.motion_json))
n = min(len(ai), len(mo))
out = []

def combine(ai_txt, m_txt, mode="concat"):
    ai_txt = (ai_txt or "").strip()
    m_txt  = (m_txt  or "").strip()

    if mode == "motion_only":
        return m_txt
    if mode == "prefer_motion":
        return m_txt if m_txt else ai_txt
    if mode == "concat":
        if not ai_txt: return m_txt
        if not m_txt: return ai_txt
        return f"{m_txt}. {ai_txt}"

    # smart mode
    if m_txt.startswith("Cloth stays still"):
        return ai_txt or "towel remains still"
    if ai_txt and ai_txt not in m_txt:
        return f"{m_txt} — {ai_txt}"
    return m_txt or ai_txt or "towel on table"

for i in range(n):
    out.append(combine(ai[i], mo[i], mode=args.mode))

os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
json.dump(out, open(args.out_json, "w"))
print(f"Wrote merged captions -> {args.out_json} (len={len(out)})")

