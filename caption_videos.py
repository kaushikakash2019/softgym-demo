import os, glob, json, argparse, torch
from PIL import Image
from transformers import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="./frames", help="root containing episode_* folders")
    ap.add_argument("--out_root", default="./captions", help="where to write captions")
    ap.add_argument("--sample_every", type=int, default=5, help="caption every Nth frame")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    cap = pipeline("image-to-text", model=args.model, device=device)

    ep_dirs = sorted([d for d in glob.glob(os.path.join(args.frames_root, "episode_*")) if os.path.isdir(d)])

    for ep in ep_dirs:
        ep_name = os.path.basename(ep)
        frame_paths = sorted(glob.glob(os.path.join(ep, "frame_*.png")))
        if not frame_paths:
            print(f"[skip] no frames in {ep}")
            continue

        sampled = [p for idx, p in enumerate(frame_paths, start=1) if idx % args.sample_every == 0]
        if not sampled:
            sampled = frame_paths  # fallback

        per_frame = []
        for p in sampled:
            try:
                img = Image.open(p).convert("RGB")
                out = cap(img, max_new_tokens=25)[0]["generated_text"].strip()
            except Exception as e:
                out = f"[ERROR] {e}"
            per_frame.append({"frame": os.path.basename(p), "caption": out})

        # naive summarization: keep the most frequent short caption
        from collections import Counter
        texts = [c["caption"] for c in per_frame if not c["caption"].startswith("[ERROR]")]
        summary = ""
        if texts:
            common = Counter(texts).most_common(1)[0][0]
            # lightly normalize
            summary = common[0].upper() + common[1:] if common else "A cloth manipulation scene."

        # write per-frame jsonl
        jsonl_path = os.path.join(args.out_root, f"{ep_name}_frames.jsonl")
        with open(jsonl_path, "w") as f:
            for row in per_frame:
                f.write(json.dumps(row) + "\n")

        # write summary txt
        txt_path = os.path.join(args.out_root, f"{ep_name}.txt")
        with open(txt_path, "w") as f:
            f.write(summary + "\n")

        print(f"[done] {ep_name}: {len(per_frame)} frames -> {txt_path}")
    print("[all done]")

if __name__ == "__main__":
    main()
