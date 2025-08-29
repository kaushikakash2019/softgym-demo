import os, re, json, math, argparse
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import cv2

from transformers import BlipProcessor, BlipForConditionalGeneration

def normalize(t: str):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [w for w in t.split() if w]
    return set(toks)

def jaccard(a: str, b: str) -> float:
    sa, sb = normalize(a), normalize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))

def sample_frames_cv(video_path: str, target_fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / max(1e-6, target_fps))))
    indices = list(range(0, total, stride))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx, Image.fromarray(rgb)))
    cap.release()
    return fps, total, frames

def caption_frames(frames: List[Tuple[int, Image.Image]], model_name: str, prompt: str, device: str):
    print("Step 2) Loading BLIP…")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True
    ).to(device)

    model.eval()

    results = []
    B = 8
    imgs = [im for _, im in frames]
    idxs = [i for i, _ in frames]
    for s in range(0, len(imgs), B):
        batch = imgs[s:s+B]
        text = [prompt] * len(batch) if prompt else None
        with torch.inference_mode():
            inputs = processor(images=batch, text=text, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=30)
            caps = processor.batch_decode(out, skip_special_tokens=True)
        for i, cap in zip(idxs[s:s+B], caps):
            results.append((i, cap.strip()))
    return results

def smooth_captions(samples: List[Tuple], sim_thresh: float = 0.6, min_len: int = 1):
    """
    samples: list of (frame_idx, caption) OR (frame_idx, score, caption)
    returns: list of (start_idx, end_idx, caption)
    """
    if not samples:
        return []

    # Normalize tuples to (idx, cap)
    norm = []
    for s in samples:
        if len(s) == 2:
            idx, cap = s
        elif len(s) >= 3:
            idx, cap = s[0], s[-1]
        else:
            # Unexpected, skip
            continue
        norm.append((int(idx), str(cap)))

    # Sort by frame index
    norm.sort(key=lambda x: x[0])

    segments = []
    cur_start, cur_cap = norm[0][0], norm[0][1]
    prev_idx = norm[0][0]

    for idx, cap in norm[1:]:
        sim = jaccard(cur_cap, cap)
        # If highly similar, extend segment; else close and start a new segment
        if sim >= sim_thresh:
            prev_idx = idx
            continue
        else:
            segments.append((cur_start, prev_idx, cur_cap))
            cur_start, cur_cap = idx, cap
            prev_idx = idx

    # close last
    segments.append((cur_start, prev_idx, cur_cap))

    # Enforce minimum length in frames (optional)
    if min_len > 1 and len(segments) > 1:
        merged = []
        for seg in segments:
            s, e, c = seg
            if (e - s + 1) >= min_len or not merged:
                merged.append(seg)
            else:
                # merge short segment into previous
                ps, pe, pc = merged[-1]
                # decide which caption to keep via similarity; prefer previous
                if jaccard(pc, c) >= 0.5:
                    merged[-1] = (ps, e, pc)
                else:
                    merged[-1] = (ps, pe, pc)  # keep previous cap; ignore short seg
        segments = merged

    return segments

def expand_to_all_frames(segments, total_frames):
    """
    segments: list of (start_idx, end_idx, caption) over **frame indices**
    returns: list[str] of length total_frames (per-frame captions)
    """
    out = [""] * total_frames
    if not segments:
        return out
    segments = sorted(segments, key=lambda x: x[0])
    # Fill before first segment with its caption
    first_s, first_e, first_c = segments[0]
    for i in range(0, max(0, min(total_frames, first_s))):
        out[i] = first_c
    # Fill each segment
    for s, e, c in segments:
        s = max(0, min(total_frames - 1, int(s)))
        e = max(0, min(total_frames - 1, int(e)))
        if e < s:
            s, e = e, s
        for i in range(s, e + 1):
            out[i] = c
    # Fill after last
    last_s, last_e, last_c = segments[-1]
    for i in range(min(total_frames, last_e + 1), total_frames):
        out[i] = last_c
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--target_fps", type=float, default=2.0)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--sim_thresh", type=float, default=0.6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Step 1) Sampling frames…")
    fps, total_frames, frames = sample_frames_cv(args.video, args.target_fps)
    print(f"Video FPS={fps:.2f}, Frames={total_frames}, Sampled={len(frames)}")

    frame_caps = caption_frames(frames, model_name=args.model, prompt=args.prompt, device=device)

    print("Step 3) Captioning sampled frames… done")
    print("Step 4) Temporal smoothing…")
    segments = smooth_captions(frame_caps, sim_thresh=args.sim_thresh)
    print(f"Segments: {len(segments)}")
    for s, e, c in segments[:5]:
        print(f"  [{s:>4}..{e:>4}] {c}")

    print("Step 5) Expand to all frames + save JSON…")
    per_frame_caps = expand_to_all_frames(segments, total_frames=total_frames)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(per_frame_caps, f, ensure_ascii=False, indent=2)

    print(f"Wrote per-frame captions -> {args.out_json} (len={len(per_frame_caps)})")
    # small sanity sample
    picks = [0, min(10, total_frames-1), max(0,total_frames//2), max(0,total_frames-1)]
    picks = sorted(set(picks))
    for p in picks:
        print(f"  frame {p:>4}: {per_frame_caps[p]}")

if __name__ == "__main__":
    main()
