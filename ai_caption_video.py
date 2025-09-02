# paste the code, save
#!/usr/bin/env python3
"""
ai_caption_video.py
- Samples frames from an input video
- Captions each sampled frame with BLIP
- Smooths/merges adjacent similar captions
- Expands to per-frame captions list
- Writes JSON compatible with burn_captions_pil_v2reader.py
"""

import argparse, os, json, math, difflib
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

def sample_frames(video_path, target_fps=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(orig_fps / float(target_fps)))
    sampled = []          # (frame_index, BGR image)
    all_count = 0

    pbar = tqdm(total=total, desc="Sampling frames", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if all_count % step == 0:
            sampled.append((all_count, frame))
        all_count += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return sampled, int(orig_fps), total

def load_blip(device):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(device)
    model.eval()
    return processor, model

@torch.inference_mode()
def caption_frame(image_bgr, processor, model, device, prompt=None, max_new_tokens=20):
    # convert BGR->RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if prompt:
        inputs = processor(images=image_rgb, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return caption

def smooth_captions(samples, sim_thresh=0.80):
    """
    Merge adjacent captions if they're very similar.
    samples: list of (frame_idx, caption)
    return: list of segments [(start_idx, end_idx, caption)]
    """
    if not samples:
        return []

    def similar(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

    segments = []
    cur_start, _, cur_cap = samples[0]
    last_idx = samples[0][0]

    for i in range(1, len(samples)):
        idx, cap = samples[i]
        if similar(cap, cur_cap) >= sim_thresh:
            last_idx = idx
            continue
        else:
            segments.append((cur_start, last_idx, cur_cap))
            cur_start, cur_cap, last_idx = idx, cap, idx

    segments.append((cur_start, last_idx, cur_cap))
    return segments

def expand_to_per_frame(segments, total_frames):
    """
    segments: [(start_idx, end_idx, caption)] on sampled indices
    Fill every frame 0..N-1 with nearest segment caption.
    """
    caps = [""] * total_frames
    # If no segments, leave blank
    if not segments:
        return caps

    # Build a piecewise constant function from sampled idx ranges
    for (start_i, end_i, cap) in segments:
        for i in range(start_i, end_i + 1):
            if 0 <= i < total_frames:
                caps[i] = cap

    # Forward/backward fill any remaining blanks
    last = ""
    for i in range(total_frames):
        if caps[i] == "":
            caps[i] = last
        else:
            last = caps[i]
    # If still all empty (edge case), put a default caption
    if all(c == "" for c in caps):
        caps = ["Action in progress"] * total_frames
    return caps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input mp4")
    ap.add_argument("--out_json", required=True, help="Where to write per-frame captions JSON")
    ap.add_argument("--target_fps", type=float, default=2.0, help="Sampling FPS for captioning")
    ap.add_argument("--prompt", type=str, default="",
                    help="Optional caption prompt (e.g., 'a robot flattening a towel')")
    ap.add_argument("--sim_thresh", type=float, default=0.80, help="Merge adjacent captions if similar >= thresh")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    args = ap.parse_args()

    video = Path(args.video)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Step 1) Sampling frames…")
    samples, orig_fps, total_frames = sample_frames(video, target_fps=args.target_fps)
    print(f"Video FPS={orig_fps:.2f}, Frames={total_frames}, Sampled={len(samples)}")

    print("Step 2) Loading BLIP…")
    processor, model = load_blip(device)

    print("Step 3) Captioning sampled frames…")
    frame_captions = []
    for idx, bgr in tqdm(samples, desc="Captioning"):
        cap = caption_frame(
            bgr, processor, model, device,
            prompt=(args.prompt or None),
            max_new_tokens=args.max_new_tokens
        )
        frame_captions.append((idx, cap))

    print("Step 4) Temporal smoothing…")
    segments = smooth_captions(frame_captions, sim_thresh=args.sim_thresh)
    print("Segments:")
    for s,e,c in segments:
        print(f"  [{s:>4} .. {e:>4}]  {c}")

    print("Step 5) Expand to per-frame JSON…")
    per_frame = expand_to_per_frame(segments, total_frames)

    with open(out_json, "w") as f:
        json.dump(per_frame, f, ensure_ascii=False, indent=None)

    print(f"Wrote per-frame captions to: {out_json}")

if __name__ == "__main__":
    main()

