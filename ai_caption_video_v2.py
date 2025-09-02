# ai_caption_video_v2.py
# Clean, consistent, 4-space indentation — supports BLIP and InstructBLIP/BLIP-2 style models.

import os, json, math, argparse
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq


# ---------- utilities ----------
def center_crop(img: np.ndarray) -> np.ndarray:
    """Square center crop (keeps min side), returns HxWxC numpy array."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s, :]


def sample_frames(video_path: str, target_fps: float = 6.0):
    """Read a video with imageio-ffmpeg and sample ~target_fps."""
    meta = iio.immeta(video_path)
    src_fps = float(meta.get("fps", 30.0))
    # n_frames might be missing on some files; we’ll compute later if needed
    frames_total = int(meta.get("n_frames", 0)) or None

    step = max(1, int(round(src_fps / max(0.1, target_fps))))
    sampled = []
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx % step == 0:
            sampled.append(center_crop(frame))

    # If total frames unknown, estimate from duration
    if frames_total is None:
        duration = float(meta.get("duration", 0.0))
        if duration > 0:
            frames_total = int(round(duration * src_fps))
        else:
            # Fallback: approximate by last sampled index * step
            frames_total = len(sampled) * step

    return sampled, src_fps, frames_total


# ---------- model loaders ----------
def load_blip(model_name: str, device):
    proc = BlipProcessor.from_pretrained(model_name, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return proc, model


def load_any_blip(model_name: str, device):
    """Generic loader for InstructBLIP / BLIP-2 style Vision2Seq models."""
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # Use float16 on CUDA if available
    dtype = torch.float16 if (torch.cuda.is_available() and device.type == "cuda") else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None
    ).to(device)
    model.eval()
    return proc, model


def generate_caption_any(proc, model, pil_img: Image.Image, prompt: str = "") -> str:
    """Caption with Vision2Seq family (InstructBLIP/BLIP-2)."""
    inputs = proc(images=pil_img, text=prompt or "", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            num_beams=5,
            do_sample=False
        )
    txt = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    return txt


# ---------- captioning ----------
def caption_frames(frames, model_name: str, prompt: str, device):
    """Caption a list of numpy HxWxC frames with either BLIP (classic) or Vision2Seq (InstructBLIP/BLIP-2)."""
    name = model_name.lower()
    use_vision2seq = ("instructblip" in name) or ("blip2" in name)

    if use_vision2seq:
        proc, model = load_any_blip(model_name, device)
        def caption_one(pil_img):  # closure uses proc/model above
            return generate_caption_any(proc, model, pil_img, prompt)
    else:
        proc, model = load_blip(model_name, device)
        def caption_one(pil_img):
            inputs = proc(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=24, num_beams=5, do_sample=False)
            return proc.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    caps = []
    with torch.inference_mode():
        for f in frames:
            pil = Image.fromarray(f)
            caps.append(caption_one(pil))
    return caps


# ---------- smoothing ----------
def _token_set(text: str):
    return set(t for t in text.lower().split() if t.isalpha() or t.isalnum())


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def smooth_captions(frame_captions, sim_thresh: float = 0.72):
    """
    Group consecutive frames whose captions are similar by token Jaccard.
    Returns list of (start_idx, end_idx_inclusive, caption).
    """
    if not frame_captions:
        return []

    segments = []
    cur_start = 0
    cur_txt = frame_captions[0]

    for i in range(1, len(frame_captions)):
        sim = _jaccard(cur_txt, frame_captions[i])
        if sim >= sim_thresh:
            # continue segment, but optionally prefer the “richer” sentence
            if len(frame_captions[i]) > len(cur_txt):
                cur_txt = frame_captions[i]
        else:
            segments.append((cur_start, i - 1, cur_txt))
            cur_start = i
            cur_txt = frame_captions[i]

    segments.append((cur_start, len(frame_captions) - 1, cur_txt))
    return segments


def expand_segments_to_frames(segments, total_frames: int, step: int):
    """
    We sampled every `step` frames. Expand segment captions to a full per-frame list of length total_frames.
    """
    if total_frames <= 0:
        total_frames = segments[-1][1] * step + 1

    per_frame = [""] * total_frames
    # Build a per-sampled-index table first
    # sampled index j corresponds to original frame index ≈ j * step
    for (s, e, txt) in segments:
        for j in range(s, e + 1):
            base = j * step
            # Fill [base, base+step) with txt, clamped to total_frames
            for k in range(base, min(base + step, total_frames)):
                per_frame[k] = txt

    # Fill any remaining empties with nearest previous text
    last = ""
    for i in range(total_frames):
        if per_frame[i]:
            last = per_frame[i]
        else:
            per_frame[i] = last

    return per_frame


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--target_fps", type=float, default=6.0)
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--sim_thresh", type=float, default=0.72)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    print("Step 1) Sampling frames…")
    frames, src_fps, frames_total = sample_frames(args.video, target_fps=args.target_fps)
    print(f"Video FPS={src_fps:.2f}, Frames={frames_total}, Sampled={len(frames)}")

    print("Step 2) Loading BLIP…")
    print("Step 3) Captioning sampled frames…", end=" ", flush=True)
    sampled_caps = caption_frames(frames, model_name=args.model, prompt=args.prompt, device=device)
    print("done")

    print("Step 4) Temporal smoothing…")
    segments = smooth_captions(sampled_caps, sim_thresh=args.sim_thresh)
    print("Segments:", len(segments))
    for s, e, cap in segments[:10]:
        print(f"  [{s:4d}..{e:4d}] {cap}")

    print("Step 5) Expand to all frames + save JSON…")
    step = max(1, int(round(src_fps / max(0.1, args.target_fps))))
    full_caps = expand_segments_to_frames(segments, frames_total, step)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(full_caps, open(args.out_json, "w"))
    print(f"Wrote per-frame captions -> {args.out_json} (len={len(full_caps)})")


if __name__ == "__main__":
    main()

