#!/usr/bin/env python3
import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio
import imageio.v2 as iio_v2
import imageio
import imageio_ffmpeg as iio_ffmpeg

def load_font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Book.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_caption(img: Image.Image, text: str, font: ImageFont.FreeTypeFont):
    if not text:
        return img
    draw = ImageDraw.Draw(img, "RGBA")
    pad, x, y = 8, 10, 10
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    return img

def probe_meta(video_path):
    try:
        nframes, duration = iio_ffmpeg.count_frames_and_secs(video_path)
        nframes = int(nframes) if nframes is not None else None
        duration = float(duration) if duration not in (None, 0) else None
        fps = (nframes / duration) if (nframes and duration) else 30.0
        return float(fps), nframes
    except Exception:
        # fallback: decode-count
        try:
            n = 0
            for _ in iio.imiter(video_path):
                n += 1
            return 30.0, n
        except Exception:
            return 30.0, None

def make_caption_getter(cap_map):
    """
    Return a function get_caption(idx)->str that works for:
      - dict {"0": "txt", ...} or {0: "txt", ...}
      - list ["txt0", "txt1", ...]
      - list of dicts [{"caption": "txt0"}, ...] or [{"text": "txt0"}, ...]
    """
    if isinstance(cap_map, dict):
        # normalize keys to str
        return lambda i: cap_map.get(str(i), cap_map.get(i, ""))
    if isinstance(cap_map, list):
        # list of strings?
        if all(isinstance(x, str) for x in cap_map):
            return lambda i: cap_map[i] if 0 <= i < len(cap_map) else ""
        # list of dicts with 'caption' or 'text' field?
        if all(isinstance(x, dict) for x in cap_map):
            def _get(i):
                if 0 <= i < len(cap_map):
                    d = cap_map[i]
                    return d.get("caption") or d.get("text") or ""
                return ""
            return _get
    # unknown shape
    return lambda i: ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input MP4")
    ap.add_argument("--captions", required=True, help="JSON mapping/list of captions per frame")
    ap.add_argument("--out", required=True, help="Output MP4 with burned captions")
    ap.add_argument("--fontsize", type=int, default=18)
    args = ap.parse_args()

    with open(args.captions, "r") as f:
        cap_map = json.load(f)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fps, _ = probe_meta(args.video)
    font = load_font(args.fontsize)
    get_caption = make_caption_getter(cap_map)

    reader = iio_v2.get_reader(args.video, format='ffmpeg')
    writer = imageio.get_writer(
        args.out,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        quality=8,
    )

    try:
        idx = 0
        for frame in reader:
            img = Image.fromarray(frame)
            txt = get_caption(idx)
            img = draw_caption(img, txt, font)
            writer.append_data(np.asarray(img))
            idx += 1
    finally:
        try:
            reader.close()
        except Exception:
            pass
        writer.close()

if __name__ == "__main__":
    main()
