import imageio
import json, argparse, os, time
import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

DEF_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def load_font(pt):
    try:
        return ImageFont.truetype(DEF_FONT, pt)
    except Exception:
        return ImageFont.load_default()

def text_wrap(draw, text, font, max_width):
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        wpx = draw.textlength(test, font=font)
        if wpx <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def fit_caption(draw, text, target_width, max_lines, start_pt):
    pt = start_pt
    while pt >= 10:
        font = load_font(pt)
        lines = text_wrap(draw, text, font, target_width)
        if len(lines) <= max_lines:
            ascent, descent = font.getmetrics()
            h = (ascent + descent) * len(lines)
            if h <= 0.4 * target_width:
                return font, lines
        pt -= 2
    font = load_font(10)
    return font, text_wrap(draw, text, font, target_width)

def draw_caption_box(draw, xy, w, h, radius=10, fill=(0,0,0,180)):
    x, y = xy
    r = min(radius, int(min(w,h)/3))
    draw.rounded_rectangle([x, y, x+w, y+h], radius=r, fill=fill)

def draw_text_with_outline(draw, xy, lines, font, fill=(255,255,255,255), outline=(0,0,0,255), line_spacing_scale=1.05):
    x, y = xy
    # robust line height
    bbox = font.getbbox("Ay")
    lh = (bbox[3] - bbox[1])
    lh = int(lh * line_spacing_scale)
    for i, line in enumerate(lines):
        yy = y + i * lh
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text((x+dx, yy+dy), line, font=font, fill=outline)
        draw.text((x, yy), line, font=font, fill=fill)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--captions", required=True, help="JSON: list(str) or dict frame->str")
    ap.add_argument("--out", required=True)
    ap.add_argument("--font_size", type=int, default=28)
    ap.add_argument("--max_width_pct", type=float, default=0.9)
    ap.add_argument("--max_lines", type=int, default=3)
    ap.add_argument("--margin_px", type=int, default=14)
    ap.add_argument("--line_spacing_scale", type=float, default=1.05)
    args = ap.parse_args()

    t0 = time.time()

    with open(args.captions, "r") as f:
        cdata = json.load(f)
    if isinstance(cdata, dict):
        max_idx = max(int(k) for k in cdata.keys()) if cdata else -1
        caps = [cdata.get(str(i), "") for i in range(max_idx+1)]
    else:
        caps = list(cdata)

    reader = iio.imiter(args.video)
    meta = iio.immeta(args.video)
    fps = float(meta.get("fps", 30))
    first = next(reader)
    H, W = first.shape[:2]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=9)

    # re-open to include frame 0
    reader = iio.imiter(args.video)

    max_text_width = int(W * args.max_width_pct)
    margin = args.margin_px

    count = 0
    for idx, frame in enumerate(reader):
        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)

        text = caps[idx] if idx < len(caps) else ""

        font, lines = fit_caption(draw, text, target_width=max_text_width, max_lines=args.max_lines, start_pt=args.font_size)

        bbox = font.getbbox("Ay")
        lh = (bbox[3] - bbox[1])
        lh = int(lh * args.line_spacing_scale)
        text_w = max((draw.textlength(line, font=font) for line in lines), default=0)
        text_h = lh * max(1, len(lines))

        box_w = int(text_w + 2*margin)
        box_h = int(text_h + 2*margin)
        x = max((W - box_w)//2, margin)
        y = max(H - box_h - margin, margin)

        draw_caption_box(draw, (x, y), box_w, box_h, radius=10, fill=(0,0,0,180))
        text_x, text_y = x + margin, y + margin
        draw_text_with_outline(draw, (text_x, text_y), lines, font, line_spacing_scale=args.line_spacing_scale)

        composed = Image.alpha_composite(img, overlay).convert("RGB")
        writer.append_data(np.asarray(composed))
        count += 1

    writer.close()
    dur = time.time() - t0
    print(f"Caption burn complete: wrote {count} frames to {args.out}")
    print(f"Runtime: {dur:.2f} sec  |  FPS: {count/max(dur,1e-6):.1f}")

if __name__ == "__main__":
    main()
