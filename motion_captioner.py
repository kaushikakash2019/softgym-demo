#!/usr/bin/env python3
import json, argparse, cv2, numpy as np, os
from collections import deque

def dir_from_delta(dx, dy, thr=0.8):
    # map dx,dy to words
    mag = np.hypot(dx, dy)
    if mag < thr: return "stays still"
    ang = np.degrees(np.arctan2(-dy, dx)) % 360
    if   337.5 <= ang or ang < 22.5:  return "moves right"
    elif 22.5  <= ang < 67.5:         return "moves up-right"
    elif 67.5  <= ang < 112.5:        return "moves up"
    elif 112.5 <= ang < 157.5:        return "moves up-left"
    elif 157.5 <= ang < 202.5:        return "moves left"
    elif 202.5 <= ang < 247.5:        return "moves down-left"
    elif 247.5 <= ang < 292.5:        return "moves down"
    else:                              return "moves down-right"

def wrinkle_score(gray):
    # edges per pixel as a crude “wrinkle” proxy
    edges = cv2.Canny(gray, 50, 150)
    return edges.mean() / 255.0  # 0..1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--target_fps", type=float, default=0)  # 0=process all frames
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = int(round(fps / args.target_fps)) if args.target_fps and args.target_fps>0 else 1

    prev_cx, prev_cy = None, None
    wr_hist = deque(maxlen=8)
    captions = []

    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Pink/magenta towel mask (tune as needed)
        # H in [0..179]. Magenta/pink ~ (140..179). Also include a “hot pink” band ~ (160..179).
        mask1 = cv2.inRange(hsv, (140, 60, 40), (179, 255, 255))
        # If your towel is brighter/lighter pink, widen the range:
        mask = mask1

        # centroid
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            cx, cy = prev_cx, prev_cy  # fall back, we’ll say “stays still” if no movement

        # wrinkle proxy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wr = wrinkle_score(gray)
        wr_hist.append(wr)
        # trend: compare current to short moving average
        base = np.mean(wr_hist) if len(wr_hist) > 4 else wr
        trend = "wrinkles decreasing" if wr < base - 0.02 else ("wrinkles increasing" if wr > base + 0.02 else "wrinkles stable")

        # motion
        if prev_cx is None or cx is None:
            motion = "stays still"
        else:
            motion = dir_from_delta(cx - prev_cx, cy - prev_cy)

        captions.append(f"Cloth {motion}; {trend}.")
        prev_cx, prev_cy = cx, cy

    cap.release()

    # Expand to per-frame (full length) by repeating nearest captions
    out = []
    j = 0
    for i in range(total):
        if i // step != j and (i // step) < len(captions):
            j = i // step
        j = min(j, len(captions)-1)
        out.append(captions[j])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"Wrote motion captions for {len(out)} frames -> {args.out_json}")

if __name__ == "__main__":
    main()
