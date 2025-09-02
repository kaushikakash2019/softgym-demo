# motion_captioner.py
import argparse, json, cv2, numpy as np, os

def smooth(x, k=5):
    if k <= 1: 
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(xp, kernel, mode="valid")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--target_fps", type=float, default=6)
    ap.add_argument("--low", type=float, default=0.008, help="low motion threshold")
    ap.add_argument("--high", type=float, default=0.020, help="high motion threshold")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(0.1, args.target_fps))))

    ret, prev = cap.read()
    if not ret:
        raise SystemExit("Could not read first frame")
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    mags = []
    idx = 1
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            mags.append(float(mag))
            prev = gray
        idx += 1
    cap.release()

    mags = np.array(mags, dtype=np.float32)
    mags = smooth(mags, k=5)

    captions = []
    moving = False
    still_count, move_count = 0, 0

    for m in mags:
        if moving:
            if m < args.low:
                still_count += 1
                move_count = 0
            else:
                move_count += 1
                still_count = 0
            if still_count >= 2:
                moving = False
        else:
            if m > args.high:
                moving = True
                move_count = 1
                still_count = 0

        if moving:
            captions.append("Cloth moves; wrinkles change.")
        else:
            captions.append("Cloth stays still; wrinkles stable.")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(captions, open(args.out_json, "w"))
    print(f"Wrote motion captions for {len(captions)} frames -> {args.out_json}")

if __name__ == "__main__":
    main()

