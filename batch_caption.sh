#!/usr/bin/env bash
set -euo pipefail

IN_DIR=./results/realadapt-towels-flattening-crumpled/realadapt-OTS/manipulation/performance_visualisation
OUT_DIR=./results/captioned
CAP_DIR=./captions
FPS=6

mkdir -p "$OUT_DIR" "$CAP_DIR"

for mp4 in "$IN_DIR"/episode_*.mp4; do
  base=$(basename "$mp4" .mp4)
  echo "== Processing $base =="

  python ai_caption_video_v2.py \
    --video "$mp4" \
    --out_json "$CAP_DIR/${base}_ai.json" \
    --target_fps $FPS \
    --model "Salesforce/blip-image-captioning-large" \
    --prompt ""

  python motion_captioner.py \
    --video "$mp4" \
    --out_json "$CAP_DIR/${base}_motion.json" \
    --target_fps $FPS

  python merge_captions.py \
    --ai_json "$CAP_DIR/${base}_ai.json" \
    --motion_json "$CAP_DIR/${base}_motion.json" \
    --out_json "$CAP_DIR/${base}_merged.json" \
    --mode concat

  python burn_captions_pil_v2reader.py \
    --video "$mp4" \
    --captions "$CAP_DIR/${base}_merged.json" \
    --out "$OUT_DIR/${base}_ai_motion_captioned.mp4"

  ffmpeg -v error -i "$OUT_DIR/${base}_ai_motion_captioned.mp4" -f null - && echo "OK: $base"
done
