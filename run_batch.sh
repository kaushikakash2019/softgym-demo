#!/usr/bin/env bash
# Robust batch runner (won't auto-close your terminal)

set -u  # don't use "set -e" so errors don't kill the session

# Always operate from this script's directory (repo root)
cd "$(dirname "$0")"

# Activate conda env if available
if [ -f "/home/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/$USER/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda/etc/profile.d/conda.sh" ]; then
  source "/home/$USER/miniconda/etc/profile.d/conda.sh"
fi
conda activate softgym >/dev/null 2>&1 || true

# Paths
VIDEO_DIR="./results/realadapt-towels-flattening-crumpled/realadapt-OTS/manipulation/performance_visualisation"
mkdir -p captions artifacts

process_episode () {
  local EP="$1"
  local VIDEO="$VIDEO_DIR/episode_${EP}.mp4"

  if [[ ! -f "$VIDEO" ]]; then
    echo "[SKIP] $VIDEO not found"
    return 0
  fi

  echo "=== Episode $EP ==="
  local AI_JSON="./captions/episode_${EP}_ai.json"
  local MOTION_JSON="./captions/episode_${EP}_motion.json"
  local MERGED_JSON="./captions/episode_${EP}_merged.json"
  local OUT_MP4="./artifacts/episode_${EP}_ai_motion_captioned_readable.mp4"

  local START_TOT=$SECONDS

  echo "[AI] captioning…"
  python ai_caption_video_v2.py \
    --video "$VIDEO" \
    --out_json "$AI_JSON" \
    --target_fps 6 \
    --model "Salesforce/blip-image-captioning-large" \
    --prompt "" || echo "[WARN] AI caption step failed (continuing)"

  echo "[MOTION] captioning…"
  python motion_captioner.py \
    --video "$VIDEO" \
    --out_json "$MOTION_JSON" \
    --target_fps 6 \
    --low 0.008 --high 0.020 || echo "[WARN] Motion caption step failed (continuing)"

  echo "[MERGE] combining…"
  python merge_captions.py \
    --ai_json "$AI_JSON" \
    --motion_json "$MOTION_JSON" \
    --out_json "$MERGED_JSON" \
    --mode prefer_motion || echo "[WARN] Merge step failed (continuing)"

  echo "[BURN] writing video…"
  python burn_captions_pil_v2reader.py \
    --video "$VIDEO" \
    --captions "$MERGED_JSON" \
    --out "$OUT_MP4" \
    --font_size 30 --max_width_pct 0.92 --max_lines 3 \
    --margin_px 18 --box_alpha 0.65 --box_pad 10 --stroke_px 2 || echo "[WARN] Burn step failed (continuing)"

  if ffmpeg -v error -i "$OUT_MP4" -f null - >/dev/null 2>&1; then
    echo "[OK] $OUT_MP4"
  else
    echo "[ERR] Corrupt output: $OUT_MP4"
  fi

  local ELAPSE_TOT=$(( SECONDS - START_TOT ))
  echo "[TIME] Episode $EP total: ${ELAPSE_TOT}s"
  echo
}

# ---- SELECT EPISODES HERE ----
for EP in 0; do
  process_episode "$EP"
done
