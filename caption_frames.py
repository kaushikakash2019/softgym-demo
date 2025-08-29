import os, glob, json
from PIL import Image
from tqdm import tqdm

# BLIP (Salesforce) via HF Transformers
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

FRAMES_DIR = "frames/episode_0"
OUT_JSON   = "captions/episode_0_frames.json"
os.makedirs("captions", exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
    out = []
    for f in tqdm(frames, desc="Captioning frames"):
        image = Image.open(f).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=25)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        out.append({"frame": os.path.basename(f), "caption": caption})

    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Wrote {OUT_JSON} with {len(out)} captions")

if __name__ == "__main__":
    main()
