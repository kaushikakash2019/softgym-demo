import json, os, textwrap

JSON_IN = "captions/episode_0_frames.json"
ASS_OUT = "captions/episode_0.ass"
FPS     = 2.0  # must match the extraction fps used earlier

# Very simple ASS template
ASS_HEADER = r"""[Script Info]
ScriptType: v4.00+
PlayResX: 256
PlayResY: 256

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,20,&H00FFFFFF,&H000000FF,&HAA000000,&HAA000000,0,0,0,0,100,100,0,0,1,2,0,9,10,10,10,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def to_ass_time(t):
    h = int(t // 3600); t -= h*3600
    m = int(t // 60);   t -= m*60
    s = int(t)
    cs = int((t - s)*100)  # centiseconds
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def main():
    with open(JSON_IN) as f:
        items = json.load(f)

    # Each frame lasts 1/FPS seconds. We show caption from frame_n to frame_{n+1}
    lines = [ASS_HEADER]
    for i, it in enumerate(items):
        start = i / FPS
        end   = (i + 1) / FPS
        cap   = it["caption"].replace("\n", " ").strip()
        # wrap long lines
        cap_wrapped = "\\N".join(textwrap.wrap(cap, width=28)) or " "
        lines.append(f"Dialogue: 0,{to_ass_time(start)},{to_ass_time(end)},Caption,,0,0,0,,{cap_wrapped}")

    with open(ASS_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {ASS_OUT}")

if __name__ == "__main__":
    main()
