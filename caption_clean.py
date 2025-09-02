# caption_clean.py
import re

REPL = {
    r"\bpaper\b": "towel",
    r"\bsquare\b": "cloth",
    r"\bbutton\b": "gripper",
    r"\bball\b": "gripper tip",
    r"\bcardboard\b": "table",
}
ALLOW = {"towel","cloth","wrinkle","wrinkles","fold","flat","flattened","gripper","table"}

def clean(text: str) -> str:
    s = text.lower().strip()
    for pat, repl in REPL.items():
        s = re.sub(pat, repl, s)
    # keep it short & relevant
    s = re.sub(r"^\s*(a|the)\s+", "", s)
    s = re.sub(r"[^a-z0-9 ,.-]", "", s)
    # prefer sentences mentioning towel/cloth
    if not any(w in s for w in ("towel","cloth")):
        # force context
        s = "towel on table; " + s
    # compress whitespace
    s = re.sub(r"\s{2,}", " ", s).strip(" ,.-")
    return s
