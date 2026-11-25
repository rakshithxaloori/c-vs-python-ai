"""Inject results/summary.json (+ a few sample images) into report/template.html,
producing report/index.html, a single self-contained page."""

import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "results", "summary.json")) as f:
    summary = json.load(f)
with open(os.path.join(ROOT, "data", "meta.json")) as f:
    meta = json.load(f)
with open(os.path.join(ROOT, "data", "samples.json")) as f:
    raw = json.load(f)
summary["n_params"] = meta["n_params"]

# Two images per class, ordered so the strip reads class by class.
items, seen = [], {}
for s in raw["samples"]:
    if seen.get(s["cls"], 0) < 2:
        seen[s["cls"]] = seen.get(s["cls"], 0) + 1
        items.append({"cls": s["cls"], "b64": base64.b64encode(bytes(s["px"])).decode()})
items.sort(key=lambda s: s["cls"])
samples = {"classes": raw["classes"], "items": items}

with open(os.path.join(ROOT, "report", "template.html")) as f:
    html = f.read()
html = html.replace("/*__DATA__*/null", json.dumps(summary, separators=(",", ":")))
html = html.replace("/*__SAMPLES__*/null", json.dumps(samples, separators=(",", ":")))
out = os.path.join(ROOT, "report", "index.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
