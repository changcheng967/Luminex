#!/usr/bin/env python3
"""Mine Stash CHANGELOG.md: rank all SPRT-verified patches by Elo."""
import re
import sys

text = open(sys.argv[1] if len(sys.argv) > 1 else "CHANGELOG.md", encoding="utf-8").read()
secs = re.split(r"## (v\d+\.\d+)", text)
out = []
for i in range(1, len(secs), 2):
    ver, body = secs[i], secs[i + 1]
    cur = None
    for ln in body.split("\n"):
        if ln.startswith("- "):
            cur = ln[2:95]
        m = re.search(r"ELO\s*\|\s*([+-]?\d+\.\d+)", ln)
        if m and cur:
            out.append((float(m.group(1)), ver, cur))
            cur = None
out.sort(reverse=True)
for e, v, t in out[:35]:
    print(f"{e:+8.2f} {v:>6}  {t}")
print(f"total patches with Elo: {len(out)}")
