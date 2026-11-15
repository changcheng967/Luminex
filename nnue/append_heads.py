"""Append the heads section (LHE1) to an engine net (LNB8 or LNI8), from the heads file fit_heads.py writes.

usage: append_heads.py NET_IN HEADS_PT NET_OUT [--draw]
       append_heads.py NET_IN --random SEED NET_OUT     (test heads of plausible size)

Section, after the net's tails (engines without heads support skip it):
  "LHE1", int32 L1, int32 buckets, int32 heads (1: the error head; 2: error and draw);
  then per head and bucket: float32 scale, int8[2*L1] weights, float32 coefficient of |v|/400, float32 bias.
The engine reads the 2*L1 activation bytes a = round(h * 127) (side to move first), so a head
w . h + wp * |v|/400 + b is stored as q = round(w * scale), scale = 127 / max|w|, and computed as
(q . a) / (127 * scale) + wp * |v|/400 + b."""
import struct, sys
import numpy as np


def net_shape(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"LNB8", b"LNI8"):
            sys.exit(f"{path}: not an int8 engine net (magic {magic!r})")
        L1, L2, L3, NI = struct.unpack("<4i", f.read(16))
        NB = struct.unpack("<i", f.read(4))[0] if magic == b"LNB8" else 1
    return magic, L1, NB


def section(heads, L1, NB):
    """heads: list of [NB, 2*L1 + 2] float arrays (weights over h, then |pred|/400, then the bias)."""
    out = bytearray(b"LHE1" + struct.pack("<3i", L1, NB, len(heads)))
    for W in heads:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (NB, 2 * L1 + 2):
            sys.exit(f"a head has shape {W.shape}, the net needs ({NB}, {2 * L1 + 2})")
        for b in range(NB):
            w, wp, bias = W[b, :2 * L1], W[b, 2 * L1], W[b, 2 * L1 + 1]
            scale = 127.0 / max(np.abs(w).max(), 1e-12)
            q = np.clip(np.round(w * scale), -127, 127).astype(np.int8)
            out += struct.pack("<f", scale) + q.tobytes() + struct.pack("<2f", wp, bias)
    return bytes(out)


def main():
    a = sys.argv[1:]
    if len(a) == 4 and a[1] == "--random":
        net_in, seed, net_out = a[0], int(a[2]), a[3]
        magic, L1, NB = net_shape(net_in)
        rng = np.random.default_rng(seed)
        heads = []
        for _ in range(2):
            W = np.zeros((NB, 2 * L1 + 2))
            W[:, :2 * L1] = rng.normal(0, 0.002, (NB, 2 * L1))
            W[:, 2 * L1] = 0.01
            W[:, 2 * L1 + 1] = 0.03
            heads.append(W)
    elif len(a) in (3, 4):
        import torch
        net_in, heads_pt, net_out = a[0], a[1], a[2]
        magic, L1, NB = net_shape(net_in)
        h = torch.load(heads_pt, map_location="cpu", weights_only=False)
        if h.get("L1") != L1 or h.get("buckets") != NB:
            sys.exit(f"{heads_pt} is for L1={h.get('L1')} buckets={h.get('buckets')}, {net_in} has L1={L1} buckets={NB}")
        heads = [h["err"].numpy()] + ([h["draw"].numpy()] if "--draw" in a[3:] else [])
    else:
        sys.exit(__doc__)
    data = open(net_in, "rb").read()
    if b"LHE1" in data[-(4 * 1024 * 1024):] and data.rfind(b"LHE1") > len(data) - 2 * (2 * L1 + 16) * NB * 4 - 64:
        sys.exit(f"{net_in} already ends in a heads section")
    sec = section(heads, L1, NB)
    open(net_out, "wb").write(data + sec)
    print(f"{net_out}: {magic.decode()} L1={L1} buckets={NB}, {len(heads)} head(s), {len(sec):,} bytes of heads after {len(data):,}")


if __name__ == "__main__":
    main()
