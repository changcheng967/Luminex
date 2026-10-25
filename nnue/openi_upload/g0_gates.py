#!/usr/bin/env python3
"""G0.1 / G0.2 measurement battery for an LNN1 float net (see v2 design docs).

G0.1  affine calibration: fit a*pred + b against targets on held-out data.
      a != 1 means eval-scale shrinkage the sigmoid-power loss cannot see.
G0.2  ridge linear head on the FROZEN accumulator: how much of the target is
      linear in the FT output?  R2_lin vs R2_full splits the eval into its
      linear and interaction components and decides which v2 lever to buy.

Input: a featurized .rec stream (140-byte records from src/featurize
--fen-eval) plus the float .nnue.  numpy only.

    xz -dc slice.tsv.xz | ./featurize --fen-eval --threads 8 > slice.rec
    python g0_gates.py net.nnue slice.rec [n_positions]
"""
import sys, struct
import numpy as np

NI = 24576
REC = 140


def load_net(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'LNN1', "not a float LNN1 net"
        L1, L2, L3, NIdecl = struct.unpack('iiii', f.read(16))

        def rt():
            (n,) = struct.unpack('i', f.read(4))
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

        ft_w = rt().reshape(L1, NI)      # [channel, feature] — exporter writes EmbeddingBag.T()
        ft_b = rt()
        l2_w = rt().reshape(L2, 2 * L1); l2_b = rt()
        l3_w = rt().reshape(L3, L2);     l3_b = rt()
        out_w = rt().reshape(1, L3);     out_b = rt()[0]
    return dict(L1=L1, ft_w=ft_w, ft_b=ft_b, l2_w=l2_w, l2_b=l2_b,
                l3_w=l3_w, l3_b=l3_b, out_w=out_w, out_b=out_b)


def load_records(path, n_max):
    rows = []
    with open(path, 'rb') as f:
        while len(rows) < n_max:
            buf = f.read(REC * 8192)
            if not buf:
                break
            m = len(buf) // REC
            rows.append(np.frombuffer(buf[:m * REC], dtype=np.uint8).reshape(m, REC))
    a = np.concatenate(rows)[:n_max]
    w = a[:, :64].copy().view(np.int16).reshape(-1, 32).astype(np.int64)
    b = a[:, 64:128].copy().view(np.int16).reshape(-1, 32).astype(np.int64)
    s = a[:, 128:132].copy().view(np.float32).reshape(-1)
    t = a[:, 132:136].copy().view(np.float32).reshape(-1)
    return w, b, s, t


def accumulate(ft_w, ft_b, idx):
    """acc[i] = sum over real (non-pad) features of ft_w[:, idx[i,j]] + bias
    (ft_w is channel-major [L1, NI], matching the exporter's transpose)."""
    acc = np.tile(ft_b, (idx.shape[0], 1)).astype(np.float32)
    for j in range(idx.shape[1]):
        col = idx[:, j]
        ok = col < NI
        if ok.any():
            acc += ft_w[:, np.where(ok, col, 0)].T * ok.astype(np.float32)[:, None]
    return acc


def screlu(x):
    return np.clip(x, 0.0, 1.0) ** 2


def forward(net, w, b, s, batch=8192):
    preds = np.empty(len(w), dtype=np.float32)
    accs = np.empty((len(w), 2 * net["L1"]), dtype=np.float32)
    for i in range(0, len(w), batch):
        aw = accumulate(net["ft_w"], net["ft_b"], w[i:i + batch])
        ab = accumulate(net["ft_w"], net["ft_b"], b[i:i + batch])
        stm_is_white = s[i:i + batch] > 0.5
        a_stm = np.where(stm_is_white[:, None], aw, ab)
        a_nstm = np.where(stm_is_white[:, None], ab, aw)
        h = np.concatenate([a_stm, a_nstm], axis=1)
        accs[i:i + batch] = h
        h = screlu(h)
        h = screlu(h @ net["l2_w"].T + net["l2_b"])
        h = screlu(h @ net["l3_w"].T + net["l3_b"])
        preds[i:i + batch] = (h @ net["out_w"].T + net["out_b"]).ravel() * 300.0
    return preds, accs


def r2(y, yhat):
    ss = np.sum((y - yhat) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss / st


def main():
    net_path, rec_path = sys.argv[1], sys.argv[2]
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    net = load_net(net_path)
    w, b, s, t = load_records(rec_path, n_max)
    t = t.astype(np.float64)
    print(f"positions: {len(t)}   target std: {t.std():.1f}cp  MAE(floor0): {np.abs(t).mean() * 0 + 0:.0f}")

    preds, accs = forward(net, w, b, s)
    preds = preds.astype(np.float64)

    # sanity: predictions must POSITIVELY correlate with targets; a negative
    # correlation means the stm convention is flipped
    c = np.corrcoef(preds, t)[0, 1]
    print(f"corr(pred, target) = {c:.4f}   (must be positive)")
    assert c > 0.3, "stm convention looks wrong - investigate before trusting results"

    mae = np.abs(preds - t).mean()
    print(f"\nnet MAE = {mae:.2f}cp   pred std = {preds.std():.1f}   "
          f"R2_full = {r2(t, preds):.4f}")

    # ---- G0.1: affine calibration a*pred + b ----
    X = np.column_stack([preds, np.ones_like(preds)])
    (a, b0), *_ = np.linalg.lstsq(X, t, rcond=None)
    fit = a * preds + b0
    print(f"\n[G0.1] a = {a:.4f}   b = {b0:+.2f}cp   "
          f"R2 {r2(t, preds):.4f} -> {r2(t, fit):.4f}   "
          f"MAE {mae:.2f} -> {np.abs(fit - t).mean():.2f}")
    err = np.abs(preds - t)
    print(f"      |err| p50/p90/p99 = {np.percentile(err,50):.0f}/"
          f"{np.percentile(err,90):.0f}/{np.percentile(err,99):.0f}cp   "
          f"share of SSE in worst 10% of positions: "
          f"{np.sort(err**2)[-max(1,len(err)//10):].sum() / (err**2).sum():.2f}")

    # ---- G0.2: ridge linear head on the frozen accumulator ----
    n_tr = len(t) * 3 // 4
    A = accs.astype(np.float64)
    mu, sd = A[:n_tr].mean(0), A[:n_tr].std(0) + 1e-6
    An = (A - mu) / sd
    best = None
    for lam in (1e-2, 1.0, 100.0, 1e4):
        Xa = np.column_stack([An[:n_tr], np.ones(n_tr)])
        wreg = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ (t[:n_tr] - t.mean()))
        pred_tr = Xa @ wreg + t.mean()
        pred_te = np.column_stack([An[n_tr:], np.ones(len(t) - n_tr)]) @ wreg + t.mean()
        rr = r2(t[n_tr:], pred_te)
        mae_r = np.abs(pred_te - t[n_tr:]).mean()
        print(f"[G0.2] lambda={lam:>8.0f}  held-out R2_lin = {rr:.4f}  MAE = {mae_r:.2f}cp")
        if best is None or rr > best[0]:
            best = (rr, mae_r, lam)
    print(f"\n[G0.2] best R2_lin = {best[0]:.4f}  vs R2_full = {r2(t, preds):.4f}  "
          f"(ratio {best[0] / r2(t, preds):.2f})")


if __name__ == "__main__":
    main()
