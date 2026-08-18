#!/usr/bin/env python3
"""Ridge solve for HCE distillation, eigen-path form (numerically stable).

Normal equations from luminex-evaltrace --solve: A = X'X, b = X'r over rows
  r = y_white - tempo,  x_j = f_j*ph/24 (MG) or f_j*(24-ph)*sf/768 (EG),
in solver space (1214 = 607 MG + 607 EG single coefs). Ridge toward the
CURRENT engine coefs c0 with per-feature scaling d_j = sqrt(A_jj/N)
(standardized space; rare features pin to c0, dense features move freely):

  minimize ||Xc - r||^2 + lam * || D(c - c0) ||^2,   D = diag(d)

Substitute u = D(c - c0), whiten: (A~ + lam I)u = h with A~ = D^-1 A D^-1,
h = D^-1(b - A c0). Diagonalize A~ = V L V' ONCE, then the whole lam path is
closed-form:  u = V (V'h)_i / (lam + L_i),  c = c0 + D^-1 u,
and train SSE(lam) = SSE(c0) - sum_i beta_i^2 (2 lam + L_i)/(lam + L_i)^2.
A is collinear (PST sums vs material, EG tradedown vs material) so the
eigen form is mandatory: np.linalg.solve on A is garbage (non-monotone lam
path, 1e6 coefficients on near-null directions).

.bin layout: [int64 N][double A[1214*1214]][double b[1214]]
             [+ v2 tail: double sum_r, double sum_r2]

Usage:
  python solve_ridge.py <prefix> <c0_file> sweep          # lam grid + train path + gates
  python solve_ridge.py <prefix> <c0_file> <lam> [<out>]  # single solve + gates
Validate generalization: NNUE_SCORE_COEFS=<coefs> luminex-evaltrace --score < holdout
"""
import sys
import numpy as np

NPHASE = 607
NS = 2 * NPHASE


def load(prefix):
    with open(prefix + ".bin", "rb") as fh:
        n = int(np.fromfile(fh, dtype=np.int64, count=1)[0])
        A = np.fromfile(fh, dtype=np.float64, count=NS * NS).reshape(NS, NS)
        b = np.fromfile(fh, dtype=np.float64, count=NS)
        tail = np.fromfile(fh, dtype=np.float64, count=2)
        sum_r, sum_r2 = (float(tail[0]), float(tail[1])) if tail.size == 2 else (np.nan, np.nan)
    return n, A, b, sum_r, sum_r2


def load_coefs(path):
    with open(path) as fh:
        vals = [float(ln.split()[1]) for ln in fh if ln.strip()]
    assert len(vals) == NS, f"coefs file has {len(vals)} lines, want {NS}"
    return np.array(vals)


class RidgePath:
    def __init__(self, n, A, b, c0, sum_r, sum_r2):
        self.n, self.A, self.b, self.c0 = n, A, b, c0
        w = np.diag(A).copy()
        self.act = w > 1e-12
        na = int(self.act.sum())
        d = np.sqrt(w[self.act] / n)
        self.d = d
        Aa = A[np.ix_(self.act, self.act)]
        self.At = (Aa / d[:, None]) / d[None, :]          # whitened = D^-1 A D^-1
        self.At = 0.5 * (self.At + self.At.T)
        g = (b - A @ c0)[self.act] / d
        self.L, self.V = np.linalg.eigh(self.At)           # A~ = V L V'
        beta = self.V.T @ g
        self.beta2 = beta * beta
        self.g = g
        self.sse0 = c0 @ (A @ c0) - 2 * b @ c0 + sum_r2    # train SSE at c0
        self.sst = sum_r2 - sum_r * sum_r / n
        self.inactive = na

    def coefs(self, lam):
        c = self.c0.copy()
        beta = self.V.T @ self.g
        if lam > 0:
            u = self.V @ (beta / (lam + self.L))
        else:
            u = self.V @ np.where(self.L > 1e-9 * self.L.max(), beta / np.maximum(self.L, 1e-300), 0.0)
        c[self.act] = self.c0[self.act] + u / self.d
        return c

    def train_r2(self, lam):
        L, b2 = self.L, self.beta2
        sse = self.sse0 - np.sum(b2 * (2 * lam + L) / (lam + L) ** 2)
        return 1 - sse / self.sst, np.sqrt(max(sse, 0) / self.n)


def gates(c, c0):
    ok = True
    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  [GATE FAIL] {msg}")
    for pt, lo, hi, nm in [(0, 60, 140, "pawn"), (1, 250, 400, "knight"),
                           (2, 250, 400, "bishop"), (3, 420, 640, "rook"), (4, 750, 1100, "queen")]:
        mg = c[384 + pt]; eg = c[607 + 384 + pt]
        print(f"  {nm + ' value':<22} mg={mg:8.2f}  eg={eg:8.2f}")
        chk(lo <= mg <= hi and lo <= eg <= hi, f"{nm} value {mg:.0f}/{eg:.0f} outside [{lo},{hi}]")
    for base, n_m, nm in [(390, 9, "knight mobility"), (399, 14, "bishop mobility"),
                          (413, 15, "rook mobility"), (428, 28, "queen mobility")]:
        mg = c[base:base + n_m]; eg = c[607 + base:607 + base + n_m]
        mono_mg = np.all(np.diff(mg) > -25)
        print(f"  {nm:<22} mg [{mg[0]:.0f}..{mg[-1]:.0f}] eg [{eg[0]:.0f}..{eg[-1]:.0f}]"
              f"{'  (non-monotone)' if not mono_mg else ''}")
    return ok


def write_coefs(path, c):
    with open(path, "w") as fh:
        for j, v in enumerate(c):
            fh.write(f"{j} {v:.6f}\n")


def main():
    prefix, c0_path = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "sweep"
    n, A, b, sum_r, sum_r2 = load(prefix)
    c0 = load_coefs(c0_path)
    rp = RidgePath(n, A, b, c0, sum_r, sum_r2)
    have_tail = sum_r2 == sum_r2
    print(f"rows={n:,}  active={rp.inactive}/{NS}  eig[min,max]=[{rp.L.min():.3g},{rp.L.max():.3g}]")
    if have_tail:
        r0, rm0 = 1 - rp.sse0 / rp.sst, np.sqrt(rp.sse0 / n)
        print(f"base c0 train fit: R2={r0:.4f} RMSE={rm0:.2f}")
    if mode == "sweep":
        for lam in [1e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]:
            c = rp.coefs(lam)
            shift = np.abs(c - c0)
            extra = ""
            if have_tail:
                r2t, rmset = rp.train_r2(lam)
                extra = f"  trainR2={r2t:+.4f} RMSE={rmset:.1f}"
            print(f"lam={lam:>9g}  max|c-c0|={shift.max():9.1f}  mean|c-c0|={shift.mean():8.2f}{extra}")
            write_coefs(f"{prefix}.coefs.lam{lam:g}", c)
        print("wrote lam grid; validate: NNUE_SCORE_COEFS=<file> luminex-evaltrace --score < holdout")
    else:
        lam = float(mode)
        c = rp.coefs(lam)
        out = sys.argv[4] if len(sys.argv) > 4 else f"{prefix}.coefs"
        write_coefs(out, c)
        print(f"wrote {out} (lam={lam})")
    print("\nGates:")
    gates(c, c0)


if __name__ == "__main__":
    main()
