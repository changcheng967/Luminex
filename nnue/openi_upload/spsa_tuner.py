#!/usr/bin/env python3
"""SPSA tuner — game-based optimization of the 12 Luminex search constants.

Self-contained pod-bench edition (no cutechess, no anchor opponents):
  - Built-in UCI match runner (python-chess adjudication, subprocess pipes):
    no cutechess dependency, no Qt builds.
  - Self-play perturbation pairs: A(+delta) vs A(-delta) Luminex instances,
    colors alternated — SPSA gradient needs no external anchor engine.
  - Fixed-node games via the engine-native NodesPerMove UCI option
    (node-limited games cannot time out — the v1 lesson).
  - Checkpoint after every iteration: survives pod recycling.

Inherited from the original fishtest-era tuner: normalized [0,1] optimization space, per-param INTEGER
perturbation with min +/-1, step clamping (noise guard), iterate averaging
(theta_bar is the validated estimator, not noisy theta).

Usage:
  python3 spsa_tuner.py --engine ./luminex --net net.nnue \
                        --openings openings.epd \
                        [--iterations 300] [--games 120] [--concurrency 12] \
                        [--nodes 100000] [--ckpt spsa5.ckpt]
"""
import os, sys, subprocess, random, math, time, argparse, threading, queue

# (name, default, lo, hi) — ORDER MUST MATCH SPSAParams::load() in spsa_params.h
PARAMS = [
    ("lmr_scale_quiet",   40,  20,  70),
    ("lmr_scale_noisy",   24,  10,  50),
    ("futility_coeff",   130,  80, 200),
    ("futility_offset",    50,   0, 120),
    ("nmp_base",            3,   2,   6),
    ("nmp_thresh1",         5,   3,   8),
    ("nmp_thresh2",        12,   8,  18),
    ("razor_base",        300, 150, 500),
    ("razor_coeff",        60,  30, 120),
    ("rev_fut_coeff",     100,  60, 160),
    ("aspiration_delta",    50,  20, 100),
    ("singular_margin",   200, 100, 400),
]
NP_ = len(PARAMS)

try:
    import chess
except ImportError:
    print("pip install python-chess", file=sys.stderr); sys.exit(1)


def write_params(path, values):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v}\n")


class UCIEngine:
    def __init__(self, path, workdir, net, nodes):
        self.p = subprocess.Popen(
            [path], cwd=workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.until("uciok")
        self.send(f"setoption name UseNNUE value true")
        self.send(f"setoption name NNUEFile value {net}")
        self.send(f"setoption name NodesPerMove value {nodes}")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 16")
        self.send("isready")
        self.until("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")

    def until(self, tag):
        for _ in range(200000):
            line = self.p.stdout.readline()
            if not line:
                return False
            if line.startswith(tag):
                return True
        return False

    def bestmove(self, position_cmd):
        self.send(position_cmd)
        self.send("go")
        for _ in range(200000):
            line = self.p.stdout.readline()
            if not line:
                return None
            if line.startswith("bestmove "):
                return line.split()[1]
        return None

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


MAX_PLY = 400


def play_game(eng_a, eng_b, fen):
    """1 = A wins, 0 = draw, -1 = B wins. python-chess adjudicates game end."""
    board = chess.Board(fen)
    moves = []
    turn_a = board.turn == chess.WHITE
    while board.ply() < MAX_PLY:
        if board.is_game_over():
            r = board.result()
            if r == "1-0":
                return 1 if turn_a else -1
            if r == "0-1":
                return -1 if turn_a else 1
            return 0
        eng = eng_a if board.turn == chess.WHITE else eng_b
        mv = eng.bestmove("position fen " + fen + " moves " + " ".join(moves))
        if mv is None:
            return -1 if board.turn == chess.WHITE else 1   # crash = loss
        try:
            board.push_uci(mv)
        except ValueError:
            return -1 if board.turn == chess.WHITE else 1   # illegal = loss
        moves.append(mv)
    return 0


def run_pairs(engine, dir_a, dir_b, net, nodes, pairs, concurrency, rng_seed):
    """Play `pairs` color-alternating (+,-) games; return mean score for + side."""
    fens = load_fens()
    rng = random.Random(rng_seed)
    jobs = queue.Queue()
    for _ in range(pairs):
        jobs.put(rng.choice(fens))
    score = [0]

    def worker():
        a = b = None
        while True:
            try:
                fen = jobs.get_nowait()
            except queue.Empty:
                break
            if a is None:
                a = UCIEngine(engine, dir_a, net, nodes)
                b = UCIEngine(engine, dir_b, net, nodes)
            s1 = play_game(a, b, fen)          # A(+) as white
            s2 = -play_game(b, a, fen)         # A(+) as black
            score[0] += s1 + s2
        if a: a.quit()
        if b: b.quit()

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    return score[0] / (2 * pairs)


_FENS = None
def load_fens():
    global _FENS
    if _FENS is None:
        _FENS = [l.split(";")[0].strip() for l in open(ARGS.openings)
                 if len(l.strip()) > 10]
    return _FENS


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--openings", required=True)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=100000)
    ap.add_argument("--ckpt", default="spsa.ckpt")
    args = ap.parse_args()
    ARGS = args

    engine = os.path.abspath(args.engine)
    net = os.path.abspath(args.net)

    theta = [p[1] for p in PARAMS]
    theta_bar = list(theta)
    start_iter = 0
    rng_state = 0x5EED
    if os.path.exists(args.ckpt):
        with open(args.ckpt) as f:
            vals = [int(x) for x in f.read().split()]
        if len(vals) == 2 + 2 * NP_:
            start_iter, rng_state = vals[0], vals[1]
            theta = vals[2:2 + NP_]
            theta_bar = vals[2 + NP_:]
            print(f"[resume] iteration {start_iter}", flush=True)

    rng = random.Random(rng_state)
    for d in ("spsa_side_A", "spsa_side_B"):
        os.makedirs(d, exist_ok=True)

    pairs = args.games // 2
    print(f"spsa: {args.iterations} iters x {args.games} games, "
          f"conc={args.concurrency}, nodes={args.nodes}", flush=True)

    for it in range(start_iter, args.iterations):
        k = it + 1
        dp = []
        for (_, _, lo, hi) in PARAMS:
            rng_range = hi - lo
            d = max(1, round(0.06 * rng_range))
            dp.append(d if rng.random() < 0.5 else -d)
        tp = [clamp(theta[i] + dp[i], PARAMS[i][2], PARAMS[i][3]) for i in range(NP_)]
        tm = [clamp(theta[i] - dp[i], PARAMS[i][2], PARAMS[i][3]) for i in range(NP_)]
        write_params("spsa_side_A/spsa_params.txt", tp)
        write_params("spsa_side_B/spsa_params.txt", tm)

        t0 = time.time()
        sp = run_pairs(engine, "spsa_side_A", "spsa_side_B", net, args.nodes,
                       pairs, args.concurrency, rng.randrange(1 << 30))
        a_k = 0.02 / (k ** 0.602)
        for i in range(NP_):
            rng_range = PARAMS[i][3] - PARAMS[i][2]
            g = sp / (2.0 * dp[i] / rng_range + 1e-9)
            step = clamp(a_k * g * rng_range, -0.05 * rng_range, 0.05 * rng_range)
            theta[i] = clamp(round(theta[i] + step), PARAMS[i][2], PARAMS[i][3])
            theta_bar[i] = round((theta_bar[i] * k + theta[i]) / (k + 1))

        print(f"iter {k}/{args.iterations} score(+,-)={100*sp:+.1f}% "
              f"({time.time()-t0:.0f}s) theta: {' '.join(map(str, theta))}",
              flush=True)
        with open(args.ckpt, "w") as f:
            f.write(" ".join(map(str, [it + 1, rng.randrange(1 << 30)]
                                 + theta + theta_bar)) + "\n")
        write_params("spsa_params_best.txt", theta_bar)

    print("DONE. theta_bar in spsa_params_best.txt:")
    for i, (name, d, _, _) in enumerate(PARAMS):
        print(f"  {name:18s} {theta_bar[i]:4d} (default {d})")


if __name__ == "__main__":
    main()
