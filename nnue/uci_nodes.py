"""Fixed-depth searches through UCI: for each position, the last 'info depth' line's nodes and score, and
the best move, plus any 'info string NNUE' lines. Short by design (laptop): one thread, 16 MB hash.
usage: uci_nodes.py ENGINE NET DEPTH [Option=Value ...]"""
import subprocess, sys

POS = ["startpos",
       "fen r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
       "fen 2r3k1/pp3ppp/4p3/3pP3/1q1P4/1P2Q3/P4PPP/2R3K1 w - - 0 25"]
eng, net, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
opts = [o.split("=", 1) for o in sys.argv[4:]]
p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def send(s):
    p.stdin.write(s + "\n")
    p.stdin.flush()


def until(tok):
    out = []
    for line in p.stdout:
        line = line.rstrip()
        out.append(line)
        if line.startswith(tok):
            return out
    return out


send("uci"); until("uciok")
for k, v in [("Threads", "1"), ("Hash", "16"), ("NNUEFile", net)] + opts:
    send(f"setoption name {k} value {v}")
send("isready")
for line in until("readyok"):
    if "NNUE" in line or "heads" in line:
        print("  " + line)
for fen in POS:
    send("ucinewgame"); send("isready"); until("readyok")
    send(f"position {fen}")
    send(f"go depth {depth}")
    lines = until("bestmove")
    info = [l for l in lines if l.startswith("info depth") and " nodes " in l]
    last = info[-1].split() if info else []
    nodes = last[last.index("nodes") + 1] if "nodes" in last else "?"
    score = " ".join(last[last.index("score") + 1:last.index("score") + 3]) if "score" in last else "?"
    print(f"  {fen[:40]:40s} depth {depth}: nodes {nodes}, score {score}, {lines[-1]}")
send("quit")
p.wait(timeout=30)
