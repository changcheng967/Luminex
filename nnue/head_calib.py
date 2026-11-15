"""Where the error head's estimate sits in real searches: fixed-node searches from varied positions with
the head computed but almost unused (HeadK=1) and LUMINEX_HEAD_STATS=1, which prints the head's error in
cp (and |correction|) at every node where the margins read it, as quantiles over all searches so far.
usage: head_calib.py ENGINE NET NODES"""
import os, subprocess, sys

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "2r3k1/pp3ppp/4p3/3pP3/1q1P4/1P2Q3/P4PPP/2R3K1 w - - 0 25",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5",
    "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "r2q1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9",
    "8/8/1p3k2/p1p1r3/P1P5/1P2K3/4R3/8 w - - 0 40",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 30",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 w - - 0 13",
]
eng, net, nodes = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                     env=dict(os.environ, LUMINEX_HEAD_STATS="1"))


def send(s):
    p.stdin.write(s + "\n")
    p.stdin.flush()


def until(tok):
    out = []
    for line in p.stdout:
        out.append(line.rstrip())
        if line.startswith(tok):
            return out
    return out


send("uci"); until("uciok")
for k, v in (("Threads", "1"), ("Hash", "64"), ("NNUEFile", net), ("HeadK", "1")):
    send(f"setoption name {k} value {v}")
send("isready")
print("\n".join("  " + l for l in until("readyok") if "NNUE" in l))
last = ""
for fen in FENS:
    send("ucinewgame"); send("isready"); until("readyok")
    send(f"position fen {fen}")
    send(f"go nodes {nodes}")
    for l in until("bestmove"):
        if "HEADSTATS" in l:
            last = l
print("after", len(FENS), "searches of", nodes, "nodes:")
print("  " + last)
send("quit")
p.wait(timeout=30)
