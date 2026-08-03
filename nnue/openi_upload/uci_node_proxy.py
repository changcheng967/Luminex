#!/usr/bin/env python3
"""UCI proxy: intercepts cutechess's `go wtime...` and replaces with `go nodes N`.
Makes ANY UCI engine play fixed-node games — no clock forfeits possible,
depth decoupled from wall-clock speed. Enables full concurrency for SPSA.

Usage: as the engine cmd in cutechess:
  uci_node_proxy.py <N> <real_engine_path>
"""
import sys, subprocess, threading

N = sys.argv[1]
engine = sys.argv[2:]
proc = subprocess.Popen(engine, stdin=subprocess.PIPE, stdout=sys.stdout,
                        stderr=sys.stderr, text=True, bufsize=1)

def fwd():
    for line in sys.stdin:
        if line.startswith('go ') and 'nodes' not in line:
            line = f'go nodes {N}\n'
        proc.stdin.write(line)
        proc.stdin.flush()

t = threading.Thread(target=fwd, daemon=True)
t.start()
proc.wait()
