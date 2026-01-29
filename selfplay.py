#!/usr/bin/env python3
"""Run self-play games between two Luminex engine instances."""
import subprocess
import threading
import time
import sys

def read_output(process, label):
    """Read output from a process."""
    while True:
        try:
            line = process.stdout.readline()
            if not line:
                break
            if line.strip():
                print(f"<{label}: {line.strip()}")
        except:
            break

def play_game(movetime=1000):
    """Play one game between two Luminex instances."""
    # Start two engine instances
    engine_white = subprocess.Popen(
        ["./luminex.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd="build"
    )

    engine_black = subprocess.Popen(
        ["./luminex.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd="build"
    )

    # Initialize engines
    for engine, name in [(engine_white, "Luminex(0)"), (engine_black, "Luminex(1)")]:
        engine.stdin.write("uci\n")
        engine.stdin.write("isready\n")
        engine.stdin.write("ucinewgame\n")
        engine.stdin.flush()

    # Start readers
    white_thread = threading.Thread(target=read_output, args=(engine_white, "Luminex(0)"))
    black_thread = threading.Thread(target=read_output, args=(engine_black, "Luminex(1)"))
    white_thread.daemon = True
    black_thread.daemon = True
    white_thread.start()
    black_thread.start()

    # Play game
    pos = "startpos"
    move_count = 0
    max_moves = 200

    while move_count < max_moves:
        # White move
        engine_white.stdin.write(f"position {pos}\n")
        engine_white.stdin.write(f"go movetime {movetime}\n")
        engine_white.stdin.flush()

        time.sleep(movetime/1000 + 0.5)

        # Black move
        engine_black.stdin.write(f"position {pos}\n")
        engine_black.stdin.write(f"go movetime {movetime}\n")
        engine_black.stdin.flush()

        move_count += 2
        time.sleep(movetime/1000 + 0.5)

    # Cleanup
    engine_white.stdin.write("quit\n")
    engine_black.stdin.write("quit\n")
    engine_white.stdin.flush()
    engine_black.stdin.flush()

    engine_white.wait()
    engine_black.wait()

if __name__ == "__main__":
    print("Starting Luminex self-play...")
    print(f"Time control: {1000}ms per move")
    print("="*50)

    play_game()
