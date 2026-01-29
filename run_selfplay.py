#!/usr/bin/env python3
"""Run self-play games between two Luminex instances."""
import subprocess
import time

def run_game(game_num):
    """Run a single self-play game."""
    log_file = f"game_{game_num}.txt"
    
    # Commands for each engine instance
    commands = [
        # Game setup
        ">Luminex(0): ucinewgame\n>Luminex(0): position startpos",
        # White move
        ">Luminex(0): isready",
        ">Luminex(0): go movetime 1000",
        # Black move
        ">Luminex(1): ucinewgame",
        # Get position from previous move
    ]
    
    with open(log_file, 'w') as log:
        log.write(f"=== Luminex v2.2.0 Self-Play Game {game_num} ===\n")
        log.write(f"Time control: 1000ms per move\n")
        log.write("="*50 + "\n\n")
    
    # Use cutechess-cli style approach with direct UCI
    cmd = f'cd /c/Users/chang/Downloads/Luminex/build && cat > game_{game_num}.txt << "ENDGAME"\n'
    return cmd

print("Self-play script created")
