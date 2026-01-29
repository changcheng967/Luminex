#!/bin/bash
cd build

echo "=== Luminex v2.2.0 Self-Play Game 3 (e4 opening) ===" > ../game_log3.txt

pos="startpos"
moves=0
max_moves=40

while [ $moves -lt $max_moves ]; do
    # First move - try e4 first
    if [ $moves -eq 0 ]; then
        pos="startpos moves e2e4"
        moves=1
        echo "1. e4 (forced for variety)"
    fi
    
    # Black response
    output=$(echo -e "position $pos\ngo movetime 500\nquit" | ./luminex.exe 2>&1)
    bestmove=$(echo "$output" | grep "bestmove" | tail -1 | awk '{print $2}')
    echo "$moves. ... $bestmove"
    pos="$pos $bestmove"
    moves=$((moves + 1))
    
    if [ $moves -ge $max_moves ]; then
        break
    fi
    
    # White
    output=$(echo -e "position $pos\ngo movetime 500\nquit" | ./luminex.exe 2>&1)
    bestmove=$(echo "$output" | grep "bestmove" | tail -1 | awk '{print $2}')
    echo "$moves. $bestmove ..."
    pos="$pos $bestmove"
    moves=$((moves + 1))
    
    if echo "$output" | grep -q "mate"; then
        echo "Mate!"
        break
    fi
done

echo "" >> ../game_log3.txt
echo "Game complete"
