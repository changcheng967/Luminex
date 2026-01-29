#!/bin/bash
cd build

echo "=== Luminex v2.2.0 Self-Play Game 2 ===" > ../game_log2.txt
echo "" >> ../game_log2.txt

pos="startpos"
moves=0
max_moves=50

while [ $moves -lt $max_moves ]; do
    echo "Move $moves (White)"
    
    output=$(echo -e "position $pos\ngo movetime 500\nquit" | ./luminex.exe 2>&1)
    echo "$output" >> ../game_log2.txt
    
    bestmove=$(echo "$output" | grep "bestmove" | tail -1 | awk '{print $2}')
    echo "White: $bestmove"
    
    if [ "$bestmove" = "(none)" ] || [ "$bestmove" = "" ]; then
        echo "Game over"
        break
    fi
    
    if [ "$pos" = "startpos" ]; then
        pos="startpos moves $bestmove"
    else
        pos="$pos $bestmove"
    fi
    
    moves=$((moves + 1))
    
    if [ $moves -ge $max_moves ]; then
        break
    fi
    
    # Black
    echo "Move $moves (Black)"
    echo "$output" >> ../game_log2.txt
    
    output=$(echo -e "position $pos\ngo movetime 500\nquit" | ./luminex.exe 2>&1)
    echo "$output" >> ../game_log2.txt
    
    bestmove=$(echo "$output" | grep "bestmove" | tail -1 | awk '{print $2}')
    echo "Black: $bestmove"
    
    if [ "$bestmove" = "(none)" ] || [ "$bestmove" = "" ]; then
        echo "Game over"
        break
    fi
    
    pos="$pos $bestmove"
    moves=$((moves + 1))
    
    # Check for mate
    if echo "$output" | grep -q "mate 1"; then
        echo "Mate detected!"
        break
    fi
    
    echo ""
done

echo "" >> ../game_log2.txt
echo "=== Game Complete ===" >> ../game_log2.txt
echo "Total moves: $moves" >> ../game_log2.txt

echo "=== Game 2 PGN ==="
grep "bestmove" ../game_log2.txt | awk '{print NR ":" $2}' | head -50
