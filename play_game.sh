#!/bin/bash
cd build

echo "=== Luminex v2.2.0 Self-Play Game ===" > ../game_log.txt
echo "" >> ../game_log.txt

pos="startpos"
moves=0
max_moves=50

while [ $moves -lt $max_moves ]; do
    echo "Move $moves (White to play)"
    echo "---" >> ../game_log.txt
    echo "Position: $pos" >> ../game_log.txt
    
    # White move
    output=$(echo -e "position $pos\ngo movetime 500\nquit" | ./luminex.exe 2>&1)
    echo "$output" >> ../game_log.txt
    
    bestmove=$(echo "$output" | grep "bestmove" | tail -1 | awk '{print $2}')
    echo "White plays: $bestmove"
    
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
    
    # Black move
    echo "Move $moves (Black to play)"
    echo "---" >> ../game_log.txt
    echo "Position: $pos" >> ../game_log.txt
    
    output=$(echo -e "position $pos\ngo movetime 500\nquit" | ./luminex.exe 2>&1)
    echo "$output" >> ../game_log.txt
    
    bestmove=$(echo "$output" | grep "bestmove" | tail -1 | awk '{print $2}')
    echo "Black plays: $bestmove"
    
    if [ "$bestmove" = "(none)" ] || [ "$bestmove" = "" ]; then
        echo "Game over"
        break
    fi
    
    pos="$pos $bestmove"
    moves=$((moves + 1))
    
    echo ""
done

echo "" >> ../game_log.txt
echo "=== Game Complete ===" >> ../game_log.txt
echo "Total moves: $moves" >> ../game_log.txt

cat ../game_log.txt
