#!/bin/bash
cd build

echo "=== Luminex v2.2.0 Self-Play Analysis ===" > ../game_analysis.txt
echo "Date: $(date)" >> ../game_analysis.txt
echo "================================" >> ../game_analysis.txt
echo "" >> ../game_analysis.txt

# Test position 1: Starting position
echo "--- Test 1: Starting Position ---" >> ../game_analysis.txt
echo "position startpos" >> ../game_analysis.txt
echo "go movetime 500" >> ../game_analysis.txt
./luminex.exe << 'INPUT' > /dev/null 2>&1 &
position startpos
go movetime 500
quit
INPUT
sleep 1
echo "" >> ../game_analysis.txt

# Test position 2: After 1.e4 e5
echo "--- Test 2: After 1.e4 e5 ---" >> ../game_analysis.txt
echo "position startpos moves e2e4 e7e5" >> ../game_analysis.txt
echo "go movetime 500" >> ../game_analysis.txt
./luminex.exe << 'INPUT' 2>&1 | tee -a ../game_analysis.txt
position startpos moves e2e4 e7e5
go movetime 500
quit
INPUT

echo "Analysis complete. Check game_analysis.txt"
