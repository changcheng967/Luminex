@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0build"

echo Running Luminex self-play games...

:: Game 1
echo === Game 1 === > ../game_1.txt
echo. >> ../game_1.txt
luminex.exe < ../game_1_commands.txt >> ../game_1.txt 2>&1

echo Game 1 complete. Check game_1.txt
