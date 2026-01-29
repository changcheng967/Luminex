@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0build"

echo === Luminex v2.2.0 Self-Play === > ..\game_log.txt
echo Time: %date% %time% >> ..\game_log.txt
echo ================================ >> ..\game_log.txt

set "pos=startpos"
set /a moves=0
set /a maxmoves=60

:loop
if %moves% geq %maxmoves% goto end

echo %pos% | findstr /C:"moves" >nul
if errorlevel 1 (
    set "cmd=position %pos%"
) else (
    set "cmd=position startpos moves"
)

echo White thinking...
echo. >> ..\game_log.txt
echo === Move %moves% White === >> ..\game_log.txt
(
    echo %cmd%
    echo go movetime 500
    echo quit
) | luminex.exe 2>&1 > ..\temp_out.txt

findstr "bestmove" ..\temp_out.txt >> ..\game_log.txt
for /f "tokens=2" %%a in ('findstr "bestmove" ..\temp_out.txt') do (
    set "bestmove=%%a"
    echo Move %moves%: %%a
    if "!pos!"=="startpos" (
        set "pos=startpos moves %%a"
    ) else (
        set "pos=!pos! %%a"
    )
    set /a moves+=1
    goto black
)
goto end

:black
if %moves% geq %maxmoves% goto end

echo Black thinking...
echo. >> ..\game_log.txt
echo === Move %moves% Black === >> ..\game_log.txt
(
    echo position !pos!
    echo go movetime 500
    echo quit
) | luminex.exe 2>&1 > ..\temp_out.txt

findstr "bestmove" ..\temp_out.txt >> ..\game_log.txt
for /f "tokens=2" %%a in ('findstr "bestmove" ..\temp_out.txt') do (
    set "bestmove=%%a"
    echo Move %moves%: %%a
    set "pos=!pos! %%a"
    set /a moves+=1
)

goto loop

:end
echo. >> ..\game_log.txt
echo Game complete after %moves% moves. >> ..\game_log.txt
del ..\temp_out.txt
type ..\game_log.txt
echo Game complete!
