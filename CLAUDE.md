# Luminex Chess Engine

## HARD RULES — VIOLATION = REVERT AND START OVER
- `git config user.name "changcheng967"` before ANY commit
- ONE lowercase commit message, NO co-authors, NO AI attribution
- ONE logical change per commit. Run perft between each.
- COMMIT and PUSH after EVERY change, regardless of outcome (debug, test, exploratory). GitHub remote must ALWAYS be up to date with local.
- NEVER write to std::cerr anywhere in the codebase. On Windows, CuteChess merges stderr into stdout, corrupting the UCI protocol stream. All diagnostics go to file only (std::ofstream, std::ios::app).
- NEVER commit debug logging. Remove ALL file-based logging before committing.
- Build Release mode: cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja && cmake --build build

## PERFT — GROUND TRUTH (run after EVERY change, no exceptions)
```
build/luminex.exe bench
```
Expected values — if ANY number is wrong, your change broke something. Revert immediately.
```
Startpos:   depth 1=20  depth 2=400  depth 3=8902  depth 4=197281  depth 5=4865609
Kiwipete:   fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -
            depth 1=48  depth 2=2039  depth 3=97862  depth 4=4085603
Position 3: fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
            depth 1=14  depth 2=191  depth 3=2812  depth 4=43238  depth 5=674624
Position 4: fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -
            depth 5=15833292
Position 5: fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -
            depth 4=2103487
```
To debug a wrong perft: run perft divide at failing depth, compare each root move count against Stockfish (`go perft <depth>`). First mismatch isolates the bug. Recurse into that subtree until you find the exact failing position and move type.

## UCI PROTOCOL (non-negotiable)
- All scoring: side-to-move perspective. Positive = side to move is winning.
- bestmove must always include a move: use `bestmove 0000` for terminal positions, never bare `bestmove`.
- Flush stdout after every output line: `std::cout.flush()` or `"\n"` with `std::cout.flush()`.
- Engine receives complete position with every `position` command. No incremental state — parse from scratch each time.
- The `position startpos moves e2e4 e7e5 ...` command replays ALL moves. If any move fails to match GEN_LEGAL, the remaining moves are lost and the position is wrong. This is the #1 source of illegal-move bugs.

## ESCALATION — WHEN STUCK, STOP
If any of these are true:
- 3+ diagnostic attempts without finding root cause
- Going in circles (fix A → new symptom → fix B → original symptom returns)
- Bug defies understanding (code looks correct but behavior is wrong)
- 30+ minutes on one issue without progress

Then STOP. Do not guess. Do not try "one more thing."
1. Commit and push everything to GitHub
2. Write a summary: exact symptom with evidence, what you tried, current hypothesis, relevant file:line, commit hash
3. Wait for Opus 4.6 guidance

## THE CYCLE
### 1. MEASURE — data only, no opinions
Run test games. Record: score, illegal moves, stalls, depth reached, NPS.
```bash
# Quick (5 games, ~30s):
"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe" -rounds 5 -engine cmd="build/luminex.exe" name=Luminex -engine cmd="C:\Users\chang\Downloads\fruit_21\fruit_21\fruit_21.exe" name=Fruit -each proto=uci tc=1+0.1 -recover -pgnout test.pgn
# Full (20 games):
# same command with -rounds 20
```

### 2. THINK — most important step, do not skip
- What is the SINGLE biggest problem? Rank by: crashes > illegal moves > search depth > move ordering > eval quality > pruning > speed > endgame > tuning.
- WHY does it exist? Trace to root cause, not surface symptom.
- What would Stockfish do? What does chessprogramming.org say?
- What is the SIMPLEST fix that addresses the root cause?
- What could go wrong? What side effects?

If not confident: add temporary file-based logging, run an experiment, read more code. Understanding > coding.

### 3. IMPLEMENT — smallest possible change
- One commit, one logical change
- If writing >50 lines, stop and simplify
- If you don't understand every line you're writing, think more

### 4. VERIFY
- `ninja -C build` (must compile clean with -Werror)
- `build/luminex.exe bench` (ALL perft values must match)
- If perft fails: `git checkout -- src/` immediately. Your change is wrong.
- If perft passes: run 5 games vs Fruit
- If regression: `git revert HEAD`, return to step 2

### 5. LOG
Update PROGRESS.md with: date, commit, score, illegal moves, depth, NPS, what changed.

## PRIORITY ORDER
1. **Crashes/illegal moves** — engine must finish every game
2. **Board correctness** — perft must be perfect. If wrong, nothing else matters.
3. **Search depth** — deeper = stronger. If depth < 6 at tc=1+0.1, something is broken.
4. **Move ordering** — TT move → captures (MVV-LVA) → killers → history
5. **Tactical strength** — qsearch, SEE, check extensions
6. **Evaluation** — PSTs, mobility, pawn structure, king safety
7. **Pruning** — LMR, null move, futility
8. **Speed** — NPS > 200k. Profile if slow.
9. **Endgame** — tapered eval, king centralization, passed pawns
10. **Tuning** — only when everything else is solid

## DEBUGGING PATTERNS

### Wrong-color move from GUI:
Position replay in handle_position broke mid-sequence. Side-to-move is wrong.
→ Add file logging to handle_position: log every move parsed, every match result, FEN after each do_move.
→ Add file logging to handle_go: log FEN and side-to-move before search.
→ Run one game. Read the log. First wrong FEN = the bug.

### Move passes our GEN_LEGAL but GUI rejects it:
Board state drifted from reality. do_move/undo_move asymmetry.
→ Check EVERY move type: quiet, capture, en passant, castling (K+Q for both colors), promotion (4 types), capture-promotion (4 types), double pawn push.
→ Verify undo_move restores: Zobrist key, castling_rights_[], piece_list[], index[], bitboards, king_square[], ep_square.

### TT returning illegal move:
Hash collision. TT move from different position passes pos.legal() by coincidence.
→ Validate TT moves against full GEN_LEGAL list, not just pos.legal().

### Search hangs:
→ Check qsearch depth limit (must have hard floor, e.g., depth < -8 returns eval)
→ Check check_time() is being called (every 256 nodes minimum)
→ Check stop flag is readable during search (PeekNamedPipe on Windows)

## NEVER
- NNUE, neural nets, opening books, endgame tablebases, external dependencies
- Guess at fixes without evidence
- Change multiple things between tests
- Skip perft after code changes
- Ignore regressions — revert immediately
- Write to std::cerr (use file logging only, remove before commit)

## KEY FILES
- `src/board.cpp` — Position, do_move, undo_move, legal, Zobrist
- `src/movegen.cpp` — move generation (generate<GEN_LEGAL> etc.)
- `src/search.cpp` — alpha-beta, qsearch, time management
- `src/evaluation.cpp` — eval function, PSTs
- `src/uci.cpp` — UCI protocol, handle_position, handle_go
- `src/transposition.cpp` — TT
- `src/types.h` — Move encoding (16-bit), piece/square/flag constants
- `src/bitboard.h` — bitboard utilities, attack tables

## ARCHITECTURE
- Square mapping: a1=0, h8=63
- Move encoding: 16-bit (6 from + 6 to + 4 flags)
- State: StateInfo stack inside Position (incremental, not copy-restore)
- Zobrist: incremental XOR updates in do_move, must be perfectly mirrored in undo_move

## ENVIRONMENT
- Intel Ultra 9 275HX, 64GB RAM, Windows 11
- Clang/LLVM C++23, CMake + Ninja
- CuteChess CLI for testing
- Opponent: Fruit 2.1

## REFERENCES
- https://www.chessprogramming.org/ — encyclopedia
- https://www.chessprogramming.org/Perft_Results — all perft test positions
- https://github.com/official-stockfish/Stockfish — reference implementation
- https://github.com/namebrandon/seajay-chess — Claude Code built engine (~2450 ELO)
- https://github.com/changcheng967/Douchess — our previous working engine (reference for architecture)
```

---

