# CLAUDE.md - Luminex Chess Engine

## ⚠️ ABSOLUTE RULES - VIOLATE THESE AND START OVER

### Git - NON-NEGOTIABLE
- `git config user.name "changcheng967"` before ANY commit
- ONE LINE commit messages only, lowercase, no descriptions
- NO co-authors, NO "Co-authored-by", NO AI attribution EVER
- **COMMIT ON EVERY CODE CHANGE** - save all work immediately
- **PUSH AFTER EVERY COMMIT** - `git push origin main`

### Quality - NON-NEGOTIABLE
- NEVER release broken software (commits are for development, releases are for users)

### Documentation - NON-NEGOTIABLE
- **UPDATE VERSION** in `src/luminex.h` when releasing (match GitHub releases)
- **UPDATE README.md** frequently with latest changes, features, progress

---

## 🚀 DEVELOPMENT MODE

**You are the implementation engine. The user is the architect.**

**Mission:** Beat Fruit 2.1 in a 20-game match (score 10+/20).

**Workflow:**
1. User provides detailed analysis and implementation plan
2. Claude implements the changes exactly as specified
3. Build and test
4. **STOP after EVERY test — provide summary, WAIT for instructions**

**Forbidden:**
- NNUE / neural networks
- Opening books / endgame tablebases
- External dependencies
- Making changes without user approval

---

## 📊 PROGRESS TRACKING

**Maintain `PROGRESS.md` in repo root:**

```markdown
| Version | Illegal Move Rate | Score vs Fruit | What Changed | What I Learned |
|---------|------------------|----------------|--------------|----------------|
```

**Update PROGRESS.md:**
- After EVERY test run
- After ANY significant code change
- Keep detailed notes on what worked/failed

---

## 🎮 TESTING

### Time Management (FAST feedback - complete in ~1 minute)
- **Quick Test**: `tc=1+0.1` - 5 games in ~30 seconds
- **Full Test**: `tc=1+0.1` - 20 games in ~2 minutes
- **Goal**: Fast iteration, NOT accurate strength measurement

### Quick Test (fast iteration)
```bash
"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe" -rounds 5 -engine cmd="build/luminex.exe" name=Luminex -engine cmd="C:\Users\chang\Downloads\fruit_21\fruit_21\fruit_21.exe" name=Fruit -each proto=uci tc=1+0.1 -recover
```

### Full Test (validation)
```bash
"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe" -rounds 20 -engine cmd="build/luminex.exe" name=Luminex -engine cmd="C:\Users\chang\Downloads\fruit_21\fruit_21\fruit_21.exe" name=Fruit -each proto=uci tc=1+0.1 -recover
```

### ⚠️ CRITICAL: Stop After Every Test
**After completing ANY test (quick or full), you MUST:**
1. Provide a session summary with results
2. WAIT for user analysis and instructions
3. DO NOT proceed with further changes until directed

### Score Interpretation
| Score | Meaning |
|-------|---------|
| 0-3/20 | Major weakness, fix fundamentals |
| 4-7/20 | Getting closer, tune parameters |
| 8-10/20 | Competitive, find small edges |
| 11+/20 | **GOAL ACHIEVED** |

---

## 🔍 WHEN ANALYZING, ASK

- How are we losing? (checkmate, time, illegal move, positional)
- What phase? (opening, middlegame, endgame)
- What depth do we reach vs Fruit?
- Is eval sane from equal positions?

---

## 🛠️ QUICK REFERENCE

### Build
```bash
ninja -C build
```

### Test Position
```bash
echo -e "uci\nisready\nposition startpos\ngo depth 10\nquit" | build/luminex.exe
```

### Compare to Fruit
```bash
echo -e "uci\nposition startpos moves e2e4 e7e5\ngo depth 12\nquit" | build/luminex.exe
echo -e "uci\nposition startpos moves e2e4 e7e5\ngo depth 12\nquit" | "C:\Users\chang\Downloads\fruit_21\fruit_21\fruit_21.exe"
```

---

## 📋 ENVIRONMENT

- CPU: Intel Ultra 9 275HX (24 cores, AVX-512)
- RAM: 64GB
- OS: Windows 11
- Compiler: Clang/LLVM, C++23
- Build: CMake + Ninja

---

## 📚 RESOURCES

- https://www.chessprogramming.org/
- https://github.com/official-stockfish/Stockfish
- https://backscattering.de/chess/uci/

---

## 🏁 PROTOCOL

**After EVERY test, STOP and report:**
1. Test results (score, illegal moves, crashes, timeouts)
2. Any anomalies or unexpected behavior
3. WAIT for user analysis and next steps

**DO NOT proceed with changes until directed.**