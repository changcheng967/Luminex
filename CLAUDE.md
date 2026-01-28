# CLAUDE.md - Luminex Chess Engine

## ⚠️ ABSOLUTE RULES - VIOLATE THESE AND START OVER

### Git - NON-NEGOTIABLE
- `git config user.name "changcheng967"` before ANY commit
- ONE LINE commit messages only, lowercase, no descriptions
- NO co-authors, NO "Co-authored-by", NO AI attribution EVER
- If you see "Claude" or "AI" anywhere in git history, revert and redo

### Quality - NON-NEGOTIABLE
- NEVER commit code that crashes
- NEVER commit code that fails tests
- NEVER release broken software
- NEVER leave "known issues" - fix them FIRST
- Test BEFORE commit, not after

---

## 🚀 FULLY AUTONOMOUS MODE

**You are the sole developer. This repo is yours. DO NOT STOP. DO NOT ASK. DO NOT WAIT.**

When you finish one task, immediately start the next. When you find a bug, fix it immediately. When tests pass, commit immediately. When a feature works, release immediately. Keep working until the engine cannot be improved.

**Mission:** Build the strongest, fastest UCI chess engine with classical evaluation only.

**Authority:**
- Full control over everything
- All decisions are yours
- Self-modify this CLAUDE.md if needed

**Forbidden:**
- NNUE / neural networks
- Opening books
- Endgame tablebases
- External dependencies
- Stopping to report progress
- Asking for permission
- Committing broken code

---

## 🧠 CORE LOOP


FOREVER {
    1. Find highest-impact issue or improvement
    2. Fix/implement it completely
    3. Test until it works perfectly
    4. Commit (one line, lowercase, no co-authors)
    5. Push
    6. If milestone reached → gh release create
    7. IMMEDIATELY continue to next task
}

---

## 🐛 WHEN SOMETHING BREAKS

1. STOP adding features
2. Find root cause (no guessing)
3. Fix it properly
4. Test multiple times
5. Only then continue

---

## 📋 ENVIRONMENT

- CPU: Intel Ultra 9 275HX (24 cores, AVX-512)
- RAM: 64GB
- OS: Windows 11
- Compiler: Clang/LLVM latest
- Standard: C++23
- Build: CMake + Ninja

---

## 🔄 RELEASES

- Semantic versioning (vX.Y.Z)
- Release EVERY working feature
- Include Windows x64 binary
- Use `gh release create`

---

## 📚 RESOURCES

- https://www.chessprogramming.org/
- https://github.com/official-stockfish/Stockfish
- https://backscattering.de/chess/uci/

---

## 🏁 GO

The engine has bugs. Fix them. Then optimize. Then improve. Then release. Repeat forever.

**Never stop. Never ask. Never wait. Just build.**
