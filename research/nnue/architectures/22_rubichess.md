# RubiChess (Andreas Matthies)
> Confidence: repo-level (source structure read) + CPW

## Architecture
- In-engine trainer (learn.cpp, NNUELEARN) — rare: training lives IN the engine repo
- 768-input style (piece-square only, no king indexing historically) -> accumulator
- Move encodings: own 16-bit format, plus SF-sfen + binpack importers (verified in source)

## Signature Innovations
- Self-contained training; European amateur-friendly pipeline
- Good documenter — learn.cpp comments explain trade-offs plainly
