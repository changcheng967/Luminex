# Yu Nasu's Original NNUE (Shogi, 2018)
> Author: Yu Nasu | Inspired by: Kunihito Hoki's Bonanza king-indexed PST | Confidence: CPW + Nasu's paper

## Feature Set
- Shogi piece-square features indexed by own king square (the "HalfK" idea origin)

## Architecture Graph
```
sparse binary inputs -> [1x1 sparse layer = ACCUMULATOR, incrementally updated]
                     -> small dense hidden -> 1 output (win rate)
```

## Incremental Update Design
- THE foundational insight: layer-0 outputs are LINEAR in features, so
  make/unmake only add/subtract the moved piece's weight rows. Born incremental.

## Signature Innovations
- The entire "efficiently updatable" paradigm. Everything else in this folder
  is a descendant.

## Deployment
- YaneuraOu / Kristallweizen (Shogi SF derivatives) ~AlphaZero-level Shogi (2018-19)
- Nodchip (Hisayori Noda) ported to Stockfish 10 in 2019 -> SF12's +80 Elo leap
