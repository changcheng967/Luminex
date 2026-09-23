# The NNUE Developer Mindset (SF school) — distilled for Luminex
> Sources: fishtest culture, nnue-pytorch history, this session's cross-engine
> verification, and the SFNNv1→v16 arc. This is the operating system we run on.

## 1. Elo is measured, never argued
Every claim ends in an SPRT or LOS number. "Looks stronger" is not a sentence.
(corollary we already live by: pre-registered falsification clauses before
every experiment — decide what would kill the idea BEFORE running it)

## 2. The update path is a budget, not a footnote
SF's entire v10→v16 arc is feature-subtraction to afford width. Ask of every
new input feature: "what does it cost PER MOVE on target silicon?" before
"what does it know?" — we measured this discipline's worth directly (our
memory-bound triple-verification; threat-transition pre-experiment).

## 3. Data quality > architecture > parameters
The two biggest Elo jumps in SF history (NNUE adoption, Leela-data era) were
DATA events, not topology events. v12's +200 Elo was a data event too.
When stuck: audit the labels first, the loss second, the architecture last.

## 4. Subtraction is progress
Removing king-threats (v13-16), removing the small net (v16), removing HCE
(SF16) — each removal shipped Elo. Complexity must pay rent in measured Elo
or get evicted.

## 5. One variable at a time, through a gate
Fishtest tests single patches. Our G1-G5 ladder, Step 0 control experiment,
and approve/reject logs are the same ritual. Never train a new net with two
untested changes inside it unless the design literally requires coupling.

## 6. Steal shamelessly, attribute precisely, verify locally
SF is GPL and everything is readable — derivation is honor, copying without
understanding is waste. Everything borrowed gets re-benchmarked on OUR silicon
(we found our 512-wide outruns SF19's 1536 per node — nobody's numbers
transfer).

## 7. The community is the multiplier
Threat inputs diffused to 4+ engines in days. Publish, compare, adopt fast.
Feature ideas are not moats; verification discipline and data pipelines are.

## 8. Quantization is part of the architecture
SF learned this late (QAT in v16); we banked it early (int16 saturation math).
An architecture that only works in float32 doesn't work.

## 9. Speed is a floor, not a target — until it's the bottleneck
Know which regime you're in (our 1+0.01 crossover experiment) and optimize
the binding constraint only.
