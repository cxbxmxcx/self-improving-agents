# Superseded: chapter 7 as a sweep of learning regimes

This was chapter 7's first direction, retired on 2026-11-10 when the chapter turned to searching one fixed code artifact, the learning component. It asked whether chapter 6's learning mechanisms leave room to improve when they get less experience, a cheaper writer, and fewer writer calls. The listing is kept because it replays chapter 6's mechanisms from source, which is how the framework's seed mechanisms are checked.

The mechanisms it runs stay in [chapters/ch07/mechanisms/](../../chapters/ch07/mechanisms/). Each one keeps its writer prompts as constants and defines one entry point, `async def learn(lab)`, and the [mechanism runtime](../../helix/memory/mechanism_runtime.py) runs it with a metered lab as its only capability. `tests/test_ch07_seed_mechanisms.py` checks that these sources learn exactly what chapter 6's classes learned.

## Run the listing

Run from the repository root in the project environment. Replay serves saved model replies and needs no API key.

```powershell
python chapter_appendices/ch07_learning_regime_legacy/01_learning_regime.py
```

| Listing | Source | What it shows |
|---|---|---|
| 7.1 (retired) | [01_learning_regime.py](01_learning_regime.py) | Chapter 6's mechanisms under harder learning regimes, with validation match, writer calls, and dollars per class, and whether each regime leaves room to improve |

## What the recording holds

The first regime, sixteen low-effort runs with the writer at medium, replays in full for both task classes. Three mechanisms still match all sixteen validation cases on each class, so that regime leaves no room to improve. The second regime lowers the writer to low and replays four mechanisms on the first class before it reaches a call the recording lacks.

Recording the rest needs `--record`, which makes paid calls to gpt-5.4-mini. The sweep was unfinished when the chapter changed direction, and the missing replies were not recorded afterward.
