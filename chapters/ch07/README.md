# Chapter 7: Searching the harness for better learning mechanisms

Chapter 7 continues chapter 6 by putting the harness's learning code under search. Chapter 6 selected among five learning mechanisms that a person wrote; here a search writes new ones, and a changed harness is measured by how well it learns procedures for task classes it never saw.

## Learning mechanisms as source

The search can only rewrite what it can read, so each of chapter 6's mechanisms is held here as source in [mechanisms/](mechanisms/). A mechanism keeps its writer prompts as constants and defines one entry point, `async def learn(lab)`, and the [mechanism runtime](../../helix/memory/mechanism_runtime.py) runs it with a metered lab as its only capability. The lab meters writer calls, practice calls, and time, and a guard refuses imports, private names, bare `except`, and `str.format` before anything runs; like chapter 6's code-skill guard, it is not a security sandbox.

Replayed from chapter 6's recordings, the seeds learn exactly what chapter 6's classes learned: the same prompts, writer calls, functions, and validation matches. `tests/test_ch07_seed_mechanisms.py` checks this without an API key, so the search starts exactly where chapter 6 ended.

## Code listings

Use the project environment described in the [repository setup](../../README.md#setup), then run the commands below from the repository root. Replay serves saved model replies and needs no API key; `--record` adds paid calls for replies the recording lacks.

```powershell
python chapters/ch07/01_learning_regime.py
```

| Listing | Source | What to inspect |
|---|---|---|
| 7.1 | [01_learning_regime.py](01_learning_regime.py) | Chapter 6's mechanisms under harder learning regimes, with validation match, writer calls, and dollars per class, and whether each regime leaves room to improve |
