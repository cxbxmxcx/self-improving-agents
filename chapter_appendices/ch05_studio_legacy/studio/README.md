# Second Breakfast Games support desk

This is the fictional studio helpdesk used for the practical demonstrations in an earlier version of Chapter 5. Its tickets cover asset sync, audio, builds, and permissions, with user identities for testing scoped recall. The chapter uses the task's inputs, tool results, and memory records to demonstrate the implementation.

## Start with the walkthrough

From the repository root, run:

```powershell
python chapters/ch05/03_memory_learning.py
```

The default run uses a scripted responder with the real Helix agent loop, tools, memory hooks, archive, and value learner. Runbook search and episodic retrieval use the fixed `studio-token-hash-1024-v1` lexical encoder, so no API key, model download, or paper index is needed. The script creates a fresh directory under `chapters/ch05/runs/studio/` containing SQLite stores, `trace.json`, and `transcript.md`.

Use `--output PATH` to choose a new or empty directory. Existing nonempty output directories are rejected to preserve evidence. The [recorded trace](recorded/trace.json) and [transcript](recorded/transcript.md) show the expected scripted sequence, with generated IDs aliased as E1 through E4.

```powershell
python chapters/ch05/03_memory_learning.py --live --model YOUR_MODEL_ID
```

Live mode needs the requested provider's credentials and makes model API calls. It still uses fictional support tools, so it changes no real laptop, account, or studio system. Model choices can differ from the scripted trace, and its fixed numbers should not be treated as expected model performance.

## Corpus and cases

| Runbook | Problem and observation |
|---------|-------------------------|
| [VPN-01](corpus/01_orbitlink_vpn.md) | Disconnected OrbitLink tunnel after sleep; asset service unreachable |
| [ASSET-02](corpus/02_cratesync_assets.md) | NET01 connectivity, AUTH17 session expiry, and PERM04 membership |
| [BUILD-03](corpus/03_builddock_queue.md) | Queued, running, and failed BuildDock jobs |
| [AUDIO-04](corpus/04_stagelink_audio.md) | StageLink selects the laptop microphone rather than the headset |
| [ACCESS-05](corpus/05_accessdesk_permissions.md) | Project membership requires an approval; submitting a request grants nothing |
| [SUPPORT-06](corpus/06_support_handoffs.md) | Retain attempted steps and outcomes, then pass appropriate case history onward |

All six guides and their invented systems were written for this example. `StudioCorpus` indexes only these Markdown guides, split into named sections with source attribution. The separate [ticket fixtures](tickets.json) include seven scenarios and expected-action labels for validation; those labels and private simulator states never enter corpus retrieval.

The main walkthrough uses the first four requests, including a follow-up to an unresolved case. The final three tickets illustrate why audio, build-queue, and permission tasks need suitable outcome criteria. They are not included in the four-request resolution reward sequence.

## What the trace teaches

| Request | What happened | Learning |
|---------|---------------|----------|
| MB-201 | Reconnect the disconnected VPN; sync succeeds | Write E1 at 0.50; no earlier episode to credit |
| MB-202 | Recall E1 in a later session; the same fix succeeds | E1 moves from 0.50 to 0.75; write E2 at 0.50 |
| MB-203 | Recall E1; reconnecting leaves AUTH17 unresolved | E1 moves from 0.75 to 0.375; retain the failure as E3 |
| MB-203-followup | Recall E3; refreshing the asset session resolves the issue | E3 moves from 0.50 to 0.75; write E4 at 0.50 |

E3 becomes useful because it records the failed attempt and the remaining error. Its increased value does not mean that the failed action succeeded or that co-recall establishes causality. A new agent reopens the store for Maya, while Leo's identity retrieves none of her episodes.

`MemoryLearner` applies `q_new = q + 0.5 * (credit - q)` to the recalled episode. Equal credit, recall count one, utility weight 0.35, recency weight zero, the encoder, and the prompt remain fixed throughout the walkthrough. New episodes start at 0.50 and are not credited for the request that creates them.

The simulator permits one attempted support step per request and reports an observed outcome. The walkthrough's reward measures whether that step resolved the sync problem, rather than whether the tool returned successfully or the answer sounded helpful. Diagnosis, waiting on a build, and requesting access can be useful without resolving a ticket, so those tasks require a broader evaluation than this reward alone.

The `measurement` object in a fresh `trace.json` reports three verified resolutions among four assigned requests, one unresolved outcome, and no missing outcomes. `measure_resolution()` retains all assigned requests in the rate's denominator and reports missing outcomes separately. These counts describe the walkthrough; a controlled comparison would be needed to attribute better resolution to learning.

The scripted responder deliberately repeats an earlier fix once before the changed cause becomes visible. It demonstrates the data flow and arithmetic; it does not establish that a language model learns to troubleshoot better. Token matching, one-entry recall, changed queries, and newly added episodes also limit what the trace can show about ranking.

## Code paths

- [corpus.py](corpus.py): fixed lexical encoding and runbook retrieval.
- [02_studio_agent.py](../02_studio_agent.py): the studio agent, two tools, and support simulator.
- [scripted.py](scripted.py): the explicit teaching responder used in default mode.
- [03_memory_learning.py](../03_memory_learning.py): store construction, learner attachment, requests, trace, and handoff.
- [MemoryLearner](../../../helix/memory/learning.py): the shared fixed learning step.
- [MemoryHooks](../../../helix/memory/hooks.py): unchanged recall, injection, and write-back.

```powershell
python -m pytest tests/test_studio_memory_walkthrough.py -q
```

The tests cover source retrieval, outcome semantics, encoder reuse, value updates, tool execution, immutable versions, cross-session recall, and scoped handoff to a second agent. They also block model-provider calls in scripted mode and reject reuse of a nonempty output directory. Earlier research examples and their recorded scores remain separate from this studio trace.
