# Recorded personal calendar walkthrough

These files were produced and verified for Chapter 5. The agent loop, tools, memory hooks, stores, and learner execute normally; the model responder and user reviews are scripted. The results demonstrate mechanics rather than language-model performance.

- [calendar.json](calendar.json): four created events, their original times, and observed feedback.
- [trace.json](trace.json): requests, tool calls, recalled versions, rewards, updates, configuration, and hashes.
- [transcript.md](transcript.md): readable requests and final reports.
- [recall-audit.json](recall-audit.json): the three candidates' relevance, utility, and ranking scores before request four.

The last request refers to prior rejected-booking feedback without repeating its preferred time. E3 ranks at approximately `0.63931`, ahead of E2 at `0.61875` and E1 at `0.57500`. The audit also records that a bare scheduling query selected E2 in the initial check.

Reproduce the walkthrough with `python chapters/ch05/03_memory_learning.py` from the repository root. A new run generates new IDs and a new directory, while the E1 through E4 aliases, scripted actions, and fixed learning settings remain comparable. The recall audit is supplementary validation evidence; the script's normal output consists of the calendar, stores, trace, and transcript.
