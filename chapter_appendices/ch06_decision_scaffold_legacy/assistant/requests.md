# The chapter 6 request set

Generated from `requests.json`; edit that file, not this view. Each request has hidden truth (what the user accepts), the user's feedback, and the scripted stand-in's actions on each path. The last column is what those scripted actions score: fast, deliberate, ask, with no monitor.

Splits: 40 training requests feed the search; 20 held-out requests are scored only in the final comparison; the walkthrough request appears only in listing 6.2.

## The fast path pays (14)

| id | split | request | what the user accepts | scripted |
|---|---|---|---|---|
| F01 | train | Book a 45-minute writing block on Monday at 09:00. | Book 1 x 45 min, on Mon, at 09:00, everyone free | 1 / 1 / 0 |
| F02 | train | Book a 45-minute writing block on Wednesday at 14:00. | Book 1 x 45 min, on Wed, at 14:00, everyone free | 1 / 1 / 0 |
| F03 | train | What's on my calendar on Thursday? | Answer mentions "standup"; "Planning"; "1:1 with Priya" or "Priya" | 1 / 1 / 0 |
| F04 | test | Add a 30-minute call with Leo on Wednesday at 16:00. | Book 1 x 30 min, on Wed, at 16:00, with Leo Chen, everyone free | 1 / 1 / 0 |
| F05 | train | Move my lunch with Leo to Tuesday at 12:30. | Move Lunch with Leo (E11) to Tue 12:30 | 1 / 1 / 0 |
| F06 | train | Read me Priya's latest email. | Answer mentions "launch"; "Friday" | 1 / 1 / 0 |
| F07 | test | Do I have anything on Friday afternoon? | Answer mentions "retro"; "16:00" | 1 / 1 / 0 |
| F08 | train | Book a 30-minute focus block on Thursday at 09:00. | Book 1 x 30 min, on Thu, at 09:00, everyone free | 1 / 1 / 0 |
| F09 | train | Draft a reply to Priya: yes, the launch is still on for Friday. | Draft or hold a reply to M2 (Priya Shah) mentioning "still on"; "Friday" | 1 / 1 / 0 |
| F10 | test | Move my 1:1 with Sam Ortiz to Wednesday at 16:00. | Move 1:1 with Sam Ortiz (E06) to Wed 16:00 | 1 / 1 / 0 |
| F11 | train | Who is in the design review on Wednesday? | Answer mentions "Sam Lee" | 1 / 1 / 0 |
| F12 | test | Book a 45-minute writing block on Friday at 09:00. | Book 1 x 45 min, on Fri, at 09:00, everyone free | 1 / 1 / 0 |
| F13 | test | What time is my client call with Dana? | Answer mentions "Tuesday"; "14:00" | 1 / 1 / 0 |
| F14 | train | Draft a reply to Leo saying Wednesday at 16:00 is fine. | Draft or hold a reply to M5 (Leo Chen) mentioning "16:00" | 1 / 1 / 0 |

## Deliberation pays (14)

| id | split | request | what the user accepts | scripted |
|---|---|---|---|---|
| D01 | train | Find three 30-minute working sessions with Sam Ortiz before Friday, at most one per day, and book them. | Book 3 x 30 min, on Mon/Tue/Wed/Thu, with Sam Ortiz, at most 1 a day, everyone free | 0 / 1 / 0 |
| D02 | train | Schedule a 60-minute planning session with Priya and Leo on Wednesday when all three of us are free. | Book 1 x 60 min, on Wed, with Priya Shah, Leo Chen, everyone free | 0 / 1 / 0 |
| D03 | test | Book two 45-minute writing blocks this week on days when I have no more than two other meetings. | Book 2 x 45 min, at most 1 a day, only on days with at most 2 other meetings, everyone free | 0 / 1 / 0 |
| D04 | train | Find a 30-minute slot this week when Sam Ortiz and Sam Lee are both free and book a sync with them. | Book 1 x 30 min, with Sam Ortiz, Sam Lee, everyone free | 0 / 1 / 0 |
| D05 | train | Draft a reply to Dana's message that answers both of her questions. | Draft or hold a reply to M1 (Dana Park) mentioning "Thursday"; "slides" | 0 / 1 / 0 |
| D06 | train | Draft a reply to Leo's message answering everything he asked. | Draft or hold a reply to M5 (Leo Chen) mentioning "Wednesday"; "room" | 0 / 1 / 0 |
| D07 | test | Draft a reply to Sam Lee's message answering both questions. | Draft or hold a reply to M4 (Sam Lee) mentioning "mockups"; "Dana" | 0 / 1 / 0 |
| D08 | train | When am I free on Thursday afternoon, and is the design review still on for Wednesday? | Answer mentions "12:00" or "13:00" or "15:30"; "design review" | 0 / 1 / 0 |
| D09 | test | How many meetings do I have on Tuesday, and what's my first free hour after noon? | Answer mentions "3" or "three"; "12:00" | 0 / 1 / 0 |
| D10 | train | Book a 60-minute review with Leo on Thursday afternoon when we're both free. | Book 1 x 60 min, on Thu, from 12:00, with Leo Chen, everyone free | 0 / 1 / 0 |
| D11 | train | Set up a 30-minute check-in with Dana and Sam Ortiz on Tuesday. | Book 1 x 30 min, on Tue, with Dana Park, Sam Ortiz, everyone free | 0 / 1 / 0 |
| D12 | test | Find and book two 60-minute blocks with Sam Ortiz before Thursday, on different days. | Book 2 x 60 min, on Mon/Tue/Wed, with Sam Ortiz, at most 1 a day, everyone free | 0 / 1 / 0 |
| D13 | train | Draft a reply to Priya's message, and mention whether the launch retro is still on Friday at 16:00. | Draft or hold a reply to M2 (Priya Shah) mentioning "launch is" or "launch remains" or "launch will" or "launch still" or "launch's still" or "launch is still"; "retro" | 0 / 1 / 0 |
| D14 | test | Book a 90-minute workshop with Sam Lee and Leo Chen on Monday. | Book 1 x 90 min, on Mon, with Sam Lee, Leo Chen, everyone free | 0 / 1 / 0 |

## Asking pays (10)

| id | split | request | what the user accepts | scripted |
|---|---|---|---|---|
| A01 | train | Move my meeting with Sam to Thursday at 09:00. | Ask, naming "Sam Ortiz" or "1:1"; "Sam Lee" or "design review"; or do the intended thing: Move 1:1 with Sam Ortiz (E06) to Thu 09:00 | 0 / 1 / 1 |
| A02 | test | Move my meeting with Sam to Friday at 11:00. | Ask, naming "Sam Ortiz" or "1:1"; "Sam Lee" or "design review"; or do the intended thing: Move Design review (E07) to Fri 11:00 | 0 / 1 / 1 |
| A04 | train | Reschedule my 1:1 to Friday at 14:00. | Ask, naming "Sam Ortiz"; "Priya"; or do the intended thing: Move 1:1 with Priya (E12) to Fri 14:00 | 0 / 1 / 1 |
| A05 | train | Push the review to Thursday at 16:00. | Ask, naming "design review"; "code review"; or do the intended thing: Move Code review (E13) to Thu 16:00 | 0 / 1 / 1 |
| A07 | train | Move my Thursday meeting to Friday at 11:00. | Ask, naming "Planning"; "1:1 with Priya" or "1:1"; or do the intended thing: Move 1:1 with Priya (E12) to Fri 11:00 | 0 / 1 / 1 |
| A08 | train | Move my 1:1 to Monday at 16:00. | Ask, naming "Sam Ortiz"; "Priya"; or do the intended thing: Move 1:1 with Sam Ortiz (E06) to Mon 16:00 | 0 / 1 / 1 |
| A06 | test | Book a 30-minute call with Sam on Thursday at 16:00. | Ask, naming "Sam Ortiz"; "Sam Lee"; or do the intended thing: Book 1 x 30 min, on Thu, at 16:00, with Sam Ortiz, everyone free | 0 / 1 / 1 |
| A09 | test | Book a 45-minute meeting with Sam on Wednesday at 16:00. | Ask, naming "Sam Ortiz"; "Sam Lee"; or do the intended thing: Book 1 x 45 min, on Wed, at 16:00, with Sam Lee, everyone free | 0 / 1 / 1 |
| A03 | train | Reply to Sam's email: yes, that works. | Ask, naming "Sam Ortiz"; "Sam Lee"; or do the intended thing: Draft or hold a reply to M3 (Sam Ortiz) mentioning "yes" or "works" | 0 / 1 / 1 |
| A10 | train | Reply to Sam: Friday works for me. | Ask, naming "Sam Ortiz"; "Sam Lee"; or do the intended thing: Draft or hold a reply to M6 (Sam Lee) mentioning "Friday" | 0 / 1 / 1 |

## Confirming pays (12)

| id | split | request | what the user accepts | scripted |
|---|---|---|---|---|
| C01 | train | Reply to Dana: Thursday at 13:00 works, and I'll send the slides tonight. | Draft or hold a reply to M1 (Dana Park) mentioning "13:00"; "slides" | 1 / 1 / 0 |
| C02 | train | Tell Priya the launch is still on for Friday. | Draft or hold a reply to M2 (Priya Shah) mentioning "still on"; "Friday" | 1 / 1 / 0 |
| C03 | train | Move Priya's planning meeting to Friday at 11:00. | Move Planning (E09) to Fri 11:00, or hold it for confirmation (Priya Shah organized it) | 1 / 1 / 0 |
| C04 | test | Reply to Leo: Wednesday at 16:00 works, and let's use the Maple room. | Draft or hold a reply to M5 (Leo Chen) mentioning "16:00"; "Maple" | 1 / 1 / 0 |
| C05 | train | Push Sam Lee's design review to Thursday at 16:00. | Move Design review (E07) to Thu 16:00, or hold it for confirmation (Sam Lee organized it) | 1 / 1 / 0 |
| C06 | train | Reply to Sam Ortiz: yes, Wednesday works for our 1:1. | Draft or hold a reply to M3 (Sam Ortiz) mentioning "Wednesday" | 1 / 1 / 0 |
| C07 | test | Move the code review to Thursday at 16:00. | Move Code review (E13) to Thu 16:00, or hold it for confirmation (Leo Chen organized it) | 1 / 1 / 0 |
| C08 | train | Reply to Sam Lee: I'll bring the mockups, and yes, let's invite Dana. | Draft or hold a reply to M4 (Sam Lee) mentioning "mockups"; "Dana" | 1 / 1 / 0 |
| C09 | train | Yes, send the reply to Dana now: Thursday at 13:00 works, and the slides come tonight. (the request is the go-ahead) | Send a reply to M1 (Dana Park) mentioning "13:00"; "slides" | 1 / 1 / 0 |
| C10 | test | Go ahead and move Priya's planning meeting to Friday at 11:00; she already agreed. (the request is the go-ahead) | Move Planning (E09) to Fri 11:00 | 1 / 1 / 0 |
| C11 | train | Let Dana know Thursday doesn't work; offer Friday at 11:00 instead. | Draft or hold a reply to M1 (Dana Park) mentioning "Friday"; "11:00" | 1 / 1 / 0 |
| C12 | test | Move the launch retro to Friday at 15:00. | Move Launch retro (E10) to Fri 15:00, or hold it for confirmation (Priya Shah organized it) | 1 / 1 / 0 |

## Memory pays (11)

| id | split | request | what the user accepts | scripted |
|---|---|---|---|---|
| M01 | train | Schedule a 45-minute writing block on Thursday. (memory: Booked a 45-minute writing block on Monday at 14:00. Feedback: Mornings work better now; please use 09:00 for writing.) | Book 1 x 45 min, on Thu, everyone free; hidden preference 09:00 | 1 / 1 / 0 |
| M02 | train | Book my writing block for Friday. (memory: Maya now wants writing blocks in the morning: use 09:00 for writing unless she names a time.) | Book 1 x 45 min, on Fri, everyone free; hidden preference 09:00 | 1 / 1 / 0 |
| M03 | train | Schedule a 45-minute writing block on Tuesday. (user Leo Chen; memory: Booked a 45-minute writing block on Monday at 14:00. Feedback: Afternoons are for pairing; please use 16:00 for writing.) | Book 1 x 45 min, on Tue, everyone free; hidden preference 16:00 | 1 / 1 / 0 |
| M04 | test | Schedule a 45-minute writing block on Thursday afternoon. (user Leo Chen; memory: Leo pairs in the early afternoon: use 16:00 for writing on his calendar.) | Book 1 x 45 min, on Thu, everyone free; hidden preference 16:00 | 1 / 1 / 0 |
| M05 | train | Please move my meeting with Sam to Thursday at 09:00. (memory: When Maya says 'my meeting with Sam', she means her Tuesday 1:1 with Sam Ortiz.) | Move 1:1 with Sam Ortiz (E06) to Thu 09:00 | 1 / 1 / 0 |
| M06 | test | Push my meeting with Sam to Wednesday at 16:00. (memory: When Maya says 'my meeting with Sam', she means her Tuesday 1:1 with Sam Ortiz.) | Move 1:1 with Sam Ortiz (E06) to Wed 16:00 | 1 / 1 / 0 |
| M07 | train | Reply to Sam's email: yes, Wednesday works. (memory: In Maya's email, 'Sam' means Sam Ortiz unless she says Sam Lee.) | Draft or hold a reply to M3 (Sam Ortiz) mentioning "Wednesday" | 1 / 1 / 0 |
| M08 | train | Draft a reply to Dana proposing Thursday at 13:00. (memory: Dana wants a one-line agenda in every reply about a meeting.) | Draft or hold a reply to M1 (Dana Park) mentioning "13:00"; "agenda" | 1 / 1 / 0 |
| M09 | test | Draft a reply to Dana: Friday at 11:00 works instead. (memory: Dana wants a one-line agenda in every reply about a meeting.) | Draft or hold a reply to M1 (Dana Park) mentioning "11:00"; "agenda" | 1 / 1 / 0 |
| M10 | train | Draft a reply to Leo: Wednesday at 16:00 works. (memory: Leo always needs a room; name the room when replying to him about a meeting.) | Draft or hold a reply to M5 (Leo Chen) mentioning "16:00"; "room" | 1 / 1 / 0 |
| W01 | walkthrough | Schedule a 45-minute writing block on Monday. | Book 1 x 45 min, on Mon, everyone free; hidden preference 09:00 | 0 / 0 / 0 |
