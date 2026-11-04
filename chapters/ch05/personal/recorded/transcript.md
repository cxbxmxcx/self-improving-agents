# Personal calendar memory walkthrough

scripted walkthrough; not measured model performance

## CAL-201

Schedule a 45-minute writing block on 2026-10-05.

Created a 45-minute writing block on 2026-10-05 at 14:00. Booking feedback: accepted. User feedback: That works for me. The event remains at its booked time.

Reward: 1.0

Recalled: `[]`

## CAL-202

Schedule a 45-minute writing block on 2026-10-06.

Created a 45-minute writing block on 2026-10-06 at 14:00. Booking feedback: accepted. User feedback: That works for me. The event remains at its booked time.

Reward: 1.0

Recalled: `[{"episode": "E1", "recalled_ref": ["memory.episodic.bf16ae23-21a8-4c53-8d69-550d7fb9a254", 1], "current_ref": ["memory.episodic.bf16ae23-21a8-4c53-8d69-550d7fb9a254", 2], "q_before": 0.5, "q_after": 0.75}]`

## CAL-203

Schedule a 45-minute writing block on 2026-10-07.

Created a 45-minute writing block on 2026-10-07 at 14:00. Booking feedback: rejected. User feedback: Mornings work better now; please use 09:00 for writing. The event remains at its booked time.

Reward: 0.0

Recalled: `[{"episode": "E1", "recalled_ref": ["memory.episodic.bf16ae23-21a8-4c53-8d69-550d7fb9a254", 2], "current_ref": ["memory.episodic.bf16ae23-21a8-4c53-8d69-550d7fb9a254", 3], "q_before": 0.75, "q_after": 0.375}]`

## CAL-204

Schedule a 45-minute writing block on 2026-10-08. Use my feedback from the rejected writing booking.

Created a 45-minute writing block on 2026-10-08 at 09:00. Booking feedback: accepted. User feedback: That works for me. The event remains at its booked time.

Reward: 1.0

Recalled: `[{"episode": "E3", "recalled_ref": ["memory.episodic.e6daa064-9a33-4b22-b28c-5d6dd6de8a57", 1], "current_ref": ["memory.episodic.e6daa064-9a33-4b22-b28c-5d6dd6de8a57", 2], "q_before": 0.5, "q_after": 0.75}]`
