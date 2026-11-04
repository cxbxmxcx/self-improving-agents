# Studio memory walkthrough

scripted walkthrough; not measured model performance

## MB-201

Moonbase Mayhem asset sync failed after my laptop woke. CrateSync shows NET01 and OrbitLink is disconnected.

Attempted reconnect_vpn; resolved. OrbitLink connected; the asset sync succeeded. Source consulted: 01_orbitlink_vpn.md.

Reward: 1.0

Recalled: `[]`

## MB-202

Moonbase Mayhem asset sync is stuck again after my laptop woke. Can we use what worked last time?

Attempted reconnect_vpn; resolved. OrbitLink connected; the asset sync succeeded. Source consulted: 01_orbitlink_vpn.md.

Reward: 1.0

Recalled: `[{"episode": "E1", "recalled_ref": ["memory.episodic.09815cf8-14ba-4944-b759-ea270bdf618b", 1], "current_ref": ["memory.episodic.09815cf8-14ba-4944-b759-ea270bdf618b", 2], "q_before": 0.5, "q_after": 0.75}]`

## MB-203

Moonbase Mayhem asset sync is stuck again. Should we try the previous fix?

Attempted reconnect_vpn; unresolved. OrbitLink is connected; CrateSync still reports AUTH17. Source consulted: 01_orbitlink_vpn.md.

Reward: 0.0

Recalled: `[{"episode": "E1", "recalled_ref": ["memory.episodic.09815cf8-14ba-4944-b759-ea270bdf618b", 2], "current_ref": ["memory.episodic.09815cf8-14ba-4944-b759-ea270bdf618b", 3], "q_before": 0.75, "q_after": 0.375}]`

## MB-203-followup

Reconnecting did not help. The VPN is connected, but CrateSync shows AUTH17. What should we try next?

Attempted refresh_asset_token; resolved. CrateSync session refreshed; the asset sync succeeded. Source consulted: 02_cratesync_assets.md.

Reward: 1.0

Recalled: `[{"episode": "E3", "recalled_ref": ["memory.episodic.20508c59-44a1-41ed-ae0d-68f6faa15775", 1], "current_ref": ["memory.episodic.20508c59-44a1-41ed-ae0d-68f6faa15775", 2], "q_before": 0.5, "q_after": 0.75}]`
