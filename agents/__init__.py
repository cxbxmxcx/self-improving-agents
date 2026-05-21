"""Agent specifications.

Each module in this package describes one named agent. The convention is:

    # agents/<name>.py
    from helix.agent import Agent
    from helix.artifact import Artifact, ArtifactKind, genesis

    SYSTEM_PROMPT_ID = "prompt.<name>.system"
    DEFAULT_SYSTEM_PROMPT = \"\"\"...\"\"\"

    def build_genesis_prompt() -> Artifact:
        return genesis(
            id=SYSTEM_PROMPT_ID,
            kind=ArtifactKind.PROMPT,
            content=DEFAULT_SYSTEM_PROMPT,
            created_by="human",
        )

    def build(system_prompt: Artifact | None = None, **overrides) -> Agent:
        if system_prompt is None:
            system_prompt = build_genesis_prompt()
        return Agent(
            system_prompt=system_prompt,
            tools=[...],
            model=overrides.pop("model", "claude-haiku-4-5"),
            memory_tiers=...,
            **overrides,
        )

The platform's helix.agents loader resolves names to modules and calls
build(). Chat apps, eval harnesses, and chapter scripts all consume the same
spec. Multi-agent later: just point at multiple modules.

Reference example: agents/helpdesk.py
"""
