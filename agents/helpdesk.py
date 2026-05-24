"""Helpdesk agent: a RAG assistant with memory.

The reference Helix agent spec. Wires together:
  - Retrieval against the LanceDB corpus at data/helix_corpus.lance
  - Episodic memory at data/episodic.sqlite (per-session + per-user recall)
  - Semantic memory at data/semantic.sqlite (per-user preferences, per-org policy)
  - Procedural memory at data/procedural.sqlite (skill library — currently empty)

This is what a Streamlit chat app or chapter script consumes via:
    from helix.agents import load_agent
    agent = load_agent("helpdesk")
"""

from __future__ import annotations

from pathlib import Path

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Artifact, ArtifactKind, genesis
from helix.eval import RecentTrajectorySource
from helix.improvement import OfflineImprover, ImproverPolicy, Schedule
from helix.memory import EpisodicMemory, ProceduralMemory, SemanticMemory
from helix.retrieval.index import open_index
from helix.search.spo import SPO
from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree
from helix.tools import Tool, tool


# ---------------------------------------------------------------------------
# Metadata (read by helix.agents.describe_agent)
# ---------------------------------------------------------------------------

DESCRIPTION = (
    "RAG helpdesk assistant with episodic / semantic / procedural memory. "
    "Retrieves from a LanceDB corpus, recalls similar past interactions, "
    "and respects per-user and per-org facts."
)

DEFAULT_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT_ID = "prompt.helpdesk.system"


# Paths are relative to the repository root. Memory backends are
# process-wide singletons (lazy-initialized below) so multiple Agent
# instances share the same store.
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "helix_corpus.lance"
EPISODIC_PATH = REPO_ROOT / "data" / "episodic.sqlite"
SEMANTIC_PATH = REPO_ROOT / "data" / "semantic.sqlite"
PROCEDURAL_PATH = REPO_ROOT / "data" / "procedural.sqlite"


# ---------------------------------------------------------------------------
# Default system prompt (the artifact under improvement)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant for users asking about the agent literature.

When a user asks a question:
1. Use the `retrieve` tool to find relevant passages from the document corpus.
2. Answer based on what you retrieved. Quote or cite the source filename when
   you use retrieved content.
3. If the corpus does not contain the answer, say so directly rather than
   guessing. Distinguish "this corpus doesn't cover that topic" from "this is
   a trap question that presupposes false context."
4. If past interactions in the system prompt are relevant, refer to them; do
   not repeat answers verbatim if you already gave them.
5. Respect any per-user facts in the system prompt (preferred units,
   environment, etc.) and any per-org policies.

Be concise. Prefer specific citations over vague answers."""


def build_genesis_prompt() -> Artifact:
    """Return the genesis prompt artifact for this agent."""
    return genesis(
        id=SYSTEM_PROMPT_ID,
        kind=ArtifactKind.PROMPT,
        content=DEFAULT_SYSTEM_PROMPT,
        created_by="human",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Lazy-initialized so importing this module doesn't load the LanceDB index.
_retrieval_index = None


def _get_index():
    global _retrieval_index
    if _retrieval_index is None:
        if not CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"No corpus found at {CORPUS_PATH}. "
                "Run `python ingestion/build_index.py` first."
            )
        _retrieval_index = open_index(CORPUS_PATH)
    return _retrieval_index


def _build_retrieve_tool() -> Tool:
    """Build the corpus-retrieval tool fresh on every Agent construction.

    Tools wrap callables; we can reuse the underlying index across them.
    """
    index = _get_index()

    @tool(description=(
        "Search the document corpus for passages relevant to a query. "
        "Returns the top matching passages with their source filename and page."
    ))
    async def retrieve(query: str, k: int = 5) -> list[dict]:
        hits = index.search(query, k=k, mode="hybrid")
        return [
            {"source": h.source, "page": h.page, "text": h.text, "score": h.score}
            for h in hits
        ]

    return retrieve


# ---------------------------------------------------------------------------
# Memory singletons
# ---------------------------------------------------------------------------

_episodic: EpisodicMemory | None = None
_semantic: SemanticMemory | None = None
_procedural: ProceduralMemory | None = None


def _get_episodic() -> EpisodicMemory:
    global _episodic
    if _episodic is None:
        EPISODIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        _episodic = EpisodicMemory(EPISODIC_PATH, check_same_thread=False)
    return _episodic


def _get_semantic() -> SemanticMemory:
    global _semantic
    if _semantic is None:
        SEMANTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        _semantic = SemanticMemory(SEMANTIC_PATH, check_same_thread=False)
    return _semantic


def _get_procedural() -> ProceduralMemory:
    global _procedural
    if _procedural is None:
        PROCEDURAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _procedural = ProceduralMemory(PROCEDURAL_PATH, check_same_thread=False)
    return _procedural


# ---------------------------------------------------------------------------
# build() — the contract every spec exports
# ---------------------------------------------------------------------------

def build(
    system_prompt: Artifact | None = None,
    *,
    model: str = DEFAULT_MODEL,
    max_iterations: int = 10,
    max_tool_calls: int = 20,
    skip_memory: bool = False,
    **overrides,
) -> Agent:
    """Construct a fresh helpdesk Agent.

    Parameters:
        system_prompt: optional Artifact to use as the system prompt. When
            None, uses build_genesis_prompt() — but in practice an Improver
            or eval harness will pass `archive.best()` here so the agent
            uses the current champion.
        model: LiteLLM model id; defaults to Claude Haiku 4.5.
        max_iterations / max_tool_calls: per-run caps.
        skip_memory: if True, builds the agent without any persistent memory
            tiers (useful for offline eval where you don't want production
            traffic to populate the cache).

    Memory tiers are shared singletons across Agent instances so a chat
    app's multiple sessions write to the same episodic store. Each
    Agent.run() takes a MemoryContext to scope reads and writes.
    """
    if system_prompt is None:
        system_prompt = build_genesis_prompt()

    memory_tiers = {} if skip_memory else {
        "episodic": _get_episodic(),
        "semantic": _get_semantic(),
        "procedural": _get_procedural(),
    }

    return Agent(
        system_prompt=system_prompt,
        tools=[_build_retrieve_tool()],
        model=model,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        memory_tiers=memory_tiers,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Improvement plan — the agent's declared improvers
# ---------------------------------------------------------------------------

# Default archive location for helpdesk improvements. Chat UI can override.
DEFAULT_IMPROVER_ARCHIVE = REPO_ROOT / "data" / "chat_archive.sqlite"
DEFAULT_IMPROVER_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_IMPROVER_PROPOSER_MODEL = "claude-sonnet-4-6"
DEFAULT_IMPROVER_INTERVAL_SEC = 300.0
DEFAULT_IMPROVER_BUFFER_SIZE = 30
DEFAULT_IMPROVER_QUESTIONS_PER_ROUND = 4


def list_improvable_artifacts(agent: Agent) -> list[Artifact]:
    """Artifacts on this agent that can be put under improvement.

    Helpdesk v1 exposes one improvable artifact: the system prompt. Later
    chapters that introduce per-tool descriptions or skill-library entries
    will append those here. Each entry's id + kind appears in the chat UI's
    "Attach improver" dropdown so a user can target any of them.
    """
    return [agent.system_prompt]


def build_improvers(
    agent: Agent,
    *,
    archive_path: str | None = None,
) -> list[OfflineImprover]:
    """Declared default improvers for this agent.

    Returns one Improver targeting the system prompt with the Pattern A
    recipe: SwapAndAgree(PairwiseJudge) judging candidate prompts against
    the current champion, RecentTrajectorySource buffering live chat
    traffic, SPO mutating cheaply, INTERVAL schedule firing every 5 min.

    The chat UI calls this when the user clicks "Attach default improvers
    from spec." Users can pause/detach individual improvers afterward.
    Specs that want more elaborate plans (composite signals, strategy
    chains, multiple improvers across artifacts) override this function.
    """
    # Resolve archive path. None -> the helpdesk's default sqlite file.
    arc_path = Path(archive_path) if archive_path else DEFAULT_IMPROVER_ARCHIVE
    arc_path.parent.mkdir(parents=True, exist_ok=True)
    archive = SQLiteArchive(arc_path, check_same_thread=False)

    # Eval source consumes live chat trajectories from the event bus.
    eval_source = RecentTrajectorySource(buffer_size=DEFAULT_IMPROVER_BUFFER_SIZE)

    # Signal: pairwise judge with position-bias mitigation. Cheap, simple,
    # right default for live-traffic prompt improvement.
    signal = SwapAndAgree(PairwiseJudge(model=DEFAULT_IMPROVER_JUDGE_MODEL))

    # Search: SPO. One mutation per Improver round.
    search = SPO(proposer_model=DEFAULT_IMPROVER_PROPOSER_MODEL, rounds=1)

    # Policy: interval schedule, modest concurrency.
    policy = ImproverPolicy(
        schedule=Schedule.INTERVAL,
        interval_sec=DEFAULT_IMPROVER_INTERVAL_SEC,
        questions_per_round=DEFAULT_IMPROVER_QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
        max_concurrent_questions=2,
    )

    # The improver clones the agent via `agent.with_artifacts({...})` to test
    # each candidate. We build an agent here without memory tiers so the
    # improver's internal eval runs don't pollute live memory.
    improver_agent = build(
        system_prompt=agent.system_prompt,
        skip_memory=True,
    )

    improver = OfflineImprover(
        agent=improver_agent,
        target_artifact_id=agent.system_prompt.id,
        signal=signal,
        search=search,
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=agent.system_prompt,
        improver_id="helpdesk-prompt-spo",
    )
    return [improver]
