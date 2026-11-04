"""Chapter 5: the helpdesk traffic set and its plain-code checker.

The research helpdesk answers questions about the agent-literature corpus.
Chapter 2 wrote twenty of those questions with reference answers; this
module turns them into replayable traffic (three users, several sessions,
reworded askings so recall cannot parrot) and scores every answer with
code rather than a judge, the same recipe the revenue task used in chapters
3 and 4. A checker is code that can be wrong, so `00_baseline.py` prints its
verdicts next to the answers before anything trusts it.

The reward each answer earns is the outcome the memory loop learns from:
credit flows back to the memory entries the agent recalled on the way to
that answer (SPEC §3.3, the utility family).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CH02_QUESTIONS = Path(__file__).resolve().parents[2] / "chapters/ch02/eval_questions.json"
TRAFFIC_PATH = Path(__file__).resolve().parent / "traffic.json"


# ---------------------------------------------------------------------------
# The checker: one rule per question, written as code.
#
# Version history (SPEC §5.2.1: a changed rule is a new signal version, and
# measurements taken under different versions are not directly comparable):
#   v1: initial rules.
#   v2: markdown emphasis stripped before matching; the abstain verb list
#       gained report/state/provide/mention/address and the "don't" forms
#       (the baseline's correct Q16 and Q20 answers scored 0.0 under v1);
#       Q19 accepts a grounded reading of AgentPRM's over-optimization
#       observation, because the corpus paper does mention reward hacking
#       as PRM scores degrade with sampling budget, so only an invented
#       production claim now fails it; and traps became abstain-or-fail
#       with sentence-scoped fabrication overrides, because the v1 mixed
#       0.5 band punished the baseline's best answer (an abstention that
#       correctly cited DGM's numbers in the next sentence); fabrication
#       terms support a "!" prefix for must-not-match, so Q16 spares a
#       sentence attributing the number to DGM.
# ---------------------------------------------------------------------------

CHECKER_VERSION = 2


def _normalize(text: str) -> str:
    """Drop markdown emphasis so **not** and `not` match like not."""
    return re.sub(r"[*_`]+", "", text)


def _has(text: str, *alternatives: str) -> bool:
    """True when any alternative appears (case-insensitive, regex allowed)."""
    return any(re.search(a, text, flags=re.IGNORECASE) for a in alternatives)


_ABSTAIN_VERBS = r"(?:contain|cover|include|discuss|appear|report|state|provide|mention|address|have)"

ABSTAIN = (
    r"not (?:in|part of|covered by|present in|included in) the corpus",
    rf"(?:does|do) not {_ABSTAIN_VERBS}",
    rf"(?:doesn't|don't) {_ABSTAIN_VERBS}",
    r"no (?:paper|document|source)s? in the corpus",
    r"none of the (?:papers|documents)",
    r"isn't (?:in|covered|included)",
    r"is not (?:in|covered|included|discussed|reported)",
    r"cannot find",
    r"could not find",
    r"no information about",
    r"not reported",
    r"no such",
)


def _fraction(text: str, groups: list[tuple[str, ...]]) -> float:
    """Score = share of required groups satisfied; each group is any-of."""
    if not groups:
        return 0.0
    return sum(1.0 for g in groups if _has(text, *g)) / len(groups)


def _fab_sentence(sentence: str, terms: tuple[str, ...]) -> bool:
    """True when a sentence asserts the planted claim.

    Every plain term must match; a term prefixed with "!" must not, which
    is how Q16 spares a correct sentence that attributes the number to DGM
    while mentioning HyperAgents in passing.
    """
    for term in terms:
        if term.startswith("!"):
            if re.search(term[1:], sentence, flags=re.IGNORECASE):
                return False
        elif not re.search(term, sentence, flags=re.IGNORECASE):
            return False
    return True


def _trap(text: str, *fabrications: str, ok: tuple[str, ...] = ()) -> float:
    """A trap is passed by abstaining and failed by everything else.

    Abstention is any ABSTAIN phrase, or a question-specific correct
    behavior in `ok` (naming a category error, reading the paper right).
    A fabrication override forces 0.0 when one sentence affirmatively
    asserts the planted false claim: every pattern in one fabrication
    tuple must hit inside the same sentence, which keeps a correct
    sentence about a neighboring paper from tripping it.
    """
    for fab in fabrications:
        terms = fab if isinstance(fab, tuple) else (fab,)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            # A sentence that negates cannot be an affirmative assertion,
            # and reference-grade abstentions negate constantly.
            if re.search(r"\b(?:not|no|never)\b", sentence, flags=re.IGNORECASE):
                continue
            if _fab_sentence(sentence, terms):
                return 0.0
    if _has(text, *ABSTAIN) or (bool(ok) and _has(text, *ok)):
        return 1.0
    return 0.0


def check(question_id: str, answer: str | None) -> float:
    """Score an answer in [0, 1] with the rule for its question."""
    t = _normalize((answer or "").strip())
    if not t:
        return 0.0
    rules = {
        # Band 1: single-paper facts.
        "Q1": lambda: _fraction(t, [(r"\b20(?:\.0)?\s*%",), (r"\b50(?:\.0)?\s*%",)]),
        "Q2": lambda: _fraction(t, [(r"\badd\b",), (r"\bupvote\b",), (r"\bdownvote\b",), (r"\bedit\b",)]),
        "Q3": lambda: _fraction(t, [(r"genetic",), (r"pareto",), (r"\biclr\b",)]),
        "Q4": lambda: _fraction(t, [(r"worse|degrad|decreas|drop|decline|hurt|harm",)]),
        "Q5": lambda: _fraction(t, [(r"intent",), (r"experience",), (r"utility",)]),
        # Band 2: disambiguation, both sides must be named.
        "Q6": lambda: _fraction(t, [(r"reflexion",), (r"gepa",), (r"verbal|self-critique|self-reflection|critique",), (r"mutat|population|pareto",)]),
        "Q7": lambda: _fraction(t, [(r"evolution|genetic|population|pareto",), (r"monte carlo|mcts|tree search",), (r"workflow|code|program",)]),
        "Q8": lambda: _fraction(t, [(r"q-?learn|q-?value|value",), (r"intent|experience|utility|triplet",), (r"upvote|downvote|vote|insight",)]),
        "Q9": lambda: _fraction(t, [(r"archive|evolution|empirical",), (r"meta|search (?:method|procedure|process) itself|improvement procedure|self-referential|hyper",), (r"code",)]),
        "Q10": lambda: _fraction(t, [(r"step|process|learned|probabil|graded|score",), (r"proof|prove|deductive|formal",)]),
        # Band 3: multi-hop, the answer must carry every paper it needs.
        "Q11": lambda: _fraction(t, [(r"darwin|dgm",), (r"shinka",), (r"hyperagent",), (r"empirical|evaluat|benchmark",)]),
        "Q12": lambda: _fraction(t, [(r"\bape\b|automatic prompt engineer",), (r"\bspo\b|self-supervised prompt|pairwise",), (r"gepa",), (r"label|ground truth|training data",)]),
        "Q13": lambda: _fraction(t, [(r"mast|14|fourteen",), (r"autogen",), (r"metagpt",), (r"specification|coordination|verification",)]),
        "Q14": lambda: _fraction(t, [(r"archive|evolution",), (r"shinka",), (r"150|sample.efficien",), (r"22|cost|expensive|budget",)]),
        "Q15": lambda: _fraction(t, [(r"huang|self-correct",), (r"reflexion",), (r"\bwait\b|self-correction bench|deliberation|cue",), (r"oracle|stop|external|anchor|memory",)]),
        # Band 4: traps. Abstention passes; a sentence that affirmatively
        # asserts the planted claim fails outright.
        "Q16": lambda: _trap(t, (r"hyperagents", r"swe.?bench", r"(?:reports?|achiev|scores?|reach)\w*[^.!?]{0,40}\d", r"!\b(?:dgm|darwin)\b")),
        "Q17": lambda: _trap(t, (r"alphaevolve", r"\d+(?:\.\d+)?\s*%", r"speed|faster|improv")),
        "Q18": lambda: _trap(t, (r"introduc", r"free energy", r"memgpt|packer|park|generative agents|schmidhuber|friston|reflexion|voyager")),
        # Q19 is a softened trap: the AgentPRM paper does observe reward
        # hacking (PRM scores degrade as the sampling budget grows), so a
        # grounded reading of that observation passes; an invented claim
        # about production guidance fails.
        "Q19": lambda: _trap(
            t,
            (r"agentprm", r"warns|recommends|argues|advises", r"production"),
            ok=(r"not the same", r"conflat", r"distinct", r"different topic", r"not about reward hacking",
                r"does not (?:discuss|address|cover) reward hacking", r"is about[^.]{0,60}not reward hacking",
                r"over-?optimi", r"sampling budget", r"inference compute", r"degrad"),
        ),
        "Q20": lambda: _trap(
            t,
            (r"schmidhuber|g[oö]del", r"recommends?|specifies|defines", r"hook"),
            ok=(r"category error", r"presuppos", r"predates", r"no connection", r"not a system",
                r"running project", r"does not exist", r"never mentions? helixagent", r"unrelated"),
        ),
    }
    rule = rules.get(question_id)
    return float(rule()) if rule else 0.0


# ---------------------------------------------------------------------------
# Traffic: users, sessions, askings.
# ---------------------------------------------------------------------------

@dataclass
class Asking:
    """One request in the traffic: who asked which question, in which words."""

    user_id: str
    session_id: str
    question_id: str
    text: str
    band: int
    reference_answer: str
    tags: list[str] = field(default_factory=list)


def load_questions() -> list[dict[str, Any]]:
    return json.loads(CH02_QUESTIONS.read_text(encoding="utf-8"))["questions"]


def load_traffic(path: Path = TRAFFIC_PATH) -> list[Asking]:
    """The replayable stream, in order; built on first use from the questions."""
    if not path.exists():
        askings = build_traffic()
        save_traffic(askings, PARAPHRASES, path)
        return askings
    rows = json.loads(path.read_text(encoding="utf-8"))["askings"]
    return [Asking(**r) for r in rows]


# Users own bands, so each user's memory fills with the kind of question
# they keep asking: alice asks facts and disambiguations, bob asks for
# syntheses, carol keeps walking into traps.
USER_BANDS = {"alice": (1, 2), "bob": (3,), "carol": (4,)}

# Second-session wordings, written by hand so the traffic is deterministic
# and free to rebuild. Each keeps every entity and constraint of the original.
PARAPHRASES: dict[str, list[str]] = {
    "Q1": ["On SWE-Bench, what improvement does the Darwin Goedel Machine paper report as its headline number?"],
    "Q2": ["Name the four operations ExpeL applies to its memory entries."],
    "Q3": ["Expand the acronym GEPA, and say which conference the paper appeared at."],
    "Q4": ["Huang et al. at ICLR 2024: without an oracle telling the model when to stop, what happens to self-correction?"],
    "Q5": ["Over what unit of storage does MemRL learn its Q-values?"],
    "Q6": ["Reflexion and GEPA both talk about 'reflection'. What is different about what each one means by it?"],
    "Q7": ["When GEPA says 'search' and when AFlow says 'search', what are they each referring to?"],
    "Q8": ["Contrast the meaning of 'memory' in MemRL with its meaning in ExpeL."],
    "Q9": ["How does 'self-improvement' as the Darwin Goedel Machine uses it differ from the HyperAgents sense of the term?"],
    "Q10": ["'Verification' in Process Reward Models versus 'verification' in a Goedel Machine: what does each mean?"],
    "Q11": ["Which corpus papers take Schmidhuber's Goedel Machine as their motivation, and in what way does each loosen its demand for proofs?"],
    "Q12": ["Walk the lineage from APE in 2022 to GEPA in 2025: what requirement does each step drop that the previous one needed?"],
    "Q13": ["List the failure modes in the MAST taxonomy from Cemri et al. 2025, and explain which of them the designs of AutoGen and MetaGPT each tend to invite."],
    "Q14": ["As search strategies, what do the Darwin Goedel Machine and ShinkaEvolve share, and how do they differ on sample efficiency?"],
    "Q15": ["Which papers in the corpus claim that naive reflection does not work, and what remedy does each one put forward?"],
    "Q16": ["In the original HyperAgents paper, what SWE-Bench score was reported?"],
    "Q17": ["Which benchmark did AlphaEvolve evaluate its Gemini matrix-multiplication kernel on, and how large was the reported speedup?"],
    "Q18": ["Which of the corpus papers is the one that introduces the Free Energy Principle?"],
    "Q19": ["What is the AgentPRM paper's position on reward hacking in production systems?"],
    "Q20": ["According to Schmidhuber 2003, what hook ordering policy does the HelixAgent framework recommend?"],
}


def build_traffic(paraphrases: dict[str, list[str]] | None = None, passes: int = 2) -> list[Asking]:
    """Lay the questions out as sessions: one pass per session, paraphrased.

    Pass 0 uses the original wording; later passes use a paraphrase, so a
    recalled past interaction has to transfer rather than match.
    """
    paraphrases = PARAPHRASES if paraphrases is None else paraphrases
    questions = load_questions()
    by_user: dict[str, list[dict[str, Any]]] = {u: [] for u in USER_BANDS}
    for q in questions:
        for user, bands in USER_BANDS.items():
            if q["band"] in bands:
                by_user[user].append(q)
    askings: list[Asking] = []
    for p in range(passes):
        for user, qs in by_user.items():
            for q in qs:
                variants = paraphrases.get(q["id"], [])
                text = q["question"] if p == 0 or not variants else variants[(p - 1) % len(variants)]
                askings.append(
                    Asking(
                        user_id=user,
                        session_id=f"{user}-s{p + 1}",
                        question_id=q["id"],
                        text=text,
                        band=q["band"],
                        reference_answer=q["reference_answer"],
                        tags=list(q.get("tags", [])),
                    )
                )
    return askings


def save_traffic(askings: list[Asking], paraphrases: dict[str, list[str]] | None = None, path: Path = TRAFFIC_PATH) -> None:
    paraphrases = PARAPHRASES if paraphrases is None else paraphrases
    path.write_text(
        json.dumps(
            {
                "description": "Chapter 5 helpdesk traffic: three users, two sessions each, reworded by hand on the second pass.",
                "paraphrases": paraphrases,
                "askings": [a.__dict__ for a in askings],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
