"""Evaluation subsystem.

EvalSet and EvalSource provide the data side of evaluation. The Eval primitive
(SPEC section 8, tournaments, drift detection, calibration) lands in a later
stratum; for now this module gives Improver-style flows a typed way to load
questions and reference answers.
"""

from helix.eval.dataset import EvalQuestion, EvalSet, load_eval_set
from helix.eval.source import EvalSource, FixedEvalSet, RecentTrajectorySource

__all__ = [
    "EvalQuestion",
    "EvalSet",
    "EvalSource",
    "FixedEvalSet",
    "RecentTrajectorySource",
    "load_eval_set",
]
