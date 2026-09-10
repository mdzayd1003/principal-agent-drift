"""Turning episodes and judgments into numbers that can be defended."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .judge import Judgment
from .negotiation import Episode
from .scenarios import CONDITIONS, CONDITIONS_BY_ID
from .stats import (
    Interval,
    cohens_kappa,
    confusion,
    holm_bonferroni,
    kappa_with_ci,
    mcnemar_exact,
    permutation_test_two_proportions,
    wilson_interval,
)


@dataclass
class ConditionResult:
    condition: str
    label: str
    n: int
    violations: int
    rate: Interval
    kinds: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "label": self.label,
            "n": self.n,
            "violations": self.violations,
            "rate": self.rate.point,
            "ci_low": self.rate.low,
            "ci_high": self.rate.high,
            "kinds": self.kinds,
        }


@dataclass
class Analysis:
    n_episodes: int
    by_condition: List[ConditionResult]
    comparisons: Dict[str, Dict[str, float | bool]]
    judge: Dict[str, Any]
    kinds: Dict[str, int]
    agent_backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "agent_backend": self.agent_backend,
            "by_condition": [c.to_dict() for c in self.by_condition],
            "vs_control": self.comparisons,
            "judge": self.judge,
            "violation_kinds": self.kinds,
        }


def drift_by_condition(episodes: Sequence[Episode]) -> List[ConditionResult]:
    grouped: Dict[str, List[Episode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.condition_id].append(episode)

    out: List[ConditionResult] = []
    for condition in CONDITIONS:
        bucket = grouped.get(condition.id, [])
        if not bucket:
            continue
        violations = sum(e.rule_violation for e in bucket)
        out.append(
            ConditionResult(
                condition=condition.id,
                label=condition.label,
                n=len(bucket),
                violations=violations,
                rate=wilson_interval(violations, len(bucket)),
                kinds=dict(Counter(k for e in bucket for k in e.violated_kinds)),
            )
        )
    return out


def compare_to_control(episodes: Sequence[Episode], *, resamples: int = 10000, seed: int = 0) -> Dict[str, Dict[str, float | bool]]:
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.condition_id].append(episode.rule_violation)

    control_id = next((c.id for c in CONDITIONS if c.is_control), "control")
    control = grouped.get(control_id, [])
    if not control:
        return {}

    raw: Dict[str, float] = {}
    deltas: Dict[str, float] = {}
    for condition_id, labels in grouped.items():
        if condition_id == control_id or not labels:
            continue
        raw[condition_id] = permutation_test_two_proportions(control, labels, resamples=resamples, seed=seed)
        deltas[condition_id] = sum(labels) / len(labels) - sum(control) / len(control)

    corrected = holm_bonferroni(raw)
    for condition_id, entry in corrected.items():
        entry["delta"] = deltas[condition_id]
    return corrected


def judge_agreement(
    episodes: Sequence[Episode],
    judgments: Sequence[Judgment],
    hand_labels: Optional[Dict[str, bool]] = None,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Judge vs rules on everything, and judge vs hand labels on the sample."""
    by_id = {j.episode_id: j for j in judgments}
    parse_failures = [j.episode_id for j in judgments if j.parse_failure]

    paired = [(e, by_id[e.id]) for e in episodes if e.id in by_id and not by_id[e.id].parse_failure]
    rule_labels = [e.rule_violation for e, _ in paired]
    judge_labels = [bool(j.violation) for _, j in paired]

    out: Dict[str, Any] = {
        "n_judged": len(judgments),
        "n_parse_failures": len(parse_failures),
        "parse_failure_rate": len(parse_failures) / len(judgments) if judgments else 0.0,
        "parse_failure_ids": parse_failures,
        "n_compared_to_rules": len(paired),
    }

    if paired:
        interval = kappa_with_ci(rule_labels, judge_labels, resamples=resamples, seed=seed)
        out["kappa_vs_rules"] = {"point": interval.point, "low": interval.low, "high": interval.high}
        out["confusion_vs_rules"] = confusion(rule_labels, judge_labels)
        out["mcnemar_vs_rules"] = mcnemar_exact(rule_labels, judge_labels)
        out["rule_positive_rate"] = sum(rule_labels) / len(rule_labels)
        out["judge_positive_rate"] = sum(judge_labels) / len(judge_labels)
        out["recall_by_kind"] = _recall_by_kind(paired)

    if hand_labels:
        sample = [(e, by_id[e.id]) for e in episodes if e.id in hand_labels and e.id in by_id]
        usable = [(e, j) for e, j in sample if not j.parse_failure]
        out["n_hand_labelled"] = len(sample)
        out["n_hand_compared"] = len(usable)
        if usable:
            hand = [hand_labels[e.id] for e, _ in usable]
            judged = [bool(j.violation) for _, j in usable]
            rules_on_sample = [e.rule_violation for e, _ in usable]
            interval = kappa_with_ci(hand, judged, resamples=resamples, seed=seed)
            out["kappa_judge_vs_hand"] = {"point": interval.point, "low": interval.low, "high": interval.high}
            out["kappa_rules_vs_hand"] = cohens_kappa(hand, rules_on_sample)
            out["confusion_judge_vs_hand"] = confusion(hand, judged)
    return out


def _recall_by_kind(paired: Sequence[tuple]) -> Dict[str, Dict[str, float]]:
    """How often the judge catches a violation, split by what kind it was.

    The split matters: a judge can look adequate overall while being blind to a
    whole class of violation, and an omission (something the agent failed to say)
    is the class most likely to be missed.
    """
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for episode, judgment in paired:
        for kind in episode.violated_kinds:
            stats[kind][1] += 1
            if judgment.violation:
                stats[kind][0] += 1
    return {kind: {"caught": caught, "total": total, "recall": caught / total} for kind, (caught, total) in stats.items()}


def analyse(
    episodes: Sequence[Episode],
    judgments: Sequence[Judgment] = (),
    hand_labels: Optional[Dict[str, bool]] = None,
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> Analysis:
    return Analysis(
        n_episodes=len(episodes),
        agent_backend=episodes[0].agent_backend if episodes else "",
        by_condition=drift_by_condition(episodes),
        comparisons=compare_to_control(episodes, resamples=resamples, seed=seed),
        judge=judge_agreement(episodes, judgments, hand_labels, seed=seed) if judgments else {},
        kinds=dict(Counter(k for e in episodes for k in e.violated_kinds)),
    )
