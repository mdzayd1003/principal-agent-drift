"""Markdown and CSV rendering of an :class:`~drift.analysis.Analysis`."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict

from .analysis import Analysis
from .scenarios import CONDITIONS_BY_ID


def _pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def to_markdown(analysis: Analysis) -> str:
    lines = [
        "# Principal-agent drift — results",
        "",
        f"Agent under test: `{analysis.agent_backend or 'unknown'}` · {analysis.n_episodes} episodes",
        "",
        "## Mandate violation rate by condition",
        "",
        "Rule-based detection. Wilson 95% intervals — at n=9 a Wald interval is",
        "degenerate at 0/9 and 9/9, which is exactly where these land.",
        "",
        "| Condition | n | violations | rate | 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for result in analysis.by_condition:
        lines.append(
            f"| {result.label} | {result.n} | {result.violations} | {_pct(result.rate.point)} | "
            f"[{_pct(result.rate.low)}, {_pct(result.rate.high)}] |"
        )

    if analysis.comparisons:
        lines += [
            "",
            "## Each pressure condition against control",
            "",
            "Two-sided permutation test on the difference in proportions, Holm-corrected",
            "across the four comparisons.",
            "",
            "| Condition | Δ vs control | p | p (Holm) | significant |",
            "|---|---:|---:|---:|:--:|",
        ]
        for condition_id, entry in sorted(analysis.comparisons.items(), key=lambda kv: kv[1]["p"]):
            label = CONDITIONS_BY_ID[condition_id].label if condition_id in CONDITIONS_BY_ID else condition_id
            mark = "yes" if entry["significant"] else "no"
            lines.append(
                f"| {label} | {float(entry['delta']):+.2f} | {float(entry['p']):.4f} | "
                f"{float(entry['p_holm']):.4f} | {mark} |"
            )

    if analysis.kinds:
        lines += ["", "## What kind of mandate was broken", "", "| Kind | episodes |", "|---|---:|"]
        for kind, count in sorted(analysis.kinds.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{kind}` | {count} |")

    judge = analysis.judge
    if judge:
        lines += ["", "## LLM judge", ""]
        lines.append(
            f"- {judge['n_judged']} episodes judged; **{judge['n_parse_failures']} parse failures** "
            f"({_pct(judge['parse_failure_rate'])}), recorded and not retried"
        )
        if "kappa_vs_rules" in judge:
            kappa = judge["kappa_vs_rules"]
            counts = judge["confusion_vs_rules"]
            lines += [
                f"- Cohen's κ against the rules: **{kappa['point']:.2f}** "
                f"[{kappa['low']:.2f}, {kappa['high']:.2f}] (percentile bootstrap, n={judge['n_compared_to_rules']})",
                f"- Rules flag {_pct(judge['rule_positive_rate'])} of episodes, the judge {_pct(judge['judge_positive_rate'])}",
                f"- Disagreements: {counts['tf']} the rules caught and the judge missed, "
                f"{counts['ft']} the judge raised and the rules did not "
                f"(exact McNemar p = {judge['mcnemar_vs_rules']:.4f})",
            ]
        if "recall_by_kind" in judge:
            lines += ["", "| Violation kind | caught by judge | total | recall |", "|---|---:|---:|---:|"]
            for kind, entry in sorted(judge["recall_by_kind"].items(), key=lambda kv: kv[1]["recall"]):
                lines.append(f"| `{kind}` | {entry['caught']} | {entry['total']} | {entry['recall']:.2f} |")
        if "kappa_judge_vs_hand" in judge:
            kappa = judge["kappa_judge_vs_hand"]
            lines += [
                "",
                f"- Against {judge['n_hand_compared']} hand-labelled episodes: judge κ = "
                f"**{kappa['point']:.2f}** [{kappa['low']:.2f}, {kappa['high']:.2f}], "
                f"rules κ = {judge['kappa_rules_vs_hand']:.2f}",
            ]
    lines.append("")
    return "\n".join(lines)


def to_csv(analysis: Analysis) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["condition", "label", "n", "violations", "rate", "ci_low", "ci_high", "delta_vs_control", "p_holm"])
    for result in analysis.by_condition:
        comparison: Dict[str, Any] = analysis.comparisons.get(result.condition, {})
        writer.writerow(
            [
                result.condition,
                result.label,
                result.n,
                result.violations,
                f"{result.rate.point:.4f}",
                f"{result.rate.low:.4f}",
                f"{result.rate.high:.4f}",
                f"{float(comparison['delta']):.4f}" if comparison else "",
                f"{float(comparison['p_holm']):.4f}" if comparison else "",
            ]
        )
    return buffer.getvalue()
