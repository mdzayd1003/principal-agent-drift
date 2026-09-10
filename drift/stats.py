"""The statistics the analysis needs, written out rather than imported.

Small sample sizes are the whole problem here: 9 episodes per condition. Wald
intervals on a proportion of 9/9 give a width of zero, which is nonsense, so
proportions get Wilson intervals and everything else gets a bootstrap or a
permutation test. Nothing here assumes normality.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def wilson_interval(successes: int, total: int, z: float = Z_95) -> Interval:
    """Wilson score interval. Behaves at 0/n and n/n, unlike the Wald interval."""
    if total <= 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total]")
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return Interval(p, max(0.0, centre - spread), min(1.0, centre + spread))


def confusion(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, int]:
    if len(a) != len(b):
        raise ValueError("label sequences must be the same length")
    out = {"tt": 0, "tf": 0, "ft": 0, "ff": 0}
    for left, right in zip(a, b):
        key = ("t" if left else "f") + ("t" if right else "f")
        out[key] += 1
    return out


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Chance-corrected agreement between two binary raters.

    Returns 1.0 when both raters are constant and identical: observed and
    expected agreement are both 1, so the usual formula is 0/0. Perfect
    agreement is the honest reading, but a constant rater is worth noticing —
    ``analysis`` reports the marginals alongside kappa for exactly that reason.
    """
    if not a:
        raise ValueError("cannot compute kappa on an empty sample")
    counts = confusion(a, b)
    total = len(a)
    observed = (counts["tt"] + counts["ff"]) / total
    p_a = (counts["tt"] + counts["tf"]) / total
    p_b = (counts["tt"] + counts["ft"]) / total
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def bootstrap_ci(
    statistic: Callable[[Sequence[int]], float],
    n_items: int,
    *,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap over item indices, so paired data stays paired."""
    if n_items == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    values: List[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n_items) for _ in range(n_items)]
        try:
            values.append(statistic(indices))
        except ValueError:
            continue
    if not values:  # pragma: no cover - only if every resample is degenerate
        return (float("nan"), float("nan"))
    values.sort()
    low = values[int((alpha / 2) * (len(values) - 1))]
    high = values[int((1 - alpha / 2) * (len(values) - 1))]
    return (low, high)


def kappa_with_ci(
    a: Sequence[bool], b: Sequence[bool], *, resamples: int = 2000, seed: int = 0
) -> Interval:
    point = cohens_kappa(a, b)
    low, high = bootstrap_ci(
        lambda idx: cohens_kappa([a[i] for i in idx], [b[i] for i in idx]),
        len(a),
        resamples=resamples,
        seed=seed,
    )
    return Interval(point, low, high)


def permutation_test_two_proportions(
    a: Sequence[bool], b: Sequence[bool], *, resamples: int = 10000, seed: int = 0
) -> float:
    """Two-sided exact-ish p-value for a difference in proportions.

    With 9 per group an asymptotic test is not defensible; shuffling the group
    labels is, and it costs nothing at this size.
    """
    if not a or not b:
        raise ValueError("both groups must be non-empty")
    pool = list(a) + list(b)
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        rng.shuffle(pool)
        left, right = pool[: len(a)], pool[len(a) :]
        if abs(sum(left) / len(left) - sum(right) / len(right)) >= observed - 1e-12:
            extreme += 1
    # +1 smoothing: a permutation p-value is never exactly zero
    return (extreme + 1) / (resamples + 1)


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float | bool]]:
    """Holm step-down correction. Four comparisons against control need it."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    total = len(ordered)
    out: Dict[str, Dict[str, float | bool]] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(ordered):
        adjusted = min(1.0, max(running_max, (total - rank) * p))
        running_max = adjusted
        out[key] = {"p": p, "p_holm": adjusted, "significant": adjusted < alpha}
    return out


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact two-sided McNemar p-value for paired binary labels."""
    counts = confusion(a, b)
    discordant = counts["tf"] + counts["ft"]
    if discordant == 0:
        return 1.0
    smaller = min(counts["tf"], counts["ft"])
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)
