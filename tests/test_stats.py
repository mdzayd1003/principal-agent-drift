import math

import pytest

from drift.stats import (
    bootstrap_ci,
    cohens_kappa,
    confusion,
    holm_bonferroni,
    kappa_with_ci,
    mcnemar_exact,
    permutation_test_two_proportions,
    wilson_interval,
)


def test_wilson_is_not_degenerate_at_the_boundaries():
    at_one = wilson_interval(9, 9)
    at_zero = wilson_interval(0, 9)
    assert at_one.point == 1.0 and at_one.low < 1.0
    assert at_zero.point == 0.0 and at_zero.high > 0.0


def test_wilson_narrows_as_the_sample_grows():
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large.high - large.low) < (small.high - small.low)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_kappa_is_one_for_identical_raters():
    labels = [True, False, True, True, False]
    assert cohens_kappa(labels, labels) == 1.0


def test_kappa_is_one_when_both_raters_are_constant_and_agree():
    assert cohens_kappa([True] * 5, [True] * 5) == 1.0


def test_kappa_is_zero_when_one_rater_is_constant_and_they_disagree():
    assert cohens_kappa([True] * 5, [False] * 5) == 0.0


def test_kappa_is_near_zero_for_independent_raters():
    a = [i % 2 == 0 for i in range(100)]
    b = [i % 3 == 0 for i in range(100)]
    assert abs(cohens_kappa(a, b)) < 0.2


def test_kappa_matches_a_worked_example():
    # 20 both yes, 5 a-only, 10 b-only, 15 both no
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4, abs=0.01)


def test_kappa_needs_a_sample():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        confusion([True], [True, False])


def test_the_bootstrap_interval_brackets_the_point_estimate():
    a = [True] * 7 + [False] * 3
    b = [True] * 6 + [False] * 4
    interval = kappa_with_ci(a, b, resamples=500, seed=1)
    assert interval.low <= interval.point <= interval.high


def test_the_bootstrap_is_reproducible_from_its_seed():
    a, b = [True, False] * 10, [True, True, False, False] * 5
    first = bootstrap_ci(lambda idx: sum(a[i] for i in idx) / len(idx), len(a), resamples=200, seed=7)
    second = bootstrap_ci(lambda idx: sum(a[i] for i in idx) / len(idx), len(a), resamples=200, seed=7)
    assert first == second


def test_a_clear_difference_gives_a_small_permutation_p():
    assert permutation_test_two_proportions([False] * 9, [True] * 9, resamples=2000) < 0.01


def test_identical_groups_give_a_large_permutation_p():
    labels = [True, False, True, False]
    assert permutation_test_two_proportions(labels, labels, resamples=2000) > 0.5


def test_a_permutation_p_is_never_exactly_zero():
    assert permutation_test_two_proportions([False] * 20, [True] * 20, resamples=1000) > 0


def test_permutation_needs_two_non_empty_groups():
    with pytest.raises(ValueError):
        permutation_test_two_proportions([], [True])


def test_holm_is_monotone_and_never_shrinks_a_p_value():
    corrected = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert all(corrected[k]["p_holm"] >= corrected[k]["p"] for k in corrected)
    ordered = sorted(corrected.values(), key=lambda v: v["p"])
    assert [v["p_holm"] for v in ordered] == sorted(v["p_holm"] for v in ordered)


def test_holm_is_stricter_than_no_correction():
    corrected = holm_bonferroni({"a": 0.02, "b": 0.03, "c": 0.04, "d": 0.045})
    assert not any(entry["significant"] for entry in corrected.values())


def test_mcnemar_detects_a_one_sided_disagreement():
    a = [True] * 10 + [False] * 10
    b = [True] * 2 + [False] * 8 + [False] * 10
    assert mcnemar_exact(a, b) < 0.05


def test_mcnemar_is_one_when_there_is_nothing_to_disagree_about():
    labels = [True, False, True]
    assert mcnemar_exact(labels, labels) == 1.0


def test_every_statistic_stays_finite_on_a_degenerate_sample():
    assert math.isnan(wilson_interval(0, 0).point)
    assert math.isnan(bootstrap_ci(lambda idx: 1.0, 0)[0])
