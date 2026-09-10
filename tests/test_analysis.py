import json
from pathlib import Path

from drift.analysis import analyse, compare_to_control, drift_by_condition, judge_agreement
from drift.judge import SimulatedJudge, judge_episodes
from drift.report import to_csv, to_markdown
from drift.scenarios import SCENARIOS_BY_ID

HAND_LABELS = json.loads(Path(__file__).resolve().parents[1].joinpath("data/hand_labels.json").read_text())


def test_every_condition_is_reported_with_nine_episodes(episodes):
    results = drift_by_condition(episodes)
    assert len(results) == 5
    assert all(r.n == 9 for r in results)
    assert all(r.rate.low <= r.rate.point <= r.rate.high for r in results)


def test_control_is_compared_against_every_other_condition(episodes):
    comparisons = compare_to_control(episodes, resamples=2000)
    assert set(comparisons) == {"time_pressure", "authority", "reciprocity", "incremental"}
    assert all(entry["delta"] > 0 for entry in comparisons.values())
    assert all(entry["p_holm"] >= entry["p"] for entry in comparisons.values())


def test_judge_agreement_reports_parse_failures_separately(episodes):
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    summary = judge_agreement(episodes, judgments, resamples=300)
    assert summary["n_judged"] == len(episodes)
    assert summary["n_parse_failures"] > 0
    assert summary["n_compared_to_rules"] == len(episodes) - summary["n_parse_failures"]


def test_the_judge_is_scored_per_violation_kind(episodes):
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    recall = judge_agreement(episodes, judgments, resamples=300)["recall_by_kind"]
    assert set(recall) <= {"price_floor", "no_disclosure", "no_commitment", "must_escalate"}
    assert all(0.0 <= entry["recall"] <= 1.0 for entry in recall.values())
    assert all(entry["caught"] <= entry["total"] for entry in recall.values())


def test_the_hand_labels_cover_a_stratified_sample(episodes_by_id):
    labels = HAND_LABELS["labels"]
    assert len(labels) == 15
    assert set(labels) <= set(episodes_by_id)
    conditions = {episodes_by_id[e].condition_id for e in labels}
    scenarios = {episodes_by_id[e].scenario_id for e in labels}
    assert len(conditions) == 5 and len(scenarios) == 3


def test_every_hand_label_carries_a_written_justification():
    assert all(entry["note"] for entry in HAND_LABELS["labels"].values())


def test_the_rules_agree_with_the_hand_labels_on_the_sample(episodes_by_id):
    labels = HAND_LABELS["labels"]
    for episode_id, entry in labels.items():
        episode = episodes_by_id[episode_id]
        assert episode.rule_violation == entry["violation"], episode_id
        assert set(episode.violated_kinds) == set(entry["kinds"]), episode_id


def test_the_judge_is_scored_against_the_hand_labels(episodes):
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    hand = {k: v["violation"] for k, v in HAND_LABELS["labels"].items()}
    summary = judge_agreement(episodes, judgments, hand, resamples=300)
    assert summary["n_hand_labelled"] == 15
    assert summary["kappa_rules_vs_hand"] == 1.0
    assert -1.0 <= summary["kappa_judge_vs_hand"]["point"] <= 1.0


def test_the_full_analysis_serialises(episodes):
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    payload = json.loads(json.dumps(analyse(episodes, judgments, resamples=1000).to_dict()))
    assert payload["n_episodes"] == 45
    assert len(payload["by_condition"]) == 5
    assert payload["judge"]["n_judged"] == 45


def test_the_markdown_report_contains_every_condition_and_the_judge_section(episodes):
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    markdown = to_markdown(analyse(episodes, judgments, resamples=1000))
    for label in ("Control", "Time pressure", "Authority appeal", "Reciprocity", "Incremental commitment"):
        assert label in markdown
    assert "parse failures" in markdown and "Cohen" in markdown


def test_the_csv_has_one_row_per_condition(episodes):
    rows = to_csv(analyse(episodes, resamples=500)).strip().splitlines()
    assert len(rows) == 6
    assert rows[0].startswith("condition,label,n,violations")
