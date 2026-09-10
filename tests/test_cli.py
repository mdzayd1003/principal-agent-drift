import json
from pathlib import Path

from drift.__main__ import load_episodes, load_hand_labels, load_judgments, main


def test_quick_runs_the_whole_pipeline(tmp_path, capsys):
    assert main(["--results", str(tmp_path), "quick"]) == 0
    out = capsys.readouterr().out
    assert "ran 45 episodes" in out and "parse failures" in out

    for name in ("episodes.jsonl", "judgments.jsonl", "analysis.json", "report.md", "by_condition.csv"):
        assert (tmp_path / name).exists(), name

    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert analysis["n_episodes"] == 45
    assert len(analysis["by_condition"]) == 5


def test_saved_episodes_reload_identically(tmp_path):
    main(["--results", str(tmp_path), "run", "--simulated"])
    episodes = load_episodes(tmp_path / "episodes.jsonl")
    assert len(episodes) == 45
    assert all(e.violations or not e.rule_violation for e in episodes)
    assert episodes[0].transcript()


def test_run_honours_the_round_and_seed_counts(tmp_path):
    main(["--results", str(tmp_path), "run", "--simulated", "--rounds", "2", "--seeds", "1"])
    episodes = load_episodes(tmp_path / "episodes.jsonl")
    assert len(episodes) == 15
    assert all(len(e.agent_turns) == 2 for e in episodes)


def test_judgments_reload_with_their_parse_failures(tmp_path):
    main(["--results", str(tmp_path), "run", "--simulated"])
    main(["--results", str(tmp_path), "judge", "--simulated"])
    judgments = load_judgments(tmp_path / "judgments.jsonl")
    assert len(judgments) == 45
    failures = [j for j in judgments if j.parse_failure]
    assert failures and all(j.violation is None and j.raw for j in failures)


def test_hand_labels_load_from_the_shipped_file():
    labels = load_hand_labels(Path("data/hand_labels.json"))
    assert len(labels) == 15
    assert all(isinstance(v, bool) for v in labels.values())


def test_missing_hand_labels_are_not_fatal(tmp_path):
    assert load_hand_labels(tmp_path / "nope.json") == {}
