import json

from drift.judge import SimulatedJudge, build_judge_prompt, judge_episodes, parse_judgment
from drift.llm import SimulatedAgent
from drift.scenarios import SCENARIOS_BY_ID


def test_a_well_formed_verdict_parses():
    judgment = parse_judgment("e1", '{"violation": true, "mandates_broken": ["floor"], "quote": "17,000", "confidence": 0.9}')
    assert judgment.violation is True
    assert judgment.mandates_broken == ["floor"] and not judgment.parse_failure


def test_a_verdict_wrapped_in_prose_still_parses():
    judgment = parse_judgment("e1", 'Sure: {"violation": false, "mandates_broken": []} — hope that helps')
    assert judgment.violation is False and not judgment.parse_failure


def test_prose_with_no_json_is_a_parse_failure():
    judgment = parse_judgment("e1", "On balance the agent probably overstepped.")
    assert judgment.parse_failure and judgment.violation is None
    assert judgment.raw.startswith("On balance")


def test_a_non_boolean_verdict_is_a_parse_failure():
    assert parse_judgment("e1", '{"violation": "yes"}').parse_failure


def test_a_malformed_mandate_list_is_a_parse_failure():
    assert parse_judgment("e1", '{"violation": true, "mandates_broken": "floor"}').parse_failure


def test_the_judge_prompt_carries_the_mandate_and_the_transcript(episodes_by_id):
    episode = episodes_by_id["licensing-control-0"]
    prompt = build_judge_prompt(episode, SCENARIOS_BY_ID["licensing"])
    assert "MANDATE:" in prompt and "TRANSCRIPT:" in prompt
    assert episode.turns[0].text in prompt


def test_the_judge_ignores_the_mandate_text_when_looking_for_cues(episodes_by_id):
    # The mandate names the very figure the agent is told to withhold. A judge that
    # matched against it would fire on every episode, including compliant ones.
    compliant = [e for e in episodes_by_id.values() if not e.rule_violation]
    judgments = judge_episodes(compliant, SCENARIOS_BY_ID, SimulatedJudge())
    assert not any(j.violation for j in judgments if not j.parse_failure)


def test_parse_failures_are_recorded_and_never_retried(episodes):
    judge = SimulatedJudge()
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, judge)
    assert len(judge.calls) == len(episodes), "one call per episode, no retries"
    assert any(j.parse_failure for j in judgments)
    assert all(j.violation is None for j in judgments if j.parse_failure)


def test_judging_is_reproducible(episodes):
    first = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    second = judge_episodes(episodes, SCENARIOS_BY_ID, SimulatedJudge())
    assert [j.to_dict() for j in first] == [j.to_dict() for j in second]


def test_a_judgment_round_trips_through_json(episodes):
    judgment = judge_episodes(episodes[:1], SCENARIOS_BY_ID, SimulatedJudge())[0]
    assert json.loads(json.dumps(judgment.to_dict()))["episode_id"] == episodes[0].id
