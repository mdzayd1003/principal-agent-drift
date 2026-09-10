import json

from drift.llm import SimulatedAgent
from drift.negotiation import build_agent_prompt, counterparty_turn, run_episode
from drift.scenarios import CONDITIONS, CONDITIONS_BY_ID, SCENARIOS, SCENARIOS_BY_ID, episode_grid, principal_prompt


def test_the_grid_is_the_full_crossed_design():
    grid = episode_grid()
    assert len(grid) == len(SCENARIOS) * len(CONDITIONS) * 3 == 45
    assert len({(s.id, c.id, seed) for s, c, seed in grid}) == 45


def test_exactly_one_condition_is_the_control():
    assert sum(c.is_control for c in CONDITIONS) == 1


def test_the_principal_prompt_states_every_mandate():
    scenario = SCENARIOS_BY_ID["licensing"]
    prompt = principal_prompt(scenario)
    for mandate in scenario.mandates:
        assert mandate.text in prompt


def test_the_counterparty_is_identical_across_conditions_at_the_opening():
    scenario = SCENARIOS_BY_ID["licensing"]
    openings = {counterparty_turn(scenario, c, 0) for c in CONDITIONS}
    assert openings == {scenario.opening}


def test_the_counterparty_differs_between_conditions_once_pressure_starts():
    scenario = SCENARIOS_BY_ID["licensing"]
    later = {counterparty_turn(scenario, c, 1) for c in CONDITIONS}
    assert len(later) == len(CONDITIONS)


def test_the_out_of_scope_term_is_raised_exactly_once():
    scenario = SCENARIOS_BY_ID["delivery"]
    condition = CONDITIONS_BY_ID["control"]
    turns = [counterparty_turn(scenario, condition, r) for r in range(6)]
    assert sum("penalty" in t.lower() for t in turns) == 1


def test_an_episode_alternates_speakers_and_ends_with_the_agent():
    episode = run_episode(SCENARIOS_BY_ID["licensing"], CONDITIONS_BY_ID["control"], 0, SimulatedAgent(), rounds=3)
    assert [t.speaker for t in episode.turns] == ["counterparty", "agent"] * 3
    assert len(episode.agent_turns) == 3


def test_episodes_are_reproducible():
    args = (SCENARIOS_BY_ID["consulting"], CONDITIONS_BY_ID["time_pressure"], 1, SimulatedAgent())
    assert run_episode(*args).transcript() == run_episode(*args).transcript()


def test_the_seed_changes_the_transcript():
    scenario, condition = SCENARIOS_BY_ID["consulting"], CONDITIONS_BY_ID["time_pressure"]
    transcripts = {run_episode(scenario, condition, seed, SimulatedAgent()).transcript() for seed in range(3)}
    assert len(transcripts) > 1


def test_the_agent_prompt_carries_the_episode_identity_and_history():
    scenario, condition = SCENARIOS_BY_ID["licensing"], CONDITIONS_BY_ID["authority"]
    episode = run_episode(scenario, condition, 2, SimulatedAgent(), rounds=2)
    prompt = build_agent_prompt(scenario, condition, 2, 1, episode.turns)
    assert "EPISODE: licensing-authority-2" in prompt
    assert "CONDITION: authority" in prompt
    assert scenario.opening in prompt


def test_pressure_conditions_break_the_mandate_more_often_than_control(episodes):
    rates = {}
    for condition in CONDITIONS:
        bucket = [e for e in episodes if e.condition_id == condition.id]
        rates[condition.id] = sum(e.rule_violation for e in bucket) / len(bucket)
    assert all(rates[c.id] > rates["control"] for c in CONDITIONS if not c.is_control)


def test_an_episode_round_trips_through_json(episodes_by_id):
    episode = episodes_by_id["licensing-incremental-0"]
    payload = json.loads(json.dumps(episode.to_dict()))
    assert payload["rule_violation"] is True
    assert payload["violations"] and payload["violations"][0]["evidence"]
    assert [t["speaker"] for t in payload["turns"]][:2] == ["counterparty", "agent"]
