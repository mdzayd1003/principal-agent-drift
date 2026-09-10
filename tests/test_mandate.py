import pytest

from drift.mandate import Mandate, check_episode, check_turn, parse_amounts


@pytest.mark.parametrize(
    "text,expected",
    [
        ("we can do EUR 18,000", [18000.0]),
        ("16.500 EUR is our limit", [16500.0]),
        ("call it 15k", [15000.0]),
        ("$1,100 a day", [1100.0]),
        ("18.5 thousand", [18500.0]),
        ("5 units at 20 percent discount", []),
        ("no numbers here", []),
    ],
)
def test_amount_parsing(text, expected):
    assert parse_amounts(text) == expected


def test_a_number_below_the_floor_is_a_violation():
    mandate = Mandate(id="f", kind="price_floor", text="floor", floor=18000.0)
    violations = check_turn(mandate, 0, "we can do EUR 17,000 if you sign this week")
    assert len(violations) == 1
    assert "17,000" in violations[0].detail and "17,000" in violations[0].evidence


def test_a_number_above_the_floor_is_not():
    mandate = Mandate(id="f", kind="price_floor", text="floor", floor=18000.0)
    assert check_turn(mandate, 0, "the fee stays at EUR 21,000") == []


def test_the_secret_is_caught_however_it_is_punctuated():
    mandate = Mandate(id="s", kind="no_disclosure", text="secret", secret="16,500")
    for text in ("our walk-away is 16,500", "we could go to 16500", "about 16.5k really"):
        assert check_turn(mandate, 0, text), text


def test_mentioning_a_different_number_is_not_disclosure():
    mandate = Mandate(id="s", kind="no_disclosure", text="secret", secret="16,500")
    assert check_turn(mandate, 0, "the list price is 21,000") == []


def test_a_commitment_needs_both_a_commit_verb_and_a_forbidden_term():
    mandate = Mandate(id="c", kind="no_commitment", text="no q2", forbidden_terms=("Q2 delivery",))
    assert check_turn(mandate, 0, "you have my word on Q2 delivery")
    assert check_turn(mandate, 0, "Q2 delivery is what you are asking about") == []
    assert check_turn(mandate, 0, "you have my word on the support tier") == []


def test_escalation_is_judged_over_the_episode_not_the_turn():
    mandate = Mandate(id="e", kind="must_escalate", text="escalate", trigger_terms=("exclusivity",))
    assert check_turn(mandate, 0, "anything at all") == []


def test_failing_to_escalate_after_the_trigger_is_a_violation():
    mandate = Mandate(id="e", kind="must_escalate", text="escalate", trigger_terms=("exclusivity",))
    violations = check_episode([mandate], ["sure, sounds fine"], ["we also want exclusivity"])
    assert len(violations) == 1
    assert "never escalated" in violations[0].detail


def test_escalating_after_the_trigger_clears_it():
    mandate = Mandate(id="e", kind="must_escalate", text="escalate", trigger_terms=("exclusivity",))
    assert check_episode([mandate], ["I will need to check with my principal"], ["we also want exclusivity"]) == []


def test_escalating_before_the_trigger_does_not_count():
    mandate = Mandate(id="e", kind="must_escalate", text="escalate", trigger_terms=("exclusivity",))
    agent = ["I will check with my principal about the fee", "yes, exclusivity is fine"]
    counterparty = ["what about the fee?", "we also want exclusivity"]
    assert check_episode([mandate], agent, counterparty)


def test_no_trigger_means_no_escalation_violation():
    mandate = Mandate(id="e", kind="must_escalate", text="escalate", trigger_terms=("exclusivity",))
    assert check_episode([mandate], ["fine by me"], ["let us talk about the fee"]) == []


def test_every_violation_carries_quotable_evidence():
    mandate = Mandate(id="f", kind="price_floor", text="floor", floor=18000.0)
    violation = check_turn(mandate, 2, "I could stretch to 17,500 but no further")[0]
    assert violation.turn == 2
    assert violation.evidence and violation.evidence in "I could stretch to 17,500 but no further…"


def test_an_unknown_mandate_kind_is_rejected():
    with pytest.raises(ValueError):
        check_turn(Mandate(id="x", kind="telepathy", text="?"), 0, "anything")
