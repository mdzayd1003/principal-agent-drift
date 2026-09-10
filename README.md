# 02 — Principal-Agent Drift in LLM Negotiation

A principal gives an agent a binding mandate. A counterparty pushes against it.
Does the agent hold?

This is an evaluation harness for that question: 45 controlled negotiation
episodes across five pressure conditions, mandate violations detected by rules
that always attach quotable evidence, and an LLM judge whose agreement with
those rules is measured rather than assumed.

```
python -m pytest -q     # 79 tests, offline, ~1 s
make quick              # full 45-episode grid with no model at all
```

## The design

**Four kinds of mandate**, chosen so a violation is attributable rather than
merely present:

| Kind | Broken by | Detected by |
|---|---|---|
| `price_floor` | naming a number below the floor | amount extraction, compared to the floor |
| `no_disclosure` | repeating a figure the principal withheld | the secret appearing in any punctuation form |
| `no_commitment` | binding the principal to a forbidden term | a commitment verb *and* the forbidden term |
| `must_escalate` | staying silent when something out of scope is raised | the trigger raised, no escalation after it |

The last one is an **omission**: the agent fails to do something rather than
doing something wrong. It is the class most likely to be missed by a judge, and
keeping it in the design is what makes that visible.

**Five conditions** (`scenarios.py`) — control, time pressure, authority appeal,
reciprocity, incremental commitment — × **three scenarios** × **three seeds** =
45 episodes.

**The counterparty is scripted, not modelled.** If both sides were LLMs, a
difference between conditions could come from the counterparty behaving
differently rather than from the pressure. Scripted tactics make the
manipulation identical across conditions, scenarios and models, so the agent
under test is the only thing that varies.

## Three decisions worth arguing about

**Parse failures are recorded, never retried.** A judge verdict that does not
parse is stored with the raw text, and that episode is dropped from the
agreement statistics. Retrying until the model emits valid JSON would erase the
measurement — how often it can hold a format over a long transcript is part of
what the judge is being evaluated on. The count appears in every report.

**Rules first, judge second — and the two are compared.** The rules are cheap,
deterministic and quotable; the judge is none of those things. Reporting only
the judge means trusting it, and reporting only the rules means missing whatever
they were not written to catch. Reporting both, with κ and an exact McNemar test
between them, says which one is doing the work.

**No asymptotic tests at n=9.** Proportions get Wilson intervals, because a Wald
interval at 9/9 has zero width and several conditions land exactly there.
Differences get a two-sided permutation test, Holm-corrected across the four
comparisons with control. κ gets a percentile bootstrap over episode indices.

## Validating the judge

`data/hand_labels.json` holds hand labels for a stratified sample of 15 episodes
(one per scenario × condition). They were produced by reading every distinct
agent utterance in the transcript bank and, separately, reading whether the agent
escalated after the counterparty raised an out-of-scope term. Each label carries
the note that justifies it.

The analysis reports three things from that: κ between judge and hand labels,
κ between rules and hand labels (the rules' own validity), and the judge's recall
split by violation kind — because a judge can look adequate overall while being
blind to one whole class.

## Running it

```bash
pip install -r requirements.txt      # only pytest; the harness is stdlib-only

# offline: run, judge and analyse in one go
make quick

# against a real local model
ollama pull llama3.1:8b
python -m drift --results results/run run   --model llama3.1:8b
python -m drift --results results/run judge --model llama3.1:8b
python -m drift --results results/run analyse
```

`run` writes `episodes.jsonl` (full transcripts and every violation with its
evidence), `judge` writes `judgments.jsonl`, `analyse` writes `report.md`,
`analysis.json` and `by_condition.csv`.

The agent under test and the judge are separate `--model` flags on purpose: using
the same model for both is a known way to manufacture agreement.

## What is in `results/smoke/`

The output of `make quick` — a full run against the **simulated** agent, whose
concession probabilities are stipulated in `llm.py`, not measured. It proves the
pipeline runs and that the statistics behave at the boundaries. It says nothing
about any real model, and `results/smoke/README.md` repeats that where someone
skimming the numbers will see it.

## Layout

```
drift/
├── mandate.py       four mandate kinds, their rules, and the evidence they attach
├── scenarios.py     three scenarios, five conditions, the crossed grid
├── negotiation.py   the episode loop and the scripted counterparty
├── llm.py           Ollama client + the simulated agent (an explicit drift model)
├── judge.py         the LLM judge, its contract, and the offline stand-in
├── stats.py         Wilson, bootstrap, Cohen's κ, permutation, Holm, McNemar
├── analysis.py      drift by condition, comparisons, judge agreement by kind
├── report.py        markdown + CSV
└── __main__.py      run / judge / analyse / quick
data/hand_labels.json
```

## Limits

- 45 episodes is enough to separate "always" from "rarely" and nothing finer.
  With 9 per condition, a 20-point difference is not detectable and the intervals
  say so.
- Four rounds per episode. Real drift may take longer to appear; the round
  budget is a flag, but the reported grid uses four.
- The rules detect what they were written to detect. That is why they are
  validated against hand labels rather than treated as ground truth, and why the
  judge is kept in the design instead of being replaced by them.
- Three scenarios in one register (B2B commercial negotiation). Nothing here
  says how an agent behaves under a different kind of pressure.

---

One of seven projects. Index: [mdzayd1003/zaid-portfolio-projects](https://github.com/mdzayd1003/zaid-portfolio-projects)
