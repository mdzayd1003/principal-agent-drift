# Smoke output

Produced by `make quick`: the full 45-episode grid run against the **simulated**
agent (`drift/llm.py::SimulatedAgent`), judged by the **simulated** judge, and
analysed.

The simulated agent is a stipulated behavioural model with a per-condition
concession propensity. Its drift rates are inputs, not measurements — they are
here so the rules, the judge-validation path and the statistics are exercised on
every run. Nothing in `report.md` is a claim about any real model.

What the numbers *do* demonstrate: the harness runs end to end, the Wilson
intervals behave at 0/9 and 9/9, Holm correction is applied across the four
comparisons against control, the judge's parse failures are counted and excluded
rather than retried, and κ against the rules comes with a bootstrap interval.

Replace with a real run:

```
python -m drift --results results/run run --model llama3.1:8b
python -m drift --results results/run judge --model llama3.1:8b
python -m drift --results results/run analyse
```
