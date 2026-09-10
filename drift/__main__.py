"""CLI: ``run`` the grid, ``judge`` the transcripts, ``analyse`` the result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .analysis import analyse
from .judge import Judgment, SimulatedJudge, judge_episodes, parse_judgment
from .llm import OllamaLLM, SimulatedAgent
from .negotiation import Episode, Turn, run_grid
from .report import to_csv, to_markdown
from .scenarios import SCENARIOS_BY_ID, episode_grid
from .mandate import Violation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drift", description=__doc__)
    parser.add_argument("--results", default="results/run", help="directory for episodes/judgments/report")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the 45-episode grid")
    run.add_argument("--simulated", action="store_true", help="use the offline behavioural stand-in")
    run.add_argument("--model", default="llama3.1:8b")
    run.add_argument("--host", default="http://localhost:11434")
    run.add_argument("--rounds", type=int, default=4)
    run.add_argument("--seeds", type=int, default=3)

    judge = sub.add_parser("judge", help="run the LLM judge over saved episodes")
    judge.add_argument("--simulated", action="store_true")
    judge.add_argument("--model", default="llama3.1:8b")
    judge.add_argument("--host", default="http://localhost:11434")

    analyse_cmd = sub.add_parser("analyse", help="aggregate and write the report")
    analyse_cmd.add_argument("--hand-labels", default="data/hand_labels.json")
    analyse_cmd.add_argument("--resamples", type=int, default=10000)

    sub.add_parser("quick", help="run + judge + analyse offline, in one go")
    return parser


def _episodes_path(results: Path) -> Path:
    return results / "episodes.jsonl"


def _judgments_path(results: Path) -> Path:
    return results / "judgments.jsonl"


def load_episodes(path: Path) -> List[Episode]:
    episodes: List[Episode] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        episode = Episode(
            id=payload["id"],
            scenario_id=payload["scenario"],
            condition_id=payload["condition"],
            seed=payload["seed"],
            agent_backend=payload.get("agent_backend", ""),
            turns=[Turn(t["round"], t["speaker"], t["text"]) for t in payload["turns"]],
            violations=[Violation(**v) for v in payload["violations"]],
        )
        episodes.append(episode)
    return episodes


def load_judgments(path: Path) -> List[Judgment]:
    out: List[Judgment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        out.append(
            Judgment(
                episode_id=payload["episode_id"],
                violation=payload["violation"],
                mandates_broken=payload.get("mandates_broken", []),
                quote=payload.get("quote", ""),
                confidence=payload.get("confidence", 0.0),
                parse_failure=payload.get("parse_failure", False),
                raw=payload.get("raw", ""),
            )
        )
    return out


def load_hand_labels(path: Path) -> Dict[str, bool]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: bool(entry["violation"]) for key, entry in payload.get("labels", {}).items()}


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_run(args, results: Path) -> int:
    agent = SimulatedAgent() if args.simulated else OllamaLLM(model=args.model, host=args.host)
    grid = episode_grid(seeds=tuple(range(args.seeds)))
    episodes = run_grid(agent, grid, rounds=args.rounds)
    _write_jsonl(_episodes_path(results), (e.to_dict() for e in episodes))
    flagged = sum(e.rule_violation for e in episodes)
    print(f"ran {len(episodes)} episodes with {agent.name}; {flagged} flagged by the rules")
    print(f"wrote {_episodes_path(results)}")
    return 0


def cmd_judge(args, results: Path) -> int:
    episodes = load_episodes(_episodes_path(results))
    judge = SimulatedJudge() if args.simulated else OllamaLLM(model=args.model, host=args.host)
    judgments = judge_episodes(episodes, SCENARIOS_BY_ID, judge)
    _write_jsonl(_judgments_path(results), (j.to_dict() for j in judgments))
    failures = sum(j.parse_failure for j in judgments)
    print(f"judged {len(judgments)} episodes; {failures} parse failures (recorded, not retried)")
    print(f"wrote {_judgments_path(results)}")
    return 0


def cmd_analyse(args, results: Path) -> int:
    episodes = load_episodes(_episodes_path(results))
    judgments = load_judgments(_judgments_path(results)) if _judgments_path(results).exists() else []
    hand_labels = load_hand_labels(Path(args.hand_labels))
    analysis = analyse(episodes, judgments, hand_labels, resamples=args.resamples)

    (results / "analysis.json").write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    (results / "report.md").write_text(to_markdown(analysis), encoding="utf-8")
    (results / "by_condition.csv").write_text(to_csv(analysis), encoding="utf-8")
    print(to_markdown(analysis))
    print(f"wrote {results/'report.md'}, {results/'analysis.json'}, {results/'by_condition.csv'}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results = Path(args.results)

    if args.command == "quick":
        quick = build_parser().parse_args(["--results", str(results), "run", "--simulated"])
        cmd_run(quick, results)
        cmd_judge(build_parser().parse_args(["--results", str(results), "judge", "--simulated"]), results)
        return cmd_analyse(build_parser().parse_args(["--results", str(results), "analyse"]), results)
    if args.command == "run":
        return cmd_run(args, results)
    if args.command == "judge":
        return cmd_judge(args, results)
    if args.command == "analyse":
        return cmd_analyse(args, results)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
