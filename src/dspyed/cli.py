"""dspyed command-line interface.

Subcommands land phase by phase (see the project plan); implemented commands
do real work, the rest are stubs that name the phase implementing them — so
`dspyed <cmd>` is honest about what exists today.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dspyed import __version__

_STUBS: dict[str, str] = {
    "compile": "Phase 4 — optimize a program and save the artifact",
    "report": "Phase 3 — regenerate figures + the README results table",
    "serve": "Phase 5 — run the FastAPI demo locally",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dspyed", description=__doc__)
    parser.add_argument("--version", action="version", version=f"dspyed {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    download = subparsers.add_parser("download", help="fetch + validate the Spider dataset")
    download.add_argument("--root", type=Path, default=Path("data/spider"))
    download.add_argument("--force", action="store_true", help="ignore an existing MANIFEST.json")

    splits = subparsers.add_parser("splits", help="build seeded, committed example-id splits")
    splits.add_argument("--root", type=Path, default=Path("data/spider"))
    splits.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    splits.add_argument("--seed", type=int, default=13)

    evaluate = subparsers.add_parser("eval", help="run a program over a split; write results JSON")
    evaluate.add_argument("--experiment", required=True, help="results file name, e.g. E01-smoke")
    evaluate.add_argument("--program", required=True, choices=("p0", "p1", "p2", "p3"))
    evaluate.add_argument("--model", default="small", choices=("small", "large"))
    evaluate.add_argument("--split", default="dev_eval_200")
    evaluate.add_argument("--limit", type=int, default=None, help="cap examples (smoke runs)")

    for name, help_text in _STUBS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def _cmd_download(args: argparse.Namespace) -> int:
    from dspyed.data.spider import download  # lazy: pulls network-side deps

    manifest = download(args.root, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_splits(args: argparse.Namespace) -> int:
    from dspyed.data.splits import build_splits

    summary = build_splits(args.root, args.splits_dir, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from dspyed.config import Settings
    from dspyed.eval.harness import RunSpec, run_eval

    spec = RunSpec(
        experiment_id=args.experiment,
        program=args.program,
        model=args.model,
        split=args.split,
        limit=args.limit,
    )
    results = run_eval(spec, Settings())
    print(json.dumps(results["summary"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()  # repo-local .env (gitignored): ANTHROPIC_API_KEY etc. for LM calls
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0
    if args.command == "download":
        return _cmd_download(args)
    if args.command == "splits":
        return _cmd_splits(args)
    if args.command == "eval":
        return _cmd_eval(args)
    print(
        f"dspyed {args.command}: not implemented yet ({_STUBS[args.command]})",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
