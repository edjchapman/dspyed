"""dspyed command-line interface.

Subcommands land phase by phase (see the project plan); each stub names the
phase that implements it, so `dspyed <cmd>` is honest about what exists today.
"""

from __future__ import annotations

import argparse
import sys

from dspyed import __version__

_PLANNED: dict[str, str] = {
    "download": "Phase 1 — fetch + validate the Spider dataset",
    "splits": "Phase 1 — build seeded, committed example-id splits",
    "eval": "Phase 2 — run an experiment config against an eval split",
    "compile": "Phase 4 — optimize a program and save the artifact",
    "report": "Phase 3 — regenerate figures + the README results table",
    "serve": "Phase 5 — run the FastAPI demo locally",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dspyed", description=__doc__)
    parser.add_argument("--version", action="version", version=f"dspyed {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    for name, help_text in _PLANNED.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0
    print(
        f"dspyed {args.command}: not implemented yet ({_PLANNED[args.command]})",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
