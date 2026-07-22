# dspyed

*Compiled, not prompted — a measured text-to-SQL pipeline on [Spider](https://yale-lily.github.io/spider), built with [DSPy](https://dspy.ai).*

> **Status: under construction.** Bootstrap + tooling are in place; the pipeline,
> eval harness, and deployed demo land phase by phase. Results tables and figures
> will appear here once the experiment matrix runs.

Natural-language question → SQL → executed against SQLite → answer, measured by
execution accuracy against a committed train/dev split, with DSPy optimizers
compiling the pipeline — and every reported number backed by a committed
results JSON.

## Quick start

```sh
uv sync            # install runtime + dev dependencies
make check         # the full local gate: links, anchors, lint, types, tests
uv run dspyed      # CLI — subcommands land per phase
```

## Development

`make check` is the single quality gate — CI (`check.yml`), the pre-commit hook,
and the weekly scheduled run all call it. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the branch → PR → squash flow and the strict commit style.

## License

MIT. The Spider dataset is © its authors, licensed CC BY-SA 4.0
(Yu et al., 2018) — an attribution notice ships with any redistributed demo
databases.
