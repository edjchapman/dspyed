"""CLI skeleton tests — argument wiring, not behaviour (stubs land per phase)."""

import pytest

from dspyed import __version__
from dspyed.cli import main


def test_version_is_semver() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_no_command_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "download" in capsys.readouterr().out


def test_stub_command_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report"]) == 2
    assert "not implemented" in capsys.readouterr().err
