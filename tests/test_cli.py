from __future__ import annotations

import pytest

from effdock.cli import main


def test_top_level_help(capsys) -> None:
    main(["--help"])
    assert "eff-dock <command>" in capsys.readouterr().out


def test_unknown_command_fails() -> None:
    with pytest.raises(SystemExit, match="unknown command"):
        main(["unknown"])


def test_physical_nested_help(capsys) -> None:
    main(["physical", "--help"])
    assert "eff-dock physical {trace}" in capsys.readouterr().out
