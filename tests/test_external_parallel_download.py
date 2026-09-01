import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.external_models.parallel_download import (  # noqa: E402
    MIN_PART_BYTES,
    download_range,
    split_ranges,
)


def test_split_ranges_is_contiguous_and_complete() -> None:
    start = 123
    end = 100 * 1024 * 1024 + 456
    ranges = split_ranges(start, end, workers=8)

    assert ranges[0][0] == start
    assert ranges[-1][1] == end
    assert all(left_end + 1 == right_start for (_, left_end), (right_start, _) in zip(ranges, ranges[1:]))
    assert sum(range_end - range_start + 1 for range_start, range_end in ranges) == end - start + 1
    assert len(ranges) == 8


def test_split_ranges_does_not_create_tiny_parts() -> None:
    ranges = split_ranges(0, MIN_PART_BYTES - 2, workers=8)
    assert ranges == [(0, MIN_PART_BYTES - 2)]


def test_split_ranges_handles_completed_download() -> None:
    assert split_ranges(10, 9, workers=8) == []


def test_download_range_preserves_progress_across_disconnects(tmp_path, monkeypatch) -> None:
    requested_ranges: list[str] = []

    def fake_run(arguments, check):
        assert check is False
        requested = arguments[arguments.index("--range") + 1]
        output = Path(arguments[arguments.index("--output") + 1])
        request_start, request_end = (int(value) for value in requested.split("-"))
        received = min(3, request_end - request_start + 1)
        output.write_bytes(b"x" * received)
        requested_ranges.append(requested)
        return SimpleNamespace(returncode=0 if request_start + received - 1 == request_end else 18)

    monkeypatch.setattr("subprocess.run", fake_run)
    part = tmp_path / "part"
    download_range("https://example.invalid/artifact", part, 10, 19)

    assert part.read_bytes() == b"x" * 10
    assert requested_ranges == ["10-19", "13-19", "16-19", "19-19"]
