#!/usr/bin/env python3
"""Render admissible selected-pose cutoff-by-jitter robustness heatmaps.

The frozen report contains official PoseBusters results only for the
confidence-selected pose. It therefore does not permit a PB-valid oracle label.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    "outputs/benchmarks/effdock_pocket_prior_robustness_extension_runs/"
    "production-cutoff14-sigma-r3-20260906-v1/report.json"
)
JITTER0_ROOT = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/"
    "cutoff-r3-production-20260901-r2"
)
JITTER_ROOT = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_jitter_robustness_runs/"
    "production-jitter-r3-20260904-v3"
)
EXTENSION_ROOT = ROOT / (
    "outputs/benchmarks/effdock_pocket_prior_robustness_extension_runs/"
    "production-cutoff14-sigma-r3-20260906-v1"
)
OUTPUT_DIR = ROOT / "docs/results/current_overview/figures"
CSV_OUTPUT = ROOT / "docs/results/current_overview/pocket_cutoff_jitter_selected_metrics.csv"
CUTOFFS = (6, 8, 10, 12, 14)
JITTERS = (0, 1, 2)
REPORTED_METRICS = (
    ("selected_rmsd_lt2", "Top-1 RMSD < 2 Å", "top1"),
    ("joint_rmsd_lt2_pb_valid", "Top-1 RMSD < 2 Å & PB-valid", "top1_joint"),
)
TOP5_METRIC = ("top5_rmsd_lt2", "Top-5 RMSD < 2 Å", "top5")
METRICS = (*REPORTED_METRICS, TOP5_METRIC)
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
INK = "#29384D"
MUTED = "#66758A"


def load_report() -> dict[str, object]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_id") != "EFFDOCK-POCKET-PRIOR-ROBUSTNESS-EXTENSION-V1"
        or payload.get("total_selected_posebusters_errors") != 0
    ):
        raise ValueError("expected completed production cutoff-by-jitter report")
    return payload


def load_rows(payload: dict[str, object]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for dataset, dataset_rows in payload["datasets"].items():
        for cutoff_row in dataset_rows["cutoff_sweep_fixed_sigma_2"]:
            cutoff = int(cutoff_row["pocket_cutoff_angstrom"])
            for jitter_row in cutoff_row["jitters"]:
                jitter = int(jitter_row["center_jitter_sigma_angstrom_per_axis"])
                for metric, _, _ in REPORTED_METRICS:
                    aggregate = jitter_row["aggregate"][metric]
                    rows.append(
                        {
                            "dataset": dataset,
                            "cutoff_A": cutoff,
                            "jitter_sigma_A_per_axis": jitter,
                            "metric": metric,
                            "mean_pct": float(aggregate["mean"]),
                            "std_pct": float(aggregate["std"]),
                        }
                    )
    expected = 2 * len(CUTOFFS) * len(JITTERS) * len(REPORTED_METRICS)
    if len(rows) != expected:
        raise ValueError(f"incomplete report matrix: {len(rows)} != {expected}")
    return rows


def condition_root(cutoff: int, jitter: int, repeat: int) -> Path:
    if cutoff == 14:
        return (
            EXTENSION_ROOT
            / "cutoff_14"
            / "sigma_02"
            / f"jitter_{jitter:02d}"
            / f"repeat_{repeat}"
        )
    if jitter == 0:
        return JITTER0_ROOT / f"cutoff_{cutoff:02d}" / f"repeat_{repeat}"
    return JITTER_ROOT / f"jitter_{jitter:02d}" / f"cutoff_{cutoff:02d}" / f"repeat_{repeat}"


def top5_outcomes(scores_path: Path) -> tuple[bool, bool]:
    with scores_path.open(encoding="utf-8", newline="") as handle:
        scores = list(csv.DictReader(handle))
    if len(scores) != 100:
        raise ValueError(f"expected 100 score rows: {scores_path}")
    parsed: list[tuple[float, int, float]] = []
    for row in scores:
        try:
            confidence_rmsd = float(row["after_confidence_rmsd"])
            pose_index = int(row["pose_index"])
            final_rmsd = float(row["final_symmetry_rmsd_angstrom"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid score row: {scores_path}") from exc
        if not math.isfinite(confidence_rmsd) or not math.isfinite(final_rmsd):
            raise ValueError(f"non-finite score row: {scores_path}")
        parsed.append((confidence_rmsd, pose_index, final_rmsd))
    if {pose_index for _, pose_index, _ in parsed} != set(range(100)):
        raise ValueError(f"invalid pose-index coverage: {scores_path}")
    ranked = sorted(parsed)
    return ranked[0][2] < 2.0, min(final_rmsd for _, _, final_rmsd in ranked[:5]) < 2.0


def reported_top1_count(
    payload: dict[str, object], dataset: str, cutoff: int, jitter: int, repeat: int
) -> int:
    dataset_rows = payload["datasets"][dataset]["cutoff_sweep_fixed_sigma_2"]
    cutoff_row = next(row for row in dataset_rows if int(row["pocket_cutoff_angstrom"]) == cutoff)
    jitter_row = next(
        row
        for row in cutoff_row["jitters"]
        if int(row["center_jitter_sigma_angstrom_per_axis"]) == jitter
    )
    return int(jitter_row["repeats"][repeat]["selected_rmsd_lt2"]["count"])


def load_cached_top5_rows() -> list[dict[str, float | int | str]]:
    with CSV_OUTPUT.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        {
            "dataset": row["dataset"],
            "cutoff_A": int(row["cutoff_A"]),
            "jitter_sigma_A_per_axis": int(row["jitter_sigma_A_per_axis"]),
            "metric": row["metric"],
            "mean_pct": float(row["mean_pct"]),
            "std_pct": float(row["std_pct"]),
        }
        for row in source_rows
        if row["metric"] == TOP5_METRIC[0] and int(row["cutoff_A"]) < 14
    ]
    expected = 2 * (len(CUTOFFS) - 1) * len(JITTERS)
    if len(rows) != expected:
        raise ValueError(f"missing cached cutoff 6–12 Top-5 rows: {len(rows)} != {expected}")
    return rows


def load_top5_rows(payload: dict[str, object]) -> list[dict[str, float | int | str]]:
    metric, _, _ = TOP5_METRIC
    rows = load_cached_top5_rows()
    for cutoff in (14,):
        for jitter in JITTERS:
            for dataset, expected in EXPECTED_COUNTS.items():
                repeat_values: list[float] = []
                for repeat in range(3):
                    score_root = condition_root(cutoff, jitter, repeat) / "full" / "confidence_chunk20_fresh" / dataset
                    scores = sorted(score_root.glob("*/scores.csv"))
                    if len(scores) != expected:
                        raise ValueError(f"coverage mismatch: {score_root} ({len(scores)} != {expected})")
                    with ThreadPoolExecutor(max_workers=32) as executor:
                        outcomes = list(executor.map(top5_outcomes, scores))
                    top1_count = sum(top1 for top1, _ in outcomes)
                    expected_top1 = reported_top1_count(payload, dataset, cutoff, jitter, repeat)
                    if top1_count != expected_top1:
                        raise ValueError(
                            f"Top-1 report mismatch c{cutoff}/j{jitter}/r{repeat}/{dataset}: "
                            f"{top1_count} != {expected_top1}"
                        )
                    repeat_values.append(100.0 * sum(top5 for _, top5 in outcomes) / expected)
                rows.append(
                    {
                        "dataset": dataset,
                        "cutoff_A": cutoff,
                        "jitter_sigma_A_per_axis": jitter,
                        "metric": metric,
                        "mean_pct": statistics.mean(repeat_values),
                        "std_pct": statistics.stdev(repeat_values),
                    }
                )
    expected_rows = 2 * len(CUTOFFS) * len(JITTERS)
    if len(rows) != expected_rows:
        raise ValueError(f"incomplete Top-5 matrix: {len(rows)} != {expected_rows}")
    return rows


def matrix(rows: list[dict[str, float | int | str]], dataset: str, metric: str, key: str) -> np.ndarray:
    lookup = {
        (int(row["cutoff_A"]), int(row["jitter_sigma_A_per_axis"])): float(row[key])
        for row in rows
        if row["dataset"] == dataset and row["metric"] == metric
    }
    return np.asarray([[lookup[(cutoff, jitter)] for jitter in JITTERS] for cutoff in CUTOFFS])


def draw_panel(
    ax: plt.Axes,
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> plt.AxesImage:
    image = ax.imshow(means, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(JITTERS)), [str(value) for value in JITTERS])
    ax.set_yticks(np.arange(len(CUTOFFS)), [str(value) for value in CUTOFFS])
    ax.set_xlabel("Center jitter σ (Å per axis)", color=INK)
    ax.set_ylabel("Pocket cutoff (Å)", color=INK)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_xticks(np.arange(-0.5, len(JITTERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CUTOFFS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means[row, column]
            color = "white" if norm(value) >= 0.56 else INK
            ax.text(
                column,
                row,
                f"{value:.1f}\n± {stds[row, column]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    return image


def render(rows: list[dict[str, float | int | str]], metric: str, title: str, stem: str) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "effdock_selected_metrics", ("#F4C7C3", "#F7E2B6", "#B8DBC9", "#83A9D8")
    )
    norm = Normalize(vmin=60.0, vmax=100.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    image = draw_panel(
        axes[0], matrix(rows, "astex", metric, "mean_pct"), matrix(rows, "astex", metric, "std_pct"),
        "A  Astex Diverse (N=85)", cmap, norm,
    )
    draw_panel(
        axes[1], matrix(rows, "posebusters", metric, "mean_pct"), matrix(rows, "posebusters", metric, "std_pct"),
        "B  PoseBusters v2 (N=308)", cmap, norm,
    )
    fig.suptitle(
        title, x=0.5, y=0.985, ha="center", fontsize=15, fontweight="bold", color=INK,
    )
    fig.subplots_adjust(left=0.08, right=0.89, top=0.84, bottom=0.16, wspace=0.30)
    color_axis = fig.add_axes((0.92, 0.20, 0.018, 0.58))
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Success rate (%)", color=INK)
    colorbar.ax.tick_params(colors=MUTED)
    figure_path = OUTPUT_DIR / f"pocket_cutoff_jitter_{stem}_heatmap.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    payload = load_report()
    rows = [*load_rows(payload), *load_top5_rows(payload)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric, title, stem in METRICS:
        render(rows, metric, title, stem)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
