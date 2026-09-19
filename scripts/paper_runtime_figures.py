"""Observed throughput and allocator memory, never inferred sampling latency."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper/20260919"
POSE_COUNTS = {
    "unguided_n100_s10": 100,
    "unguided_n40_s25": 40,
    "guided_n40_s25": 40,
    "temporal_n100_s10": 100,
}


def normalize_cost(entry):
    """Amortize measured batch costs; two confidence banks contain 2N evaluations."""
    n = POSE_COUNTS[entry["arm"]]
    result = dict(entry, poses_per_complex=n, confidence_pose_evaluations_per_complex=2 * n)
    for source, target, denominator in (
        ("pipeline_mean_s", "pipeline_amortized_s_per_generated_pose", n),
        ("pipeline_repeat_sd_s", "pipeline_amortized_repeat_sd_s_per_generated_pose", n),
        ("refinement_mean_s", "refinement_amortized_s_per_pose", n),
        ("confidence_both_banks_mean_s", "confidence_amortized_s_per_pose_evaluation", 2 * n),
        ("refinement_compute_mean_s", "minimization_amortized_s_per_pose", n),
        (
            "confidence_forward_both_mean_s",
            "confidence_forward_amortized_s_per_pose_evaluation",
            2 * n,
        ),
    ):
        result[target] = entry[source] / denominator
    return result


def summaries(data):
    result = []
    keys = sorted({(r["arm"], r["dataset"]) for r in data["rows"]})
    for arm, dataset in keys:
        rows = [r for r in data["rows"] if (r["arm"], r["dataset"]) == (arm, dataset)]
        by = {
            s: sorted([r for r in rows if r["stage"] == s], key=lambda r: r["repeat"])
            for s in ("refinement", "confidence", "sampling", "full_shard")
        }
        if any([r["repeat"] for r in rs] != [0, 1, 2] for rs in by.values()):
            raise ValueError("missing runtime repeat")
        counts = {r["complexes"] for r in rows}
        if len(counts) != 1 or next(iter(counts)) <= 0:
            raise ValueError("runtime complex count mismatch")
        times = [r["aggregate_shard_wall_seconds_per_complex"] for r in by["full_shard"]]
        entry = dict(
            arm=arm,
            dataset=dataset,
            complexes_per_repeat=next(iter(counts)),
            repeats=3,
            pipeline_mean_s=statistics.mean(times),
            pipeline_repeat_sd_s=statistics.stdev(times),
            refinement_mean_s=statistics.mean(
                r["metrics"]["wall_seconds_before_summary_write"]["mean"] for r in by["refinement"]
            ),
            confidence_both_banks_mean_s=statistics.mean(
                r["metrics"]["wall_seconds_before_summary_write"]["mean"] for r in by["confidence"]
            ),
            refinement_compute_mean_s=statistics.mean(
                r["metrics"]["stage_seconds.minimization_refinement"]["mean"]
                for r in by["refinement"]
            ),
            confidence_forward_both_mean_s=statistics.mean(
                r["metrics"]["stage_seconds.confidence_forward_total"]["mean"]
                for r in by["confidence"]
            ),
            sampling_allocator_peak_gib=max(
                r["metrics"]["cuda_max_memory_allocated_bytes"]["max"] for r in by["sampling"]
            )
            / 2**30,
            sampling_reserved_peak_gib=max(
                r["metrics"]["cuda_max_memory_reserved_bytes"]["max"] for r in by["sampling"]
            )
            / 2**30,
            devices=sorted({name for r in by["sampling"] for name in r["devices"]}),
        )
        result.append(normalize_cost(entry))
    return result


def main():
    data = json.loads((OUT / "runtime_metrics.json").read_text())
    rows = summaries(data)
    (OUT / "runtime_comparison.json").write_text(json.dumps(rows, indent=2))
    with (OUT / "runtime_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dict(r, devices="; ".join(r["devices"])) for r in rows)
    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for j, (arm, label, color) in enumerate(
        [("unguided_n100_s10", "N100/S10", "#0072B2"), ("unguided_n40_s25", "N40/S25", "#E69F00")]
    ):
        rs = [
            next(r for r in rows if r["arm"] == arm and r["dataset"] == d)
            for d in ("astex", "posebusters")
        ]
        x = np.arange(2) + (j - 0.5) * 0.32
        bars = axes[0].bar(
            x,
            [r["pipeline_mean_s"] for r in rs],
            width=0.30,
            yerr=[r["pipeline_repeat_sd_s"] for r in rs],
            capsize=3,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, labels=[f"{r['pipeline_mean_s']:.1f}" for r in rs], padding=5)
        bars = axes[1].bar(
            x, [r["sampling_allocator_peak_gib"] for r in rs], width=0.30, label=label, color=color
        )
        axes[1].bar_label(
            bars, labels=[f"{r['sampling_allocator_peak_gib']:.1f}" for r in rs], padding=5
        )
    axes[0].set_ylabel("Summed shard wall time / complex (s)")
    axes[0].set_title("Unguided generation + both-bank analysis")
    axes[1].set_ylabel("Peak sampling CUDA allocated memory (GiB)")
    axes[1].set_title("Measured PyTorch allocator peak")
    for ax in axes:
        ax.set_xticks([0, 1], ["Astex", "PoseBusters"])
        ax.margins(y=0.2)
        ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"05_runtime_memory.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# Observed runtime and memory",
        "",
        "| Dataset | Arm | Summed shard wall s/complex ± repeat SD | Refinement wall s/complex | Confidence wall, both banks s/complex | Sampling allocated peak GiB | Sampling reserved peak GiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['pipeline_mean_s']:.2f} ± {r['pipeline_repeat_sd_s']:.2f} | {r['refinement_mean_s']:.2f} | {r['confidence_both_banks_mean_s']:.2f} | {r['sampling_allocator_peak_gib']:.2f} | {r['sampling_reserved_peak_gib']:.2f} |"
        )
    lines += [
        "",
        "## Amortized cost per generated pose / scored pose",
        "",
        "These are batch-cost normalizations, **not measured single-pose latencies**. "
        "Pipeline and refinement divide by N generated poses; confidence divides by 2N evaluations "
        "because both raw and refined banks were scored. Raw/refined are two versions of the same N samples, not 2N independent generated samples.",
        "",
        "| Dataset | Arm | N | Pipeline s/generated pose ± repeat SD | Refinement wall s/pose | Minimization compute s/pose | Confidence wall s/evaluated pose | Confidence forward s/evaluated pose |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['poses_per_complex']} | "
            f"{r['pipeline_amortized_s_per_generated_pose']:.3f} ± {r['pipeline_amortized_repeat_sd_s_per_generated_pose']:.3f} | "
            f"{r['refinement_amortized_s_per_pose']:.3f} | {r['minimization_amortized_s_per_pose']:.3f} | "
            f"{r['confidence_amortized_s_per_pose_evaluation']:.3f} | {r['confidence_forward_amortized_s_per_pose_evaluation']:.3f} |"
        )
    lines += [
        "",
        "For repeat r with C complexes and shard wall durations t_j: "
        "complex cost = sum(t_j)/C; generated-pose cost = sum(t_j)/(C·N). "
        "Means and sample SD are taken across three repeat costs. Stage wall includes that stage's setup/I/O; "
        "compute-only columns use the recorded minimization/forward timers. Neither includes official PB evaluation.",
        "",
        "GPU memory is reported only as the observed peak per sampling process/shard. "
        "It is **not divided by N or C**: model/workspace memory is shared, and per-pose memory cannot be recovered by division.",
        "",
        "## Coverage and hardware",
        "",
        "| Dataset | Arm | Complexes/repeat | Repeats | Generated poses across repeats | Scored raw+refined poses across repeats | Sampling devices |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        generated = r["complexes_per_repeat"] * r["repeats"] * r["poses_per_complex"]
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['complexes_per_repeat']} | 3 | {generated:,} | {2 * generated:,} | {'; '.join(r['devices'])} |"
        )
    lines += ["", "## Interpretation and measurement boundaries", ""]
    for d in ("astex", "posebusters"):
        a = next(r for r in rows if r["arm"] == "unguided_n100_s10" and r["dataset"] == d)
        b = next(r for r in rows if r["arm"] == "unguided_n40_s25" and r["dataset"] == d)
        lines.append(
            f"- {d}: N100/S10 used {a['pipeline_mean_s'] / b['pipeline_mean_s']:.2f}× summed shard wall time per complex and {a['sampling_allocator_peak_gib'] / b['sampling_allocator_peak_gib']:.2f}× peak allocated sampling memory versus N40/S25."
        )
    lines += [
        "",
        "These are descriptive saved-run measurements on the recorded devices, not an isolated, randomized throughput benchmark. "
        "The compared Astex/PB runs used RTX 6000 Ada GPUs and the same frozen host-matched runtime. "
        "OpenBind includes both RTX 6000 Ada and H100 PCIe devices; its aggregate is not a hardware-controlled cross-dataset speed comparison. "
        "N100/S10 has 2.5× as many poses to refine and score; N*S equality covers only learned pose steps.",
        "",
        "Pipeline time is sum of completed shard wall times divided by completed complexes within each repeat, "
        "then mean/sample SD across three repeats. It includes subprocess/model loading, both raw/refined confidence, "
        "I/O and driver analysis overhead, but not the separate official PB jobs. Shard elapsed times are summed across concurrently running shards; "
        "this is not user-perceived campaign wall-clock elapsed time or pure sampling latency. Sampling-only wall time was not recorded.",
        "",
        "Stage wall means average per-complex records equally. Confidence totals score both raw and refined banks. "
        "GPU memory is the maximum of recorded sampling-process PyTorch allocator peaks over all shards and repeats, "
        "reported in GiB (2^30 bytes). Reserved memory is not allocated memory. Host requested RAM is not GPU usage; "
        "no host-RSS or refinement/scoring GPU-memory peak is claimed.",
        "",
        "**Figure 5.** Summed unguided shard wall cost per complex (left; mean ± three-repeat sample SD) and maximum "
        "recorded sampling allocator memory (right; no SD/error bar). See measurement definitions above.",
        "",
        "![Observed runtime and memory](05_runtime_memory.png)",
        "",
        "Reproduce with `scripts/collect_paper_runtime.py` then `scripts/paper_runtime_figures.py`. "
        "Detailed measured stage fields and source manifest digest are in `runtime_metrics.json`.",
        "The unit-explicit aggregate is available as [JSON](runtime_comparison.json) and [CSV](runtime_comparison.csv).",
        "",
    ]
    (OUT / "RUNTIME.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
