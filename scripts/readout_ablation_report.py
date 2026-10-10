"""Paired summary of the readout ablation (protocol: docs/READOUT_ABLATION_PROTOCOL.md)."""

import csv
import glob
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path("outputs/readout-ablation")
ARMS = ("newton_euler", "fragment_direct")
DRAWS = 10000


def bootstrap_diff(a, b, seed=20261010):
    """95% interval of mean(a) - mean(b) over paired items."""
    keys = sorted(a)
    rng = random.Random(seed)
    stats = []
    for _ in range(DRAWS):
        s = [rng.choice(keys) for _ in keys]
        stats.append(statistics.mean(a[k] for k in s) - statistics.mean(b[k] for k in s))
    stats.sort()
    return [stats[int(0.025 * DRAWS)], stats[int(0.975 * DRAWS) - 1]]


def external(arm, dataset):
    rows = {}
    for path in glob.glob(str(ROOT / arm / "external" / f"ro-{arm}-{dataset}-n100-s10*.csv")):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                rmsd = json.loads(row["candidate_rmsds_json"])
                assert len(rmsd) == 100
                rows[row["id"]] = rmsd
    return rows


def main():
    report = {"protocol": "docs/READOUT_ABLATION_PROTOCOL.md", "arms": {}, "paired": {}}
    val = {}
    for arm in ARMS:
        v = json.loads((ROOT / arm / "val_rollout.json").read_text())
        assert v["step"] == 100000 and v["n"] == 1076
        val[arm] = {r["index"]: r["rmsd"] for r in v["rows"]}
        report["arms"][arm] = {"val_success_2A": 100 * v["success_2A"], "val_median_rmsd": v["median"]}
    succ = {arm: {k: float(x < 2) for k, x in val[arm].items()} for arm in ARMS}
    report["paired"]["val_success_2A_pp"] = {
        "diff": 100 * (statistics.mean(succ[ARMS[0]].values()) - statistics.mean(succ[ARMS[1]].values())),
        "ci95": [100 * x for x in bootstrap_diff(succ[ARMS[0]], succ[ARMS[1]])]}
    for dataset, count in (("astex", 85), ("posebusters", 308)):
        data = {arm: external(arm, dataset) for arm in ARMS}
        for arm in ARMS:
            if len(data[arm]) != count:
                sys.exit(f"incomplete {arm} {dataset}: {len(data[arm])}/{count}")
        metrics = {}
        for arm in ARMS:
            d = data[arm]
            metrics[arm] = {
                "oracle100": {k: float(min(r) < 2) for k, r in d.items()},
                "oracle10": {k: float(min(r[:10]) < 2) for k, r in d.items()},
                "min_rmsd": {k: min(r) for k, r in d.items()},
                "candidate_lt2": {k: sum(x < 2 for x in r) / 100 for k, r in d.items()},
            }
            report["arms"][arm][dataset] = {
                "oracle100": 100 * statistics.mean(metrics[arm]["oracle100"].values()),
                "oracle10": 100 * statistics.mean(metrics[arm]["oracle10"].values()),
                "median_min_rmsd": statistics.median(metrics[arm]["min_rmsd"].values()),
                "candidate_lt2": 100 * statistics.mean(metrics[arm]["candidate_lt2"].values()),
            }
        for key, scale in (("oracle100", 100), ("oracle10", 100), ("candidate_lt2", 100)):
            a, b = metrics[ARMS[0]][key], metrics[ARMS[1]][key]
            report["paired"][f"{dataset}_{key}_pp"] = {
                "diff": scale * (statistics.mean(a.values()) - statistics.mean(b.values())),
                "ci95": [scale * x for x in bootstrap_diff(a, b)]}
    out = ROOT / "report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
