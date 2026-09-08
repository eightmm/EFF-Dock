"""Recover failed generation shards and reconnect the frozen evaluation chain."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = 'production-jitter-r3-20260904-v3'
OUT = ROOT / 'outputs/benchmarks/effdock_pocket_cutoff_jitter_robustness_runs' / RUN
CAP = ROOT / '.effdock_execution_capsules/EFFDOCK-POCKET-CUTOFF-JITTER-ROBUSTNESS-V1' / RUN
BASE = ROOT / 'outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/cutoff-r3-production-20260901-r2'
RECORD = OUT / 'recovery-65594-submission.json'


def main():
    if RECORD.exists():
        raise RuntimeError(f'Recovery already recorded: {RECORD}')
    result = subprocess.check_output(['sacct', '-j', '65594', '-X', '-n', '-P', '--format=JobID%40,State'], text=True)
    failed = sorted(int(line.split('|')[0].split('_')[1]) for line in result.splitlines()
                    if '|FAILED' in line and '.' not in line.split('|')[0])
    assert len(failed) == 25, failed
    for task in failed:
        c = (6, 8, 10, 12)[task // 96]
        j = (1, 2)[(task // 48) % 2]
        r = (task // 16) % 3
        ds = ('astex', 'posebusters')[(task // 8) % 2]
        raw = OUT / f'jitter_{j:02}/cutoff_{c:02}/repeat_{r}/raw'
        stem = f'effdock-pocket-cutoff-v1-{ds}-c{c:02}-r{r}-j{j:02}-n100-s10.shard-{task%8:03}-of-008'
        archive = OUT / 'recovery_archive/65594' / str(task)
        archive.mkdir(parents=True, exist_ok=True)
        for suffix in ('.csv', '.summary.json'):
            src = raw / (stem + suffix)
            if src.exists():
                shutil.copy2(src, archive / src.name)
    env = os.environ.copy()
    env.update(EFFDOCK_REPO_DIR=str(CAP), EFFDOCK_RUNTIME_VENV=str(ROOT / '.venv'),
               PYTHONPATH=str(CAP / 'src'), PYTHONDONTWRITEBYTECODE='1',
               EFFDOCK_OUTPUT_ROOT=str(OUT), EFFDOCK_JITTER0_ROOT=str(BASE))
    jobs = {}
    def submit(name, script, partition, array=None, dependency=None, stage=None):
        command = ['sbatch', '--parsable', '--partition=' + partition, '--export=ALL']
        if array:
            command += ['--array=' + array]
        if dependency:
            command += ['--dependency=afterok:' + dependency]
        e = env.copy()
        if stage:
            e['EFFDOCK_STAGE'] = stage
        job = subprocess.check_output(command + [str(script)], cwd=ROOT, env=e, text=True).strip().split(';')[0]
        assert job.isdigit(), job
        jobs[name] = job
        RECORD.write_text(json.dumps({'status': 'submitting', 'failed_tasks': failed, 'jobs': jobs}, indent=2))
        print(name, job, flush=True)
        return job
    scripts = CAP / 'benchmarks/effdock/slurm'
    g = submit('generation', ROOT / 'benchmarks/effdock/slurm/pocket_cutoff_jitter_generation_recovery.sbatch', '6000ada,heavy', ','.join(map(str, failed)) + '%8')
    m = submit('manifest', scripts / 'pocket_cutoff_jitter_manifest.sbatch', 'cpu_only', '0-23%12', g)
    r = submit('refinement', scripts / 'pocket_cutoff_jitter_refine_confidence.sbatch', '6000ada,heavy', '0-767%12', m, 'refinement')
    c = submit('confidence', scripts / 'pocket_cutoff_jitter_refine_confidence.sbatch', '6000ada,heavy', '0-767%12', r, 'confidence')
    p = submit('posebusters', scripts / 'pocket_cutoff_jitter_selected_posebusters.sbatch', 'cpu_only', '0-383%12', c)
    submit('report', scripts / 'pocket_cutoff_jitter_report.sbatch', 'cpu_only', dependency=p)
    RECORD.write_text(json.dumps({'status': 'submitted', 'failed_tasks': failed, 'jobs': jobs}, indent=2))


if __name__ == '__main__':
    main()
