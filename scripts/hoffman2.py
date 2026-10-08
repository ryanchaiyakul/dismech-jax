"""Headless CLI for running batch jobs on the UCLA Hoffman2 cluster.

Authenticates with an SSH key (no password stored). The only setting read is
the cluster username, from a gitignored ``.secrets`` file in the repo root:

    H2_USER=<username>

Usage:

    uv run scripts/hoffman2.py check
    uv run scripts/hoffman2.py submit --params params.yaml --wait --fetch
    uv run scripts/hoffman2.py submit --config simulation_parameters_random_1b6.yaml
    uv run scripts/hoffman2.py status
    uv run scripts/hoffman2.py log <job_id>
    uv run scripts/hoffman2.py fetch datafiles
    uv run scripts/hoffman2.py run "ls datafiles"
    uv run scripts/hoffman2.py shell
    uv run scripts/hoffman2.py dataset launch <name> --jobs 8 --paths 20
    uv run scripts/hoffman2.py dataset collect <name> --watch
"""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / ".secrets"
HOST = "hoffman2.idre.ucla.edu"
REMOTE_DIR = "dev/RibbonFEMData"  # relative to the remote home directory
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=5",
]


def load_user() -> str:
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "H2_USER" and v.strip():
                return v.strip().strip("'\"")
    sys.exit(f"H2_USER not set in {SECRETS.name}")


def ssh(user: str, cmd: str, capture=False, tty=False) -> subprocess.CompletedProcess:
    full = f"cd {shlex.quote(REMOTE_DIR)} && {cmd}"
    args = ["ssh", *SSH_OPTS, *(["-t"] if tty else []), f"{user}@{HOST}", full]
    return subprocess.run(args, capture_output=capture, text=True)


def scp(src: str, dst: str, recursive=False) -> None:
    args = ["scp", *SSH_OPTS, *(["-r"] if recursive else []), src, dst]
    subprocess.run(args, check=True)


def remote(user: str, path: str) -> str:
    return f"{user}@{HOST}:{REMOTE_DIR}/{path}"


def qsub(user: str, script: str, env: list[str], name=None, resources=None) -> str | None:
    """Submit `script` with `qsub -v env`; returns the job id (None on failure)."""
    cmd = ["qsub", "-v", ",".join(env)]
    if name:
        cmd += ["-N", name]
    if resources:
        cmd += ["-l", resources]
    r = ssh(user, shlex.join([*cmd, script]), capture=True)
    print(r.stdout or r.stderr, end="")
    m = re.search(r"Your job (\d+)", r.stdout)
    return m.group(1) if r.returncode == 0 and m else None


def active_jobs(user: str) -> set[str]:
    r = ssh(user, f"qstat -u {shlex.quote(user)}", capture=True)
    return {ln.split()[0] for ln in r.stdout.splitlines()[2:] if ln.strip()}


def job_running(user: str, job_id: str) -> bool:
    # qstat -j exits nonzero once the job has left the queue.
    return ssh(user, f"qstat -j {job_id} >/dev/null 2>&1", capture=True).returncode == 0


def cmd_check(user, _args):
    r = ssh(user, "hostname && which qsub", capture=True)
    print(r.stdout or r.stderr, end="")
    return r.returncode


def cmd_run(user, args):
    return ssh(user, " ".join(args.command)).returncode


def cmd_shell(user, _args):
    return ssh(user, "exec $SHELL -l", tty=True).returncode


def cmd_status(user, _args):
    return ssh(user, f"qstat -u {shlex.quote(user)}").returncode


def cmd_log(user, args):
    return ssh(user, f"cat joblog.{args.job_id}").returncode


def cmd_fetch(user, args):
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    scp(remote(user, args.path), str(dest), recursive=True)
    print(f"fetched {args.path} -> {dest}")
    return 0


def cmd_submit(user, args):
    config = args.config
    if args.params:
        config = Path(args.params).name
        scp(args.params, remote(user, config))
        print(f"uploaded {args.params} -> {config}")

    # Same env-var interface as the upstream launch_*.sh scripts.
    output = args.output or f"datafiles_{time.strftime('%Y%m%d_%H%M%S')}_cli"
    print(f"config={config}  output={output}")
    job_id = qsub(user, args.script, [f"RANDOM_CONFIG={config}", f"OUTPUT_PARENT={output}", *args.env],
                  args.name, args.resources)
    if job_id is None:
        return 1

    if not args.wait:
        return 0
    print(f"waiting for job {job_id} (poll every {args.poll}s)...")
    while job_running(user, job_id):
        time.sleep(args.poll)
    print(f"job {job_id} finished; joblog:")
    cmd_log(user, argparse.Namespace(job_id=job_id))

    if args.fetch:
        return cmd_fetch(user, argparse.Namespace(path=output, dest=args.dest))
    return 0


# Geometry/material of upstream's simulation_parameters_random_1b6.yaml (W/L = 1/6).
DATASET_BASE = {
    "l": 0.1, "w": 0.017303187567613086, "thickness": 0.001, "maxMesh": 0.00333,
    "young": 1.0e7, "poisson": 0.5, "uniform_mesh": "true", "clamp_fraction": 0.1,
}
SEEDS_PER_JOB = 10000  # > paths per job, so seed ranges of different jobs never overlap


def cmd_dataset_launch(user, args):
    """Submit `jobs` independent jobs, each running `paths` random homotopy paths."""
    out = ROOT / "data" / args.name
    launch_file = out / "launch.json"
    launch = json.loads(launch_file.read_text()) if launch_file.exists() else None
    if launch and not args.append:
        sys.exit(f"{launch_file} exists; pass --append to add jobs to this dataset")
    # The physical config must match across batches; paths per job may differ.
    cfg = {**DATASET_BASE, "path_frames": args.frames, "path_hold_time": args.hold}
    if launch:
        old = {k: v for k, v in launch["config"].items() if k != "nSimulations"}
        if old != cfg:
            sys.exit(f"--append needs the same config as the dataset: {old}")
    out.mkdir(parents=True, exist_ok=True)
    jobs = launch["jobs"] if launch else []
    for j in jobs:  # launch.json from before paths were stored per job
        j.setdefault("paths", launch["config"].get("nSimulations"))
    start = len(jobs)
    cfg_name = f"dataset_{args.name}_b{start:03d}.yaml"  # one per batch: nSimulations differs
    (out / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()))
    (out / cfg_name).write_text("".join(f"{k}: {v}\n" for k, v in {**cfg, "nSimulations": args.paths}.items()))
    scp(str(out / cfg_name), remote(user, cfg_name))

    # measured 4-18 min (mean ~14 incl. overhead) per random path at 1 s holds; budget 16
    hours = max(1, -(-args.paths * 16 // 60))
    resources = f"h_rt={hours}:00:00,h_data=4G,h_vmem=32G"
    for i in range(start, start + args.jobs):
        label = f"j{i:03d}"
        job_id = qsub(user, "submit_random.sh",
                      [f"RANDOM_CONFIG={cfg_name}", f"OUTPUT_PARENT=datasets/{args.name}",
                       f"LABEL_SUFFIX={label}", f"SEED_OFFSET={i * SEEDS_PER_JOB}"],
                      f"{args.name}_{label}", resources)
        if job_id:
            jobs.append({"label": label, "job_id": job_id, "seed_offset": i * SEEDS_PER_JOB,
                         "paths": args.paths, "resources": resources, "config_file": cfg_name})
    launch_file.write_text(json.dumps({"name": args.name, "remote": f"{REMOTE_DIR}/datasets/{args.name}",
                                       "config": cfg, "jobs": jobs}, indent=2))
    print(f"{len(jobs) - start} jobs submitted ({args.paths} paths each, {resources}); "
          f"collect with: uv run scripts/hoffman2.py dataset collect {args.name} --watch")
    return 0


def cmd_dataset_collect(user, args):
    """Fetch new *_path.json files and convert them to data/<name>/samples/*.npz."""
    from concurrent.futures import ThreadPoolExecutor

    import ribbon_data

    out = ROOT / "data" / args.name
    samples, incoming = out / "samples", out / "_incoming"
    samples.mkdir(parents=True, exist_ok=True)
    incoming.mkdir(exist_ok=True)
    launch = json.loads((out / "launch.json").read_text())
    job_ids = {j["job_id"] for j in launch["jobs"]}

    def npz_name(rel: str) -> str:  # datasets/<name>/j000/job-ribbon-..._path.json
        p = Path(rel)
        return f"{p.parent.name}__{p.name.removesuffix('_path.json')}.npz"

    def fetch_convert(rel: str) -> str:
        local = incoming / Path(rel).name
        scp(remote(user, rel), str(local))
        try:
            ribbon_data.json_to_npz(local, samples / npz_name(rel))
        finally:
            local.unlink(missing_ok=True)
        return rel

    while True:
        r = ssh(user, f"find datasets/{shlex.quote(args.name)} -name '*_path.json'", capture=True)
        todo = [p for p in sorted(r.stdout.split()) if not (samples / npz_name(p)).exists()]
        with ThreadPoolExecutor(args.workers) as ex:
            for rel in ex.map(fetch_convert, todo):
                print(f"  + {npz_name(rel)}")
        rows = [ribbon_data.summary(ribbon_data.load(p), p.name) for p in sorted(samples.glob("*.npz"))]
        if rows:
            with open(out / "index.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        running = job_ids & active_jobs(user)
        settled = sum(r["completed"] and r["max_ke_over_se"] < args.rest_tol for r in rows)
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} samples ({settled} complete and at rest, "
              f"KE/SE < {args.rest_tol:g}); {len(running)}/{len(job_ids)} jobs queued or running")
        if not args.watch or (not running and not todo):
            return 0
        time.sleep(args.interval)


def cmd_dataset(user, args):
    return {"launch": cmd_dataset_launch, "collect": cmd_dataset_collect}[args.action](user, args)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify SSH key login and scheduler access")
    sub.add_parser("shell", help="interactive login shell in the remote dir")
    sub.add_parser("status", help="qstat for your jobs")

    s = sub.add_parser("run", help="run a command in the remote dir (login node; keep it light)")
    s.add_argument("command", nargs=argparse.REMAINDER)

    s = sub.add_parser("log", help="print joblog.<id>")
    s.add_argument("job_id")

    s = sub.add_parser("fetch", help="copy a remote path back with scp")
    s.add_argument("path", nargs="?", default="datafiles")
    s.add_argument("--dest", default=str(ROOT / "hoffman2_data"))

    s = sub.add_parser("submit", help="qsub a MATLAB+Abaqus job on a compute node")
    s.add_argument("--script", default="submit_random.sh")
    s.add_argument("--config", default="simulation_parameters.yaml", help="remote YAML config")
    s.add_argument("--params", help="local YAML to upload and use as the config")
    s.add_argument("--output", help="remote output dir (default: datafiles_<timestamp>_cli)")
    s.add_argument("--name", help="job name (qsub -N)")
    s.add_argument("--resources", metavar="LIST",
                   help="override the script's qsub -l, e.g. h_rt=1:00:00,h_data=10G,h_vmem=20G")
    s.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                   help="extra qsub -v variable, e.g. SEED_OFFSET=10000 (repeatable)")
    s.add_argument("--wait", action="store_true", help="block until the job finishes")
    s.add_argument("--poll", type=int, default=30)
    s.add_argument("--fetch", action="store_true", help="with --wait, scp the output dir back")
    s.add_argument("--dest", default=str(ROOT / "hoffman2_data"))

    s = sub.add_parser("dataset", help="random homotopy-path dataset in data/<name>")
    ds = s.add_subparsers(dest="action", required=True)
    s = ds.add_parser("launch", help="submit jobs of random paths (append to grow a dataset)")
    s.add_argument("name")
    s.add_argument("--jobs", type=int, default=8, help="parallel jobs (~10 Abaqus licenses total)")
    s.add_argument("--paths", type=int, default=20, help="random paths per job")
    s.add_argument("--frames", type=int, default=10, help="frames per path, f = 1/N .. 1")
    s.add_argument("--hold", type=float, default=1.0, help="settling hold per frame [s]")
    s.add_argument("--append", action="store_true", help="add jobs to an existing dataset")
    s = ds.add_parser("collect", help="fetch finished paths and convert to npz")
    s.add_argument("name")
    s.add_argument("--watch", action="store_true", help="keep collecting until all jobs finish")
    s.add_argument("--interval", type=int, default=120, help="seconds between polls with --watch")
    s.add_argument("--workers", type=int, default=6, help="parallel fetch+convert threads")
    s.add_argument("--rest-tol", type=float, default=1e-10, help="KE/SE threshold reported as at rest")

    args = p.parse_args()
    return globals()[f"cmd_{args.cmd}"](load_user(), args)


if __name__ == "__main__":
    sys.exit(main())
