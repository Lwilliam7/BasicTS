"""Run the four server follow-ups. See PROTOCOL.md. No SSH, installs or deletions."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
DATASETS = ("ETTh1", "Electricity", "Traffic", "AppliancesEnergy", "OikolabWeather", "SolarAL", "TetouanPower")
SEEDS = (7, 11, 13, 17, 19)
METHODS = ("fame", "timerouter", "superlinear")
GRAN = Path("experiments/gate_granularity_exploratory_20261020")
MUTEX = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                f.write(b"0"); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def gpu_rows():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [tuple(int(x.strip()) for x in row.split(",")) for row in result.stdout.splitlines() if row.strip()]


def protocol_processes():
    """Inspect process command lines; never send a signal to another workload."""
    if os.name != "posix":
        return []
    found = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(Path(a).name in ("run_long_horizon.py", "run_granularity.py", "run_gate_ablation.py")
                   or a.replace("\\", "/").endswith("experiments/long_horizon_protocol/run_queue.py") for a in argv):
                found.append(int(entry.name))
        except OSError:
            pass
    return found


def identity(root):
    files = list(HERE.glob("*.py")) + [HERE / "PROTOCOL.md", HERE / "granularity_source_record.json", HERE / "launch.sh"]
    files += list((root / GRAN).glob("*.py")) + list((root / GRAN).glob("*.md"))
    for rel in ("fame_official_replication/run_fame_official_replication.py",
                "timerouter_official_replication/run_timerouter_official_replication.py",
                "super_linear_replication/run_super_linear_replication.py", "official_router_ports/common.py",
                "long_horizon_protocol/run_queue.py", "long_horizon_protocol/run_long_horizon.py",
                "wide_top3_method_freeze/freeze_manifest.json"):
        files.append(root / "experiments" / rel)
    for folder in (root / "third_party/FAME/fame", root / "third_party/TimeRouter/gift_eval",
                   root / "third_party/SuperLinear/model_files"):
        if folder.exists():
            files += list(folder.glob("*.py"))
    return {str(p.resolve()): sha(p) for p in sorted(files)}


def versions():
    out = {}
    for name in ("numpy", "torch", "xgboost", "tsfeatures", "threadpoolctl", "pandas", "scikit-learn"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def plan(root, output):
    required = ("third_party/FAME/fame/router.py", "third_party/TimeRouter/gift_eval/_router.py",
                "third_party/SuperLinear/model_files/modeling_super_linear.py",
                "experiments/long_horizon_protocol/pool_plan.json")
    try:
        gpus = gpu_rows()
    except (OSError, ValueError, subprocess.SubprocessError):
        gpus = None
    return dict(repo_root=str(root), output=str(output), versions=versions(), gpu_rows=gpus,
                free_disk_gib=shutil.disk_usage(root).free / 2**30,
                missing_files=[p for p in required if not (root / p).is_file()],
                datasets=DATASETS, conditional_horizons=[96, 192], granularity_horizons=[12, 192],
                granularity_seeds=SEEDS, gpu_jobs_at_once=1, cpu_threads=8,
                unsupported=["TimeRouter H=192: CV horizon exceeds the 96-point history"],
                evidence="Development; granularity explicitly exploratory post-result", test_set_accessed=False)


class Suite:
    def __init__(self, root, output, poll=30):
        self.root, self.output, self.poll = root, output, poll
        self.manifest = dict(source_hashes=identity(root), packages=versions(), python=sys.version,
                             datasets=DATASETS, seeds=SEEDS, test_set_accessed=False)
        # JSON roundtrip gives tuples the same representation as a loaded manifest.
        self.manifest = json.loads(json.dumps(self.manifest))
        path = output / "manifest.json"
        if path.exists() and read(path) != self.manifest:
            raise RuntimeError("Source/packages/settings changed: use a new suite output directory")
        write(path, self.manifest)
        self.state = read(output / "status.json") if (output / "status.json").exists() else {"stages": {}}
        self.state.update(status="RUNNING", pid=os.getpid(), updated_utc=now(), test_set_accessed=False)
        self.logs = output / "logs"
        self.logs.mkdir(exist_ok=True)
        self.persist()

    def persist(self):
        with MUTEX:
            self.state["updated_utc"] = now()
            write(self.output / "status.json", self.state)

    def mark(self, key, status, **extra):
        with MUTEX:
            old = self.state["stages"].get(key, {})
            self.state["stages"][key] = old | dict(status=status, updated_utc=now(), **extra)
            self.persist()
            if old.get("status") != status:
                print(f"[{now()}] {key}: {status}", flush=True)

    def verify_sources(self):
        if identity(self.root) != self.manifest["source_hashes"]:
            raise RuntimeError("Source changed during suite run")

    def run(self, key, args, gpu=None, validate=None):
        self.verify_sources()
        if shutil.disk_usage(self.root).free < 2 * 2**30:
            self.mark(key, "BLOCKED", reason="Less than 2 GiB free; no files deleted")
            return False
        env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), PYTHONUNBUFFERED="1")
        self.mark(key, "RUNNING", gpu=gpu, started_utc=now())
        log = self.logs / (key.replace("/", "_") + ".log")
        try:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"\nATTEMPT {now()}\n"); stream.flush()
                rc = subprocess.run([sys.executable, *map(str, args)], cwd=self.root, env=env,
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
            if rc:
                raise RuntimeError(f"Exit {rc}; see {log}")
            self.verify_sources()
            if validate:
                validate()
            self.mark(key, "COMPLETE", finished_utc=now(), log=str(log))
            return True
        except Exception as exc:
            self.mark(key, "FAILED", reason=str(exc), log=str(log))
            return False

    @contextlib.contextmanager
    def idle_gpu(self, key):
        last_reason = None
        while True:
            reason = "Existing protocol job is running" if protocol_processes() else "Waiting for an idle GPU"
            if not protocol_processes():
                for gpu, used, util in gpu_rows():
                    if used < 1024 and util < 5:
                        try:
                            with lock(self.root / "logs" / f"research_suite_gpu{gpu}.lock"):
                                # Recheck after acquiring our cooperative lock.
                                if not any(g == gpu and m < 1024 and u < 5 for g, m, u in gpu_rows()):
                                    continue
                                yield gpu
                                return
                        except (BlockingIOError, PermissionError):
                            continue
            if last_reason != reason:
                self.mark(key, "WAITING", reason=reason)
                last_reason = reason
            time.sleep(self.poll)

    def cpu_lane(self):
        checker = HERE / "run_router_checks.py"
        health_ok = {}
        for method in METHODS:
            checks = []
            for case in ("predictable", "shuffled"):
                dest = self.output / "health" / method / case
                checks.append(self.run(f"health/{method}/{case}", [checker, "health", "--method", method,
                    "--case", case, "--repo-root", self.root, "--output", dest],
                    validate=lambda d=dest: require(read(d / "health.json")["passed"], "Health check did not pass")))
            health_ok[method] = all(checks)
        conditional = HERE / "conditional.py"
        dest = self.output / "conditional"
        selftest = self.run("conditional/selftest", [conditional, "--self-test", "--threads", "2"])
        if selftest:
            self.run("conditional/run", [conditional, "--server", "run", "--repo-root", self.root,
                    "--output", dest, "--threads", "8", "--workers", "4", "--jobs", "2"],
                    validate=lambda: require(read(dest / "summary.json")["status"] == "COMPLETE", "Conditional queue incomplete"))
        else:
            self.mark("conditional/run", "BLOCKED", reason="Conditional self-test failed")
        for method in METHODS:
            for horizon in (96, 192):
                for ds in DATASETS:
                    key = f"ports/{method}/{ds}_h{horizon}"
                    if not health_ok[method]:
                        self.mark(key, "BLOCKED", reason="Port did not pass both synthetic controls")
                        continue
                    if method == "timerouter" and horizon > 96:
                        self.mark(key, "UNSUPPORTED", reason="Existing CV requires H <= history length 96")
                        continue
                    dest = self.output / "ports" / method / f"{ds}_h{horizon}"
                    self.run(key, [checker, "unit", "--method", method, "--dataset", ds, "--horizon", str(horizon),
                                   "--repo-root", self.root, "--output", dest],
                             validate=lambda d=dest: require(read(d / "result.json")["status"] == "COMPLETE", "Router unit incomplete"))

    def recovery_done(self):
        folder = self.root / "experiments/long_horizon_protocol"
        state = read(folder / "queue_state.json")
        require(state.get("finished") and state.get("analysis_rc") == 0, "Long-horizon analysis not complete")
        require(len(state["jobs"]) == 148 and all(j["status"] == "done" for j in state["jobs"].values()), "Long-horizon jobs incomplete")
        checks = read(folder / "integrity_checks.json")
        # The native file groups unit flags; inspect each unit record rather than accepting exit 0 alone.
        units = checks.get("units", checks)
        unit_flags = [v for v in units.values() if isinstance(v, dict) and "ok" in v]
        require(len(unit_flags) == 21 and all(v["ok"] for v in unit_flags), "Long-horizon integrity checks did not pass")

    def gpu_lane(self):
        # Save originals once per suite, including any already-returned incomplete analysis.
        backups = self.output / "original_long_horizon"
        backups.mkdir(exist_ok=True)
        for rel in ("results_bundle_long_horizon.tar", "experiments/long_horizon_protocol/queue_state.json",
                    "logs/lh_seed17_Traffic_h336.log"):
            src, dest = self.root / rel, backups / Path(rel).name
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)
        try:
            self.recovery_done()
            self.mark("traffic/recovery", "COMPLETE", reason="Existing complete recovery verified")
        except (RuntimeError, OSError, KeyError):
            with self.idle_gpu("traffic/recovery") as gpu:
                self.run("traffic/recovery", [self.root / "experiments/long_horizon_protocol/run_queue.py",
                         "--run", "--gpus", str(gpu)], gpu=gpu, validate=self.recovery_done)
        script = self.root / GRAN / "run_granularity.py"
        if not self.run("granularity/selftest", [script, "--selftest"]):
            self.mark("granularity/jobs", "BLOCKED", reason="Exploratory runner self-test failed")
            return
        with self.idle_gpu("granularity/anchor") as gpu:
            anchor = self.run("granularity/anchor", [script, "--anchor"], gpu=gpu,
                              validate=lambda: require(read(script.parent / "anchor_check.json")["ok"], "Anchor did not pass"))
        if not anchor:
            self.mark("granularity/jobs", "BLOCKED", reason="New source manifest lacks a passing EXACT anchor")
            return
        # Canary first; native per-job resume validates sources, inputs, references and tensor hashes.
        order = [("AppliancesEnergy", 12, 7)] + [(d, h, s) for h in (12, 192) for d in DATASETS for s in SEEDS
                                                if (d, h, s) != ("AppliancesEnergy", 12, 7)]
        native_state_path = script.parent / "queue_state.json"
        native_state = read(native_state_path) if native_state_path.exists() else {"jobs": {}}
        for i, (ds, horizon, seed) in enumerate(order):
            key = f"granularity/{ds}_h{horizon}_seed{seed}"
            with self.idle_gpu(key) as gpu:
                ok = self.run(key, [script, "--unit", ds, "--horizon", str(horizon), "--seeds", str(seed)], gpu=gpu,
                    validate=lambda d=ds, h=horizon, s=seed: require(read(script.parent / "_work" / f"{d}_h{h}" / f"seed{s}.json")["integrity"]["ok"], "Job integrity failed"))
            native_state["jobs"][f"{ds}_h{horizon}_seed{seed}"] = {"status": "done" if ok else "failed"}
            write(native_state_path, native_state)
            if i == 0 and not ok:
                self.mark("granularity/jobs", "BLOCKED", reason="Canary failed; remaining granularity jobs not launched")
                return
        def validate_analysis():
            checks = read(script.parent / "integrity_checks.json")
            units = [v for v in checks.values() if isinstance(v, dict) and "ok" in v]
            require(len(units) == 14 and all(v["ok"] for v in units), "Exploratory granularity incomplete/invalid")
        self.run("granularity/analyze", [script, "--analyze"], validate=validate_analysis)

    def execute(self):
        failures = []
        def lane(name, fn):
            try:
                fn()
            except Exception as exc:
                failures.append(name)
                self.mark(f"lane/{name}", "FAILED", reason=str(exc))
        cpu = threading.Thread(target=lane, args=("cpu", self.cpu_lane), name="cpu-lane")
        gpu = threading.Thread(target=lane, args=("gpu", self.gpu_lane), name="gpu-lane")
        cpu.start(); gpu.start()
        cpu.join(); gpu.join()
        statuses = [r["status"] for r in self.state["stages"].values()]
        self.state["status"] = "COMPLETE" if statuses and all(s == "COMPLETE" for s in statuses) else "PARTIAL"
        self.state["finished_utc"] = now()
        self.persist()
        lines = ["# Server follow-up suite", "", f"Status: {self.state['status']}", "",
                 "New granularity evidence is exploratory and post-result. No test split was accessed.", "",
                 "| Stage | Status | Reason |", "| --- | --- | --- |"]
        lines += [f"| {k} | {v['status']} | {v.get('reason', '')} |" for k, v in self.state["stages"].items()]
        (self.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        pack(self.root, self.output)
        return 0 if self.state["status"] == "COMPLETE" else 2


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def pack(root, output):
    path = output.with_suffix(".tar.gz")
    temporary = path.with_suffix(".tar.gz.tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        for folder in (output, root / GRAN):
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*")):
                if (p.is_file() and p.suffix in (".json", ".md", ".csv", ".log", ".npz", ".pt", ".py")
                        and not (p.name.startswith("fingerprints_") and p.suffix == ".pt")
                        and not p.name.endswith("_features.npz")):
                    tar.add(p, arcname=str(p.relative_to(root)))
    temporary.replace(path)
    print(f"Results archive: {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run", "status", "pack"))
    p.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output", type=Path)
    p.add_argument("--poll-seconds", type=int, default=30)
    a = p.parse_args()
    root = a.repo_root.resolve()
    output = (a.output or root / "experiments/research_suite_20261020").resolve()
    if not output.is_relative_to(root):
        p.error("Output must remain inside the selected repository")
    if a.poll_seconds < 5:
        p.error("Polling interval must be at least five seconds")
    if a.mode == "plan":
        print(json.dumps(plan(root, output), indent=2))
    elif a.mode == "status":
        if not (output / "status.json").exists():
            print("No suite has started")
            return
        state = read(output / "status.json")
        print(f"Suite: {state['status']}; last update {state['updated_utc']}; PID {state['pid']}")
        for prefix in ("traffic", "health", "conditional", "ports", "granularity", "lane"):
            rows = {k: v for k, v in state["stages"].items() if k.startswith(prefix + "/")}
            if rows:
                counts = {s: sum(v["status"] == s for v in rows.values()) for s in sorted({v['status'] for v in rows.values()})}
                print(f"{prefix}: {counts}")
        for key, value in state["stages"].items():
            if value["status"] in ("RUNNING", "WAITING", "FAILED", "BLOCKED"):
                print(f"  {key}: {value['status']} {value.get('reason', '')}")
    else:
        with lock(output / ".suite.lock"):
            if a.mode == "pack":
                pack(root, output)
            else:
                sys.exit(Suite(root, output, a.poll_seconds).execute())


if __name__ == "__main__":
    main()
