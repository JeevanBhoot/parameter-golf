#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


EXACT_RE = re.compile(r"final_int8_zlib_roundtrip_exact .* val_bpb:([0-9.]+)")
ARTIFACT_RE = re.compile(r"artifact_bytes_int8_zlib:([0-9]+)")
RSS_RE = re.compile(r"train_peak_rss_bytes:([0-9]+)")
REAL_RE = re.compile(r"^real\s+([0-9.]+)\s*$", re.MULTILINE)
DEFAULT_QUICK_HEADER = (
    "run_tag\tval_bpb\tartifact_bytes\ttime\ttrain_peak_rss_bytes\tstatus\tdescription\n"
)


@dataclass
class Metrics:
    run_tag: str
    val_bpb: str
    artifact_bytes: str
    wallclock: str
    train_peak_rss_bytes: str
    crashed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential queue of MLX experiment scripts.")
    parser.add_argument("spec", type=Path, help="Path to a JSON queue spec.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run entries even if their exact metric already exists in the run log.",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=None,
        help="Override the default sleep between runs.",
    )
    return parser.parse_args()


def repo_root_from_spec(spec_path: Path) -> Path:
    del spec_path
    return Path(__file__).resolve().parents[1]


def load_spec(spec_path: Path) -> dict:
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict) or not isinstance(spec.get("experiments"), list):
        raise ValueError("Queue spec must be a JSON object with an 'experiments' list.")
    return spec


def maybe_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def parse_wallclock(time_log_path: Path) -> str:
    text = maybe_text(time_log_path)
    match = REAL_RE.search(text)
    if not match:
        return "0:00.00"
    total_seconds = float(match.group(1))
    minutes = int(total_seconds // 60.0)
    seconds = total_seconds - 60.0 * minutes
    return f"{minutes}:{seconds:05.2f}"


def parse_metrics(run_dir: Path, run_tag: str) -> Metrics:
    log_path = run_dir / f"{run_tag}.txt"
    time_log_path = run_dir / "time.log"
    text = maybe_text(log_path)
    exact_match = EXACT_RE.search(text)
    artifact_match = ARTIFACT_RE.search(text)
    rss_match = RSS_RE.search(text)
    return Metrics(
        run_tag=run_tag,
        val_bpb=exact_match.group(1) if exact_match else "0.00000000",
        artifact_bytes=artifact_match.group(1) if artifact_match else "0",
        wallclock=parse_wallclock(time_log_path),
        train_peak_rss_bytes=rss_match.group(1) if rss_match else "0",
        crashed=exact_match is None,
    )


def ensure_results_file(results_path: Path) -> None:
    if results_path.is_file():
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(DEFAULT_QUICK_HEADER, encoding="utf-8")


def upsert_results_row(
    results_path: Path,
    metrics: Metrics,
    status: str,
    description: str,
) -> None:
    ensure_results_file(results_path)
    lines = results_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        lines = [DEFAULT_QUICK_HEADER.rstrip("\n")]
    header = lines[0]
    rows = [line.split("\t", 6) for line in lines[1:] if line.strip()]
    new_row = [
        metrics.run_tag,
        metrics.val_bpb,
        metrics.artifact_bytes,
        metrics.wallclock,
        metrics.train_peak_rss_bytes,
        "crash" if metrics.crashed else status,
        description,
    ]
    replaced = False
    for idx, row in enumerate(rows):
        if row and row[0] == metrics.run_tag:
            rows[idx] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    body = "\n".join("\t".join(row) for row in rows)
    results_path.write_text(f"{header}\n{body}\n", encoding="utf-8")


def run_complete(run_dir: Path, run_tag: str) -> bool:
    log_path = run_dir / f"{run_tag}.txt"
    if not log_path.is_file():
        return False
    return EXACT_RE.search(log_path.read_text(encoding="utf-8")) is not None


def resolve_experiment_paths(repo_root: Path, experiment: dict) -> tuple[Path, Path]:
    script_path = (repo_root / experiment["script_path"]).resolve()
    run_dir = script_path.parent
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing script for {experiment['run_tag']}: {script_path}")
    return script_path, run_dir


def resolve_experiment_python(repo_root: Path, env: dict[str, str]) -> str:
    override = env.get("QUEUE_PYTHON") or os.environ.get("QUEUE_PYTHON")
    if override:
        return override
    for candidate in (repo_root / ".venv" / "bin" / "python3", repo_root / ".venv" / "bin" / "python"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def launch_experiment(repo_root: Path, script_path: Path, run_dir: Path, env: dict[str, str]) -> int:
    stdout_path = run_dir / "stdout.log"
    time_path = run_dir / "time.log"
    python_executable = resolve_experiment_python(repo_root, env)
    cmd = ["/usr/bin/time", "-p", python_executable, str(script_path)]
    print(f"[queue] launching {' '.join(cmd)}")
    with stdout_path.open("wb") as stdout_handle, time_path.open("wb") as time_handle:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=stdout_handle,
            stderr=time_handle,
            check=False,
        )
    return int(completed.returncode)


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)
    repo_root = repo_root_from_spec(args.spec)
    default_buffer_seconds = float(
        args.buffer_seconds if args.buffer_seconds is not None else spec.get("buffer_seconds", 60.0)
    )

    experiments: list[dict] = spec["experiments"]
    for index, experiment in enumerate(experiments, start=1):
        run_tag = str(experiment["run_tag"])
        print(f"[queue] {index}/{len(experiments)} {run_tag}")
        script_path, run_dir = resolve_experiment_paths(repo_root, experiment)
        results_path = (repo_root / experiment["results_file"]).resolve()
        status = str(experiment.get("status", "review"))
        description = str(experiment["description"])

        if run_complete(run_dir, run_tag) and not args.force:
            print(f"[queue] existing exact metric found for {run_tag}; skipping execution")
        else:
            env = os.environ.copy()
            env["RUN_ID"] = run_tag
            env["OUT_DIR"] = str(run_dir)
            for key, value in (experiment.get("env") or {}).items():
                env[str(key)] = str(value)
            returncode = launch_experiment(repo_root, script_path, run_dir, env)
            print(f"[queue] return code for {run_tag}: {returncode}")

        metrics = parse_metrics(run_dir, run_tag)
        upsert_results_row(results_path, metrics, status=status, description=description)
        print(
            f"[queue] logged {run_tag} val_bpb={metrics.val_bpb} "
            f"artifact_bytes={metrics.artifact_bytes} time={metrics.wallclock} "
            f"rss={metrics.train_peak_rss_bytes} status={'crash' if metrics.crashed else status}"
        )

        if index < len(experiments):
            sleep_seconds = float(experiment.get("sleep_after_seconds", default_buffer_seconds))
            if sleep_seconds > 0.0:
                print(f"[queue] sleeping {sleep_seconds:.1f}s before next run")
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
