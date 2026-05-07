from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sae_tools.workflow.runtime import (
    GPUInfo,
    build_runner_target_command,
    classify_gpus,
    parse_gpu_set as _parse_gpu_set,
    query_gpus,
    requested_stages as _requested_stages,
    target_log_name as _target_log_name,
    workflow_targets,
)


@dataclass
class ActiveRun:
    stage: str
    target: str
    gpu: GPUInfo
    command: list[str]
    log_path: Path
    process: subprocess.Popen
    log_handle: object
    started_at: float
    start_used_mib: int
    peak_used_mib: int
    max_util_pct: int


@dataclass
class RunRecord:
    run_id: str
    stage: str
    target: str
    gpu_index: int
    gpu_name: str
    total_mib: int
    start_used_mib: int
    peak_used_mib: int
    end_used_mib: int
    peak_delta_mib: int
    max_util_pct: int
    status: str
    returncode: int
    duration_sec: float
    log: str
    command: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Snakemake workflow targets across idle GPUs and record GPU memory usage."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/experiments/safety_grid.yaml"))
    parser.add_argument("--registry-dir", default=str(REPO_ROOT / "configs/registry"))
    parser.add_argument("--snakefile", default=str(REPO_ROOT / "Snakefile"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs/gpu_memory"))
    parser.add_argument("--stages", default="all", help="all or comma-separated: activations,stat,geometric")
    parser.add_argument("--max-gpu-util", type=int, default=0)
    parser.add_argument(
        "--max-used-mib",
        type=int,
        default=512,
        help="Treat a GPU as empty only when used memory is at or below this driver-overhead threshold.",
    )
    parser.add_argument("--min-free-mib", type=int, default=30000)
    parser.add_argument("--include-gpus", default=None, help="Comma-separated physical GPU indices to allow.")
    parser.add_argument("--exclude-gpus", default=None, help="Comma-separated physical GPU indices to skip.")
    parser.add_argument("--max-gpus", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_EXE", "conda"))
    parser.add_argument("--conda-env", default="sae-tl3")
    parser.add_argument("--no-conda-run", action="store_true")
    parser.add_argument("--snakemake-cmd", default="snakemake")
    parser.add_argument("--extra-snakemake-arg", action="append", default=[])
    return parser.parse_args()


def _build_command(args: argparse.Namespace, target: str) -> list[str]:
    return build_runner_target_command(
        target=target,
        config_path=args.config,
        snakefile=args.snakefile,
        repo_root=REPO_ROOT,
        snakemake_cmd=args.snakemake_cmd,
        extra_snakemake_args=args.extra_snakemake_arg,
        conda_bin=args.conda_bin,
        conda_env=args.conda_env,
        no_conda_run=args.no_conda_run,
    )


def _snapshot_by_index() -> dict[int, GPUInfo]:
    return {gpu.index: gpu for gpu in query_gpus()}


def _elapsed(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _short_target(target: str, max_len: int = 88) -> str:
    if len(target) <= max_len:
        return target
    return "..." + target[-(max_len - 3) :]


def write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def write_markdown(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            values = [str(row.get(header, "")).replace("\n", " ") for header in headers]
            handle.write("| " + " | ".join(values) + " |\n")


def _record_row(record: RunRecord) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "stage": record.stage,
        "target": record.target,
        "gpu": record.gpu_index,
        "gpu_name": record.gpu_name,
        "total_mib": record.total_mib,
        "start_used_mib": record.start_used_mib,
        "peak_used_mib": record.peak_used_mib,
        "end_used_mib": record.end_used_mib,
        "peak_delta_mib": record.peak_delta_mib,
        "max_util_pct": record.max_util_pct,
        "status": record.status,
        "returncode": record.returncode,
        "duration_sec": f"{record.duration_sec:.1f}",
        "log": record.log,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "command": record.command,
    }


def _persist_records(run_dir: Path, records: list[RunRecord]) -> None:
    headers = [
        "run_id",
        "stage",
        "target",
        "gpu",
        "gpu_name",
        "total_mib",
        "start_used_mib",
        "peak_used_mib",
        "end_used_mib",
        "peak_delta_mib",
        "max_util_pct",
        "status",
        "returncode",
        "duration_sec",
        "log",
        "started_at",
        "ended_at",
        "command",
    ]
    rows = [_record_row(record) for record in records]
    write_csv(run_dir / "memory_usage.csv", rows, headers)
    table_headers = [
        "stage",
        "target",
        "gpu",
        "start_used_mib",
        "peak_used_mib",
        "peak_delta_mib",
        "max_util_pct",
        "status",
        "duration_sec",
        "log",
    ]
    write_markdown(run_dir / "memory_usage.md", rows, table_headers)


def _start_run(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    stage: str,
    target: str,
    gpu: GPUInfo,
) -> ActiveRun:
    command = _build_command(args, target)
    log_path = run_dir / "logs" / _target_log_name(stage, target)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    env["SAE_TOOLS_RUN_ID"] = run_id
    env["SAE_TOOLS_GPU_INDEX"] = str(gpu.index)
    handle = log_path.open("w", encoding="utf-8")
    handle.write(f"run_id={run_id}\nstage={stage}\ntarget={target}\ngpu={gpu.index}\n")
    handle.write("command=" + shlex.join(command) + "\n\n")
    handle.flush()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveRun(
        stage=stage,
        target=target,
        gpu=gpu,
        command=command,
        log_path=log_path,
        process=process,
        log_handle=handle,
        started_at=time.time(),
        start_used_mib=gpu.memory_used_mib,
        peak_used_mib=gpu.memory_used_mib,
        max_util_pct=gpu.utilization_gpu_pct,
    )


def _finish_run(run_id: str, run_dir: Path, active: ActiveRun, records: list[RunRecord]) -> RunRecord:
    end_snapshot = _snapshot_by_index().get(active.gpu.index, active.gpu)
    duration = time.time() - active.started_at
    returncode = active.process.returncode
    status = "success" if returncode == 0 else f"failed:{returncode}"
    try:
        active.log_handle.flush()
        active.log_handle.close()
    except Exception:
        pass
    record = RunRecord(
        run_id=run_id,
        stage=active.stage,
        target=active.target,
        gpu_index=active.gpu.index,
        gpu_name=active.gpu.name,
        total_mib=active.gpu.memory_total_mib,
        start_used_mib=active.start_used_mib,
        peak_used_mib=active.peak_used_mib,
        end_used_mib=end_snapshot.memory_used_mib,
        peak_delta_mib=max(active.peak_used_mib - active.start_used_mib, 0),
        max_util_pct=active.max_util_pct,
        status=status,
        returncode=returncode,
        duration_sec=duration,
        log=str(active.log_path.relative_to(REPO_ROOT)),
        command=shlex.join(active.command),
        ended_at=datetime.now(timezone.utc).isoformat(),
    )
    records.append(record)
    _persist_records(run_dir, records)
    return record


def run_stage(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    stage: str,
    targets: list[str],
    gpus: list[GPUInfo],
    records: list[RunRecord],
) -> bool:
    if not targets:
        return True
    pending = list(targets)
    active: list[ActiveRun] = []
    failed = False
    while pending or active:
        active_gpu_indices = {item.gpu.index for item in active}
        free_gpus = [gpu for gpu in gpus if gpu.index not in active_gpu_indices]
        while pending and free_gpus and not (failed and not args.keep_going):
            gpu = free_gpus.pop(0)
            target = pending.pop(0)
            print(f"[{stage}] gpu={gpu.index} target={target}", flush=True)
            active.append(
                _start_run(args=args, run_id=run_id, run_dir=run_dir, stage=stage, target=target, gpu=gpu)
            )

        time.sleep(args.poll_interval)
        snapshot = _snapshot_by_index()
        still_active: list[ActiveRun] = []
        status_parts: list[str] = []
        for item in active:
            current = snapshot.get(item.gpu.index)
            if current is not None:
                item.peak_used_mib = max(item.peak_used_mib, current.memory_used_mib)
                item.max_util_pct = max(item.max_util_pct, current.utilization_gpu_pct)
                status_parts.append(
                    f"gpu={item.gpu.index} used={current.memory_used_mib}/{current.memory_total_mib}MiB "
                    f"peak={item.peak_used_mib}MiB util={current.utilization_gpu_pct}% "
                    f"elapsed={_elapsed(time.time() - item.started_at)} target={_short_target(item.target)}"
                )
            returncode = item.process.poll()
            if returncode is None:
                still_active.append(item)
                continue
            record = _finish_run(run_id, run_dir, item, records)
            print(
                f"[{stage}] done gpu={record.gpu_index} status={record.status} "
                f"peak_delta={record.peak_delta_mib} MiB target={record.target}",
                flush=True,
            )
            if record.returncode != 0:
                failed = True
        active = still_active
        if status_parts:
            print(f"[status] stage={stage} pending={len(pending)} active={len(active)} | " + " | ".join(status_parts), flush=True)

    return not failed


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = _requested_stages(args.stages)
    targets_by_stage = workflow_targets(Path(args.config), Path(args.registry_dir))
    include = _parse_gpu_set(args.include_gpus)
    exclude = _parse_gpu_set(args.exclude_gpus)
    gpus, snapshot_rows = classify_gpus(
        query_gpus(),
        max_gpu_util=args.max_gpu_util,
        max_used_mib=args.max_used_mib,
        min_free_mib=args.min_free_mib,
        include=include,
        exclude=exclude,
    )
    if args.max_gpus is not None:
        gpus = gpus[: args.max_gpus]
        selected_indices = {gpu.index for gpu in gpus}
        for row in snapshot_rows:
            if row["selected"] == "yes" and row["gpu"] not in selected_indices:
                row["selected"] = "no"
                row["reason"] = "over max-gpus limit"

    snapshot_headers = ["gpu", "name", "total_mib", "used_mib", "free_mib", "util_pct", "selected", "reason"]
    write_csv(run_dir / "gpu_snapshot_initial.csv", snapshot_rows, snapshot_headers)
    write_markdown(run_dir / "gpu_snapshot_initial.md", snapshot_rows, snapshot_headers)

    plan_rows = [
        {"stage": stage, "target": target}
        for stage in stages
        for target in targets_by_stage[stage]
    ]
    write_markdown(run_dir / "target_plan.md", plan_rows, ["stage", "target"])

    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config)),
        "registry_dir": str(Path(args.registry_dir)),
        "stages": stages,
        "selected_gpus": [gpu.index for gpu in gpus],
        "max_gpu_util": args.max_gpu_util,
        "max_used_mib": args.max_used_mib,
        "min_free_mib": args.min_free_mib,
        "dry_run": args.dry_run,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"selected_gpus={[gpu.index for gpu in gpus]}")
    for stage in stages:
        print(f"{stage}: {len(targets_by_stage[stage])} targets")

    if args.dry_run:
        return
    if not gpus:
        raise SystemExit("No GPUs matched the idle filters. Lower --min-free-mib or adjust include/exclude filters.")

    records: list[RunRecord] = []
    try:
        for stage in stages:
            ok = run_stage(
                args=args,
                run_id=run_id,
                run_dir=run_dir,
                stage=stage,
                targets=targets_by_stage[stage],
                gpus=gpus,
                records=records,
            )
            if not ok and not args.keep_going:
                raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Child Snakemake processes may need manual termination.")
    finally:
        final_rows = []
        for gpu in query_gpus():
            final_rows.append(
                {
                    "gpu": gpu.index,
                    "name": gpu.name,
                    "total_mib": gpu.memory_total_mib,
                    "used_mib": gpu.memory_used_mib,
                    "free_mib": gpu.memory_free_mib,
                    "util_pct": gpu.utilization_gpu_pct,
                    "selected": "yes" if gpu.index in {item.index for item in gpus} else "no",
                    "reason": "final snapshot",
                }
            )
        write_csv(run_dir / "gpu_snapshot_final.csv", final_rows, snapshot_headers)
        write_markdown(run_dir / "gpu_snapshot_final.md", final_rows, snapshot_headers)


if __name__ == "__main__":
    main()
