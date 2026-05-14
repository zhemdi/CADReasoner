#!/usr/bin/env python3
"""
run_generate.py

Replaces every *.stl in src_dataset with a defective version in dst_dataset.
Non-STL files are copied as-is.

Example:
  python scripts/run_generate.py \
    --src_dataset /path/to/dataset \
    --dst_dataset /path/to/dataset_defected \
    --config /path/to/config.json \
    --num_workers 16
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import shutil
import time
from collections import deque
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, get_type_hints

import sys
sys.path.append(str(Path(__file__).parent.parent))

from defect_pipeline.config import PipelineConfig


MISSING_SURFACES = ["-z", "-x", "-y", "+z", "+x", "+y"]


def _construct_dataclass(dc_type, data: Dict[str, Any]):
    field_types = get_type_hints(dc_type)
    kwargs = {}
    for field in dc_type.__dataclass_fields__.values():
        name = field.name
        ftype = field_types.get(name, field.type)
        if name not in data:
            continue
        val = data[name]
        if isinstance(val, dict) and isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _construct_dataclass(ftype, val)
        else:
            kwargs[name] = val
    return dc_type(**kwargs)


def load_pipeline_config_json(path: str | Path) -> PipelineConfig:
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    return _construct_dataclass(PipelineConfig, data)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def build_tasks(src_root: Path, dst_root: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    stl_tasks: List[Tuple[Path, Path]] = []
    other_tasks: List[Tuple[Path, Path]] = []
    for dirpath, _, filenames in os.walk(src_root):
        d = Path(dirpath)
        for fn in filenames:
            src = d / fn
            dst = dst_root / src.relative_to(src_root)
            if src.suffix.lower() == ".stl":
                stl_tasks.append((src, dst))
            else:
                other_tasks.append((src, dst))
    return stl_tasks, other_tasks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_CFG_PATH: str = ""


def _init_worker(config_path: str) -> None:
    global _CFG_PATH
    _CFG_PATH = config_path
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _process_one(args: Tuple[str, str]) -> Tuple[str, str]:
    src_str, dst_str = args
    src_stl, dst_stl = Path(src_str), Path(dst_str)

    try:
        from defect_pipeline import DefectPipeline

        cfg = load_pipeline_config_json(_CFG_PATH)
        cfg = replace(cfg,
            scan=replace(cfg.scan, missing_surface=random.choice(MISSING_SURFACES)),
            defect=replace(cfg.defect, n_holes=random.randint(0, 8)),
            inp=replace(cfg.inp, seed=None),
        )

        pipe = DefectPipeline(cfg)
        mesh_def, _, _ = pipe.run_for_export(str(src_stl))

        dst_stl.parent.mkdir(parents=True, exist_ok=True)
        mesh_def.export(str(dst_stl))

        del mesh_def, pipe, cfg
        gc.collect()
        return src_stl.name, "OK"

    except Exception as e:
        return src_stl.name, f"FAIL: {e}"


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Compact rolling-window progress with ETA."""

    def __init__(self, total: int, log_interval: float = 10.0):
        self.total = total
        self.log_interval = log_interval
        self.done = 0
        self.fails = 0
        self.t_start = time.monotonic()
        self.t_last_log = self.t_start
        self._recent: deque[float] = deque()

    def tick(self, ok: bool) -> None:
        self.done += 1
        if not ok:
            self.fails += 1
        now = time.monotonic()
        self._recent.append(now)
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()

        if now - self.t_last_log >= self.log_interval:
            self._print(now)
            self.t_last_log = now

    def _print(self, now: float) -> None:
        elapsed = now - self.t_start
        left = self.total - self.done
        pct = 100.0 * self.done / self.total

        recent_rate = len(self._recent) / 60.0 if self._recent else 0.0
        avg_rate = self.done / elapsed if elapsed > 0 else 0.0

        if recent_rate > 0:
            eta_sec = left / recent_rate
        elif avg_rate > 0:
            eta_sec = left / avg_rate
        else:
            eta_sec = 0.0

        eta = _fmt_time(eta_sec)
        el = _fmt_time(elapsed)

        fail_s = f"  fails={self.fails}" if self.fails else ""
        print(
            f"[{self.done}/{self.total}] {pct:.1f}%  "
            f"last_min={len(self._recent):.0f} stl  "
            f"avg={avg_rate:.1f} stl/s  "
            f"elapsed={el}  ETA={eta}{fail_s}",
            flush=True,
        )

    def summary(self) -> None:
        elapsed = time.monotonic() - self.t_start
        avg = self.done / elapsed if elapsed > 0 else 0.0
        print(
            f"\nDone: {self.done}/{self.total} STL in {_fmt_time(elapsed)}  "
            f"avg={avg:.1f} stl/s  fails={self.fails}",
            flush=True,
        )


def _fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset", type=str, required=True)
    ap.add_argument("--dst_dataset", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_dataset).resolve()
    dst_root = Path(args.dst_dataset).resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"src_dataset not found: {src_root}")
    if dst_root.exists() and not args.overwrite:
        raise SystemExit(f"dst_dataset exists: {dst_root}\nUse --overwrite to proceed.")
    dst_root.mkdir(parents=True, exist_ok=True)

    stl_tasks, other_tasks = build_tasks(src_root, dst_root)

    for src_path, dst_path in other_tasks:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
    if other_tasks:
        print(f"Copied {len(other_tasks)} non-STL files", flush=True)

    total = len(stl_tasks)
    print(f"STL to process: {total}  workers: {args.num_workers}", flush=True)

    payloads = [(str(s), str(d)) for s, d in stl_tasks]
    tracker = ProgressTracker(total, log_interval=10.0)

    mp.set_start_method("spawn", force=True)

    if args.num_workers <= 1:
        _init_worker(args.config)
        for payload in payloads:
            name, status = _process_one(payload)
            tracker.tick(ok=status == "OK")
    else:
        cs = max(1, total // (args.num_workers * 8))
        with mp.Pool(args.num_workers, _init_worker, (args.config,)) as pool:
            for name, status in pool.imap_unordered(_process_one, payloads, chunksize=cs):
                tracker.tick(ok=status == "OK")
                if status != "OK":
                    print(f"  FAIL: {name}: {status}", flush=True)

    tracker.summary()
    print(f"  src: {src_root}\n  dst: {dst_root}")


if __name__ == "__main__":
    main()
