#!/usr/bin/env python3
"""
run_generate.py

Copies a dataset directory tree to a new location.
All files are copied as-is, EXCEPT:
- every *.stl file is replaced with a defective version produced by DefectPipeline.

Pipeline parameters are read from a JSON config (same structure as PipelineConfig.dump_json()).

Example:
  python scripts/run_generate.py \
    --src_dataset /path/to/dataset \
    --dst_dataset /path/to/dataset_defected \
    --config /path/to/config.json \
    --num_workers 1 \
    --save_meta

Notes:
- Preserves folder structure and file names.
- Uses shutil.copy2 for non-STL files (keeps timestamps/metadata).
- For STL files: writes defective STL to the same relative path in dst.
- If --save_meta: also saves per-STL JSON next to the STL:
    <name>.defect_meta.json with timings/stats + the effective config.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from dataclasses import is_dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, get_type_hints

import sys
sys.path.append(str(Path(__file__).parent.parent))  # to import defect_pipeline

from defect_pipeline import PipelineConfig, DefectPipeline 
from defect_pipeline.io import save_json 


# ----------------------------
# Config loading (JSON -> dataclasses)
# ----------------------------

def _construct_dataclass(dc_type, data: Dict[str, Any]):
    """
    Construct a dataclass instance of type dc_type from dict `data`.
    Works for nested dataclasses used in PipelineConfig.
    """
    field_types = get_type_hints(dc_type)
    kwargs = {}
    for field in dc_type.__dataclass_fields__.values():
        name = field.name
        ftype = field_types.get(name, field.type)
        if name not in data:
            continue
        val = data[name]
        # If the field type is itself a dataclass, recurse
        if isinstance(val, dict) and isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _construct_dataclass(ftype, val)
        else:
            kwargs[name] = val
    return dc_type(**kwargs)


def load_pipeline_config_json(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _construct_dataclass(PipelineConfig, data)


# ----------------------------
# File collection
# ----------------------------

def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        for fn in filenames:
            yield d / fn


def is_stl(p: Path) -> bool:
    return p.suffix.lower() == ".stl"


def rel_to(p: Path, root: Path) -> Path:
    return p.relative_to(root)


# ----------------------------
# Processing logic
# ----------------------------

def process_one_stl(
    src_stl: Path,
    dst_stl: Path,
    cfg: PipelineConfig,
    save_meta: bool,
) -> None:
    """
    Run the defect pipeline on src_stl and save defective STL to dst_stl.
    """
    dst_stl.parent.mkdir(parents=True, exist_ok=True)

    pipe = DefectPipeline(cfg)
    res = pipe.run_from_path(str(src_stl))

    # Export defective STL
    res.mesh_defected.export(str(dst_stl))

    if save_meta:
        meta = {
            "src": str(src_stl),
            "dst": str(dst_stl),
            "timings": res.timings,
            "stats": res.stats,
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg),
        }
        meta_path = dst_stl.with_suffix(dst_stl.suffix + ".defect_meta.json")
        save_json(meta, meta_path)


def copy_non_stl(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_tasks(src_root: Path, dst_root: Path) -> List[Tuple[Path, Path, bool]]:
    """
    Returns list of (src_path, dst_path, is_stl).
    """
    tasks: List[Tuple[Path, Path, bool]] = []
    for src_path in iter_files(src_root):
        r = rel_to(src_path, src_root)
        dst_path = dst_root / r
        tasks.append((src_path, dst_path, is_stl(src_path)))
    return tasks


# ----------------------------
# Optional: parallel execution
# ----------------------------

def _worker_init_and_run(args_tuple):
    """
    Separate worker function for ProcessPoolExecutor.
    Recreates PipelineConfig and runs per-file.
    """
    src_stl, dst_stl, config_path, save_meta = args_tuple
    cfg = load_pipeline_config_json(config_path)
    process_one_stl(Path(src_stl), Path(dst_stl), cfg, save_meta)
    return True


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset", type=str, required=True, help="Path to source dataset root")
    ap.add_argument("--dst_dataset", type=str, required=True, help="Path to destination dataset root")
    ap.add_argument("--config", type=str, required=True, help="Path to pipeline config JSON (PipelineConfig.dump_json)")
    ap.add_argument("--num_workers", type=int, default=1, help="1 = sequential, >1 = process-based parallel")
    ap.add_argument("--save_meta", action="store_true", help="Save per-STL timings/stats/config next to outputs")
    ap.add_argument("--overwrite", action="store_true", help="Allow destination to exist (will add/overwrite files)")
    args = ap.parse_args()

    src_root = Path(args.src_dataset).resolve()
    dst_root = Path(args.dst_dataset).resolve()
    cfg = load_pipeline_config_json(args.config)

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"src_dataset does not exist or is not a directory: {src_root}")

    if dst_root.exists() and not args.overwrite:
        raise SystemExit(
            f"dst_dataset already exists: {dst_root}\n"
            f"Use --overwrite to proceed."
        )
    dst_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(src_root, dst_root)
    stl_tasks = [(s, d) for (s, d, isstl) in tasks if isstl]
    other_tasks = [(s, d) for (s, d, isstl) in tasks if not isstl]

    # 1) Copy all non-STL files
    for src_path, dst_path in other_tasks:
        copy_non_stl(src_path, dst_path)

    # 2) Process STL files
    total_stl = len(stl_tasks)
    print(f"Starting STL processing: {total_stl} file(s), workers={args.num_workers}", flush=True)
    if args.num_workers <= 1:
        for i, (src_stl, dst_stl) in enumerate(stl_tasks, start=1):
            print(f"[{i}/{total_stl}] {src_stl}", flush=True)
            process_one_stl(src_stl, dst_stl, cfg, args.save_meta)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        payloads = [
            (str(src_stl), str(dst_stl), str(args.config), bool(args.save_meta))
            for src_stl, dst_stl in stl_tasks
        ]

        mp_ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as ex:
                futures = [ex.submit(_worker_init_and_run, p) for p in payloads]
                for i, fut in enumerate(as_completed(futures), start=1):
                    _ = fut.result()  # to catch exceptions from workers
                    print(f"[{i}/{total_stl}] done", flush=True)
        except (BrokenProcessPool, PermissionError, OSError) as e:
            print(
                f"Process pool unavailable ({e.__class__.__name__}: {e}). "
                "Falling back to sequential processing.",
                flush=True,
            )
            for i, (src_stl, dst_stl) in enumerate(stl_tasks, start=1):
                print(f"[fallback {i}/{total_stl}] {src_stl}", flush=True)
                process_one_stl(src_stl, dst_stl, cfg, args.save_meta)

    # Save dataset-level config for convenience
    (dst_root / "_defect_pipeline").mkdir(parents=True, exist_ok=True)
    out_cfg_path = dst_root / "_defect_pipeline" / "config.json"
    cfg.dump_json(out_cfg_path)
    

    print(f"Done.\n  src: {src_root}\n  dst: {dst_root}")
    print(f"  copied non-STL: {len(other_tasks)}")
    print(f"  processed STL:  {len(stl_tasks)}")
    print(f"  config saved:   {out_cfg_path}")


if __name__ == "__main__":
    main()
