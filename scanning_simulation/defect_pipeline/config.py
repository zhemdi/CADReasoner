from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any

MissingSurface = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


@dataclass(frozen=True)
class ScanConfig:
    missing_surface: MissingSurface = "-z"
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    hpr_factor: float = 100.0

    n_az: int = 2
    n_el: int = 3


@dataclass(frozen=True)
class PoissonConfig:
    depth: int = 7
    width: int = 0
    scale: float = 1.1
    linear_fit: bool = False
    n_threads: int = 1

    density_quantile: Optional[float] = None


@dataclass(frozen=True)
class CleanupConfig:
    tau_nn_mult: float = 2.5


@dataclass(frozen=True)
class DefectConfig:
    n_holes: int = 4
    n_gaussian_bumps: int = 0

    bump_radius_ratio: float = 0.008
    hole_radius_range: Tuple[float, float] = (0.5, 1.0)

    sigma: float = 0.8
    amplitude: float = 0.023

    noise_level: float = 1e-4

    min_center_dist_factor: float = 2.0

    candidates_mult: int = 50


@dataclass(frozen=True)
class SimplifyConfig:
    coef_triangle_size: Optional[float] = 5


@dataclass(frozen=True)
class InputConfig:
    pc_size: int = 100_000
    seed: Optional[int] = 42


@dataclass(frozen=True)
class PipelineConfig:
    inp: InputConfig = InputConfig()
    scan: ScanConfig = ScanConfig()
    poisson: PoissonConfig = PoissonConfig()
    cleanup: CleanupConfig = CleanupConfig()
    defect: DefectConfig = DefectConfig()
    simplify: SimplifyConfig = SimplifyConfig()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
