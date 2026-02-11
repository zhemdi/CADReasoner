# Defect Pipeline
Scan-based surface reconstruction with synthetic geometric defects

## Overview

This project implements a reproducible pipeline for generating defective 3D meshes
from clean CAD or mesh models.

The pipeline simulates partial scanning, reconstructs the surface using Poisson
reconstruction, removes reconstruction artifacts, and injects local geometric
defects such as holes, bumps, and noise.

The code is designed to be:
- modular and easy to extend
- fully configurable via dataclass-based configs
- deterministic and reproducible (explicit random seed)
- suitable for research papers and dataset generation

---

## Pipeline Steps

1. Mesh normalization  
   Centering and scaling using axis-aligned bounding box (AABB).

2. Surface point cloud sampling  
   Dense oriented point cloud sampled from the mesh surface with barycentric
   interpolation of vertex normals.

3. Partial scan simulation  
   Hidden Point Removal (HPR) from multiple camera viewpoints, with one missing
   direction (±x, ±y, ±z).

4. Poisson surface reconstruction  
   Surface reconstruction from the visible point cloud.

5. Artifact cleanup  
   Removal of triangles not supported by the scanned point cloud.

6. Local defect injection  
   - holes (local triangle removal)
   - Gaussian bumps along surface normals
   - small random vertex noise

7. Optional mesh simplification  
   Quadric decimation to control final triangle count.

---

## Project Structure

```
project/
  defect_pipeline/
    __init__.py
    config.py          # dataclass-based pipeline configuration
    io.py              # mesh loading, normalization, saving
    scan.py            # HPR scanning and camera generation
    recon.py           # Poisson reconstruction
    cleanup.py         # artifact removal
    defects_local.py   # holes / bumps / noise
    pipeline.py        # DefectPipeline + PipelineResult
    metrics.py         # mesh statistics
    viz.py             # optional visualization utilities
  scripts/
    run_generate.py    # CLI entrypoint
    debug_one.py       # minimal Python example
```

---

## Installation

Required Python packages:

- numpy
- scipy
- trimesh
- open3d

Install with pip:

```bash
pip install numpy scipy trimesh open3d
```

---

## Command-Line Usage

Generate a defective mesh from an input mesh:

```bash
python scripts/run_generate.py \
  --mesh path/to/input.stl \
  --out_dir path/to/output/sample_0001 \
  --seed 42 \
  --pc_size 100000 \
  --missing_surface -z \
  --n_holes 2 \
  --n_bumps 2 \
  --poisson_depth 7 \
  --n_threads 8 \
  --coef_triangle_size 5
```

### Output files

```
sample_0001/
  mesh_defected.stl
  config.json
  timings.json
  stats.json
```

- mesh_defected.stl — final defective mesh  
- config.json — full pipeline configuration  
- timings.json — runtime per pipeline stage  
- stats.json — mesh statistics  

---

## Python API Example

```python
from defect_pipeline import PipelineConfig, DefectPipeline

cfg = PipelineConfig(
    inp=PipelineConfig().inp.__class__(pc_size=100_000, seed=42),
    scan=PipelineConfig().scan.__class__(missing_surface="-z"),
    poisson=PipelineConfig().poisson.__class__(depth=7, n_threads=8),
    cleanup=PipelineConfig().cleanup,
    defect=PipelineConfig().defect.__class__(n_holes=2, n_gaussian_bumps=2),
    simplify=PipelineConfig().simplify.__class__(coef_triangle_size=5),
)

pipeline = DefectPipeline(cfg)
result = pipeline.run_from_path("input.stl")

result.mesh_defected.show()
print(result.timings)
print(result.stats)
```

---

## Configuration

All parameters are defined in `defect_pipeline/config.py` and grouped by purpose.
Each run saves `config.json` next to the outputs to ensure full reproducibility.

---

### InputConfig — input sampling and reproducibility

Controls point cloud sampling size and randomness.

- **pc_size (int)**  
  Number of points sampled from the input mesh surface to form an oriented point cloud.  
  Typical range: `5e4 – 1e6`  
  - Too small: Poisson reconstruction becomes noisy and incomplete  
  - Too large: slower sampling and reconstruction, higher memory usage  

- **seed (int | None)**  
  Random seed for all stochastic components (defect placement, noise, etc.).  
  Uses `numpy.random.default_rng`.  
  Typical range: `0 – 10_000`  
  - `None`: non-deterministic results  
  - Fixed value: fully reproducible runs

---

### ScanConfig — partial scan simulation (HPR)

Defines camera placement and missing surface direction to simulate incomplete scans.

- **missing_surface (str: "+x", "-x", "+y", "-y", "+z", "-z")**  
  Direction that is intentionally under-observed during scanning.  
  - Controls which surface regions are missing after HPR  
  - Used to simulate occlusion or limited sensor coverage  

- **lookat (tuple[float, float, float])**  
  Point cameras are oriented towards.  
  Typically `(0, 0, 0)` after mesh normalization.

- **radius (float)**  
  Radius of the camera sphere around the object.  
  Typical range: `0.8 – 3.0`  
  - Too small: unstable visibility  
  - Too large: little effect, but slower HPR

- **hpr_factor (float)**  
  Multiplier for the HPR sphere radius relative to object diameter.  
  Typical range: `50 – 200`  
  - Too small: over-aggressive point removal  
  - Too large: reduced occlusion effect

- **n_az (int)**  
  Number of azimuth angles.  
  Typical range: `2 – 12`  
  - Higher values: denser scanning, less missing surface

- **n_el (int)**  
  Number of elevation angles.  
  Typical range: `2 – 8`  
  - Higher values: more complete surface coverage

---

### PoissonConfig — surface reconstruction

Controls Poisson surface reconstruction parameters.

- **depth (int)**  
  Octree depth for Poisson reconstruction (main quality parameter).  
  Typical range: `6 – 9`  
  - Low values: smooth, coarse surfaces  
  - High values: detailed but slower and noisier  

- **width (int)**  
  Solver width (usually kept at `0`).  

- **scale (float)**  
  Scale factor for reconstruction bounding box.  
  Typical value: `1.1`  
  - Too small: clipped geometry  
  - Too large: unnecessary empty space

- **linear_fit (bool)**  
  Whether to use linear interpolation in Poisson solver.  
  Usually `False`.

- **n_threads (int)**  
  Number of CPU threads for Poisson reconstruction.  
  Typical range: `1 – number_of_cores`

- **density_quantile (float | None)**  
  Optional vertex density filtering threshold.  
  Typical range: `0.05 – 0.2`  
  - Lower: keeps more geometry  
  - Higher: aggressively removes low-confidence regions  
  - `None`: disables density filtering

---

### CleanupConfig — artifact removal

Removes reconstruction artifacts not supported by the scanned point cloud.

- **tau_nn_mult (float)**  
  Distance threshold multiplier for unsupported triangle removal.  
  Uses: `tau = tau_nn_mult * mean(nn_distance)`  
  Typical range: `2.0 – 4.0`  
  - Too small: removes valid geometry  
  - Too large: leaves reconstruction artifacts

---

### DefectConfig — local geometric defects

Controls synthetic defect generation on the reconstructed mesh.

- **n_holes (int)**  
  Number of local holes (triangle removal regions).  
  Typical range: `0 – 10`

- **n_gaussian_bumps (int)**  
  Number of Gaussian surface bumps.  
  Typical range: `0 – 10`

- **bump_radius_ratio (float)**  
  Radius of bumps relative to mesh bounding box diagonal.  
  Typical range: `0.005 – 0.02`  
  - Small: subtle defects  
  - Large: clearly visible deformations

- **hole_radius_range (tuple[float, float])**  
  Range (relative to bump radius) for hole sizes.  
  Typical: `(0.5, 1.0)`

- **sigma (float)**  
  Controls smoothness of Gaussian bumps.  
  Typical range: `0.5 – 1.5`  
  - Small: sharp peaks  
  - Large: smooth, wide bumps

- **amplitude (float)**  
  Displacement magnitude along surface normals.  
  Typical range: `0.01 – 0.05`

- **noise_level (float)**  
  Uniform random noise applied to vertices.  
  Typical range: `1e-5 – 1e-3`  
  - Used to simulate surface roughness or sensor noise

- **min_center_dist_factor (float)**  
  Minimum distance between defect centers relative to hole radius.  
  Typical range: `1.5 – 3.0`  
  - Larger values: more separated defects

- **candidates_mult (int)**  
  Multiplier for candidate surface points when sampling defect centers.  
  Typical range: `20 – 100`  
  - Larger values: higher chance to place all defects successfully

---

### SimplifyConfig — mesh complexity control

Optional mesh simplification after defect injection.

- **coef_triangle_size (float | None)**  
  Target triangle count as a multiple of the original mesh size.  
  Typical range: `1.0 – 10.0`  
  - `None`: no simplification  
  - Low values: aggressive decimation  
  - High values: mostly preserves original detail

---

## Reproducibility

- All randomness is handled via `numpy.random.default_rng(seed)`.
- The exact pipeline configuration is saved in `config.json`.
- Timings and mesh statistics are stored for each run.

---

## Practical Notes

- pc_size strongly affects Poisson reconstruction quality.
- poisson_depth controls reconstruction detail and runtime.
- tau_nn_mult balances artifact removal vs geometry preservation.
- coef_triangle_size helps normalize mesh complexity across samples.

---

## Common Issues

Only X centers found, needed K  
Increase `candidates_mult`, reduce `min_center_dist_factor`,
or reduce defect radius.

Poisson reconstruction is slow  
Reduce `depth`, reduce `pc_size`, or increase `n_threads`.
