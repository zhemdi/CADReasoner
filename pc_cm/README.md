# CADReasoner — point-cloud and cross-modality models

This directory contains the training, inference and evaluation code for the two geometry-conditioned
CADReasoner variants:

| Model | `--use_pc` | `--use_img` | Input modality |
|---|---|---|---|
| [`kulibinai/cadreasoner-pc`](https://huggingface.co/kulibinai/cadreasoner-pc) | `true` | `false` | point cloud |
| [`kulibinai/cadreasoner-cm`](https://huggingface.co/kulibinai/cadreasoner-cm) | `true` | `true` | point cloud + multi-view renders |

Both checkpoints use the `Cadrille` architecture defined in [`cadrille.py`](cadrille.py): a Qwen2-VL
backbone plus a Fourier point-cloud encoder. The point embeddings are scattered into a run of pad
tokens prepended to the prompt, so these checkpoints **cannot** be loaded with a bare
`Qwen2VLForConditionalGeneration.from_pretrained` — use the scripts in this directory.

The image-only model ([`kulibinai/cadreasoner`](https://huggingface.co/kulibinai/cadreasoner)) is
unaffected; it keeps using the scripts in the repository root.

## Layout

```text
pc_cm/
├── cadrille.py                     # Cadrille model (Qwen2-VL + FourierPointEncoder)
├── dataset.py                      # hybrid PC / image dataset and collate functions
├── pc_utils.py                     # discrepancy-aware point-cloud sampling
├── visualization.py                # multi-view renders (used when --use_img true)
├── augs.py                         # image augmentations used by visualization.py
├── utils.py                        # CadQuery execution and mesh generation helpers
├── train.py                        # curriculum training entry point
├── generate_refinement_samples.py  # per-group refinement sample generation
├── test.py                         # iterative inference
└── evaluate.py                     # metric computation
```

## Point-cloud conditioning

`pc_utils.make_pc_far` builds the point cloud from the **discrepancy** between the target mesh and
the current prediction, rather than from the target alone: it samples both surfaces, keeps the
points whose distance to the opposite shape exceeds a percentile threshold, and reduces them with
farthest point sampling. Each of the two halves contributes `n_points` points — `n_points` from
GT→pred and `n_points` from pred→GT — which is why the prompt reserves `2 * n_points` pad tokens.
On the first iteration, where no prediction exists yet, the bounding-box centre is used instead.

`--n_points` must match the value used in training (**128** for both released checkpoints).

## Inference

```bash
# point-cloud model
python3 test.py \
    --dataset <test_dataset> \
    --checkpoint kulibinai/cadreasoner-pc \
    --use_pc true --use_img false \
    --n_points 128 \
    --n_iters 3 \
    --outdir <outdir>
```

```bash
# cross-modality model
python3 test.py \
    --dataset <test_dataset> \
    --checkpoint kulibinai/cadreasoner-cm \
    --use_pc true --use_img true \
    --n_points 128 \
    --n_iters 3 \
    --outdir <outdir>
```

`<test_dataset>` is either a local directory of `.stl` files or a Hugging Face dataset repo id:

* `maksimko123/deepcad_test_mesh`
* `maksimko123/fusion360_test_mesh`
* `kulibinai/mcb_test`
* `kulibinai/deepcad_test_scan`
* `kulibinai/fusion360_test_scan`
* `kulibinai/mcb_test_scan`

Inference shards the samples across all visible GPUs, one process per GPU, and requires at least
one. Predictions are written to `<outdir>/<dataset_name>/<iter>/<shape>/`, and each file is renamed
to its chamfer distance so that the next iteration can pick the best candidates.

## Evaluation

```bash
python3 evaluate.py --dataset <test_dataset> --pred_dir <outdir>/<dataset_name>
```

## Training

The curriculum runs over dataset groups `0, 1, 2` (see [`../data/README.md`](../data/README.md) for
preparing the split). Group 0 starts from `Qwen/Qwen2-VL-2B-Instruct`; each later group starts from
the previous group's final checkpoint.

```bash
# point-cloud model
torchrun --nproc_per_node <n_gpus> train.py \
    --dataset_dir <train_dataset_dir> \
    --use_pc true --use_img false \
    --n_points 128
```

```bash
# cross-modality model
torchrun --nproc_per_node <n_gpus> train.py \
    --dataset_dir <train_dataset_dir> \
    --use_pc true --use_img true \
    --n_points 128
```

Checkpoints and logs go to `runs/<experiment>/<experiment>_<timestamp>/`, with the per-group weights
under `model/<group>/final`. Pass `--run_dir` and `--groups` to resume an interrupted run, and
`--skip_generate_code` / `--skip_generate_meshes` to reuse refinement samples already on disk.

To regenerate refinement samples for a single group without training:

```bash
python3 generate_refinement_samples.py \
    --dataset_dir <train_dataset_dir> \
    --run_dir <run_dir> \
    --group <group> \
    --use_pc true --use_img false
```

## Dependencies

In addition to the root [`Dockerfile`](../Dockerfile), this directory needs:

```bash
pip install opencv-python rtree
```

`opencv-python` is used by [`augs.py`](augs.py), and `rtree` backs `trimesh`'s closest-point queries
in [`pc_utils.py`](pc_utils.py) — without it the code falls back to a KD-tree over mesh vertices,
which is less accurate on coarse meshes.
