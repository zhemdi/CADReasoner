# CADReasoner: Iterative Program Editing for CAD Reverse Engineering

**CADReasoner** is a training and inference codebase for **iterative CAD reverse engineering** with **vision–language models (VLMs)**.  
The method predicts a runnable **CadQuery** program and refines it over multiple iterations using **geometric discrepancy feedback** between the target shape and the current reconstruction.

**Accepted to CVPR 2026 Findings**  

**Paper:** https://arxiv.org/abs/2603.29847  
**Hugging Face paper page:** https://huggingface.co/papers/2603.29847  

**Models:**

| Model | Input modality | Code |
|---|---|---|
| [`kulibinai/cadreasoner`](https://huggingface.co/kulibinai/cadreasoner) | multi-view renders | repository root |
| [`kulibinai/cadreasoner-pc`](https://huggingface.co/kulibinai/cadreasoner-pc) | point cloud | [`pc_cm/`](pc_cm/) |
| [`kulibinai/cadreasoner-cm`](https://huggingface.co/kulibinai/cadreasoner-cm) | point cloud + multi-view renders | [`pc_cm/`](pc_cm/) |

---

## Overview

Traditional Image2CAD methods are typically **single-pass**: they generate a CAD program once and stop.  
CADReasoner instead follows an **iterative editing** paradigm:

1. generate an initial CadQuery program,
2. render the predicted shape,
3. compare it against the target geometry,
4. feed the discrepancy back into the model,
5. refine the program over several steps.

The model combines **multi-view renders** and **point-cloud information** to improve geometric alignment and recover fine details.

## Repository layout

```text
CADReasoner/
├── data/                  # dataset split utilities and preprocessing instructions
├── pc_cm/                 # point-cloud and cross-modality models (cadreasoner-pc / -cm)
├── scanning_simulation/   # scan-simulation pipeline and related scripts
├── dataset.py             # dataset loading
├── evaluate.py            # metric computation
├── test.py                # inference
├── train_group.py         # training entry point
├── training_curriculum.sh # curriculum launcher
├── utils.py               # utility functions
└── visualization.py       # rendering / visualization utilities
```

The scripts in the repository root drive the image-only model
([`kulibinai/cadreasoner`](https://huggingface.co/kulibinai/cadreasoner)). The point-cloud and
cross-modality models use a different architecture and live in [`pc_cm/`](pc_cm/) — see
[`pc_cm/README.md`](pc_cm/README.md).
Quick start

0) Installation

Install dependencies according to the provided Docker environment:

1) Data preparation

Download and preprocess the data following:

data/README.md

This directory contains dataset split and conversion utilities.

2) Training

To launch the training curriculum, run:
```bash
./training_curriculum.sh <train_dataset> <per_device_train_batch_size> <n_gpus>
```

3) Inference

To generate CadQuery predictions, run:
```bash
python3 test.py --dataset <test_dataset> --checkpoint kulibinai/CADReasoner --n_iters <n_iters> --outdir <outdir>
```

<test_dataset> can be one of:
*	maksimko123/deepcad_test_mesh
*	maksimko123/fusion360_test_mesh
*	kulibinai/mcb_test
*	kulibinai/deepcad_test_scan
*	kulibinai/fusion360_test_scan
*	kulibinai/mcb_test_scan
4) Evaluation

To compute evaluation metrics, run:
```bash
python3 evaluate.py --dataset <test_dataset> --pred_dir <pred_dir>
```
5) Point-cloud and cross-modality models

To train, run or evaluate `cadreasoner-pc` and `cadreasoner-cm`, use the scripts in
[`pc_cm/`](pc_cm/):
```bash
python3 pc_cm/test.py --dataset <test_dataset> --checkpoint kulibinai/cadreasoner-pc \
    --use_pc true --use_img false --n_points 128 --n_iters <n_iters> --outdir <outdir>
```
See [`pc_cm/README.md`](pc_cm/README.md) for the cross-modality variant and for training.

6) Scan simulation

The repository also includes a scan-simulation pipeline used for robustness experiments and evaluation under simulated scanning artifacts.

See:
```text
scanning_simulation/README.md
```
What you get
* Iterative CAD reconstruction with geometric feedback over multiple refinement steps
* Runnable CadQuery program generation
* Training, inference, and evaluation scripts for Image2CAD
* Scan-simulation utilities for robust evaluation
* Rendering / visualization tools for geometric comparison and model inputs

Links
* Paper: [arXiv:2603.29847](https://arxiv.org/abs/2603.29847)
* Hugging Face paper page: [CADReasoner on Hugging Face Papers](https://huggingface.co/papers/2603.29847)
* Models: [kulibinai/cadreasoner](https://huggingface.co/kulibinai/cadreasoner) ·
  [kulibinai/cadreasoner-pc](https://huggingface.co/kulibinai/cadreasoner-pc) ·
  [kulibinai/cadreasoner-cm](https://huggingface.co/kulibinai/cadreasoner-cm)

Citation
```bibtex
@InProceedings{Kabisov_2026_CVPR,
    author    = {Kabisov, Soslan and Kirichuk, Vsevolod and Volkov, Andrey and Barannikov, Marina and Savrasov, Gennadiy and Konushin, Anton and Kuznetsov, Andrey and Zhemchuzhnikov, Dmitrii},
    title     = {CADReasoner: Iterative Program Editing for CAD Reverse Engineering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {6143-6153}
}
```
