# CADReasoner: Iterative Program Editing for CAD Reverse Engineering

**CADReasoner** is a training and inference codebase for **iterative CAD reverse engineering** with **vision–language models (VLMs)**.  
The method predicts a runnable **CadQuery** program and refines it over multiple iterations using **geometric discrepancy feedback** between the target shape and the current reconstruction.

**Accepted to CVPR 2026 Findings**  

**Paper:** https://arxiv.org/abs/2603.29847  
**Hugging Face paper page:** https://huggingface.co/papers/2603.29847  
**Model:** https://huggingface.co/kulibinai/cadreasoner  

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
├── scanning_simulation/   # scan-simulation pipeline and related scripts
├── dataset.py             # dataset loading
├── evaluate.py            # metric computation
├── test.py                # inference
├── train_group.py         # training entry point
├── training_curriculum.sh # curriculum launcher
├── utils.py               # utility functions
└── visualization.py       # rendering / visualization utilities
```
Quick start

0) Installation

Install dependencies according to the provided Docker environment:

1) Data preparation

Download and preprocess the data following:

data/README.md

This directory contains dataset split and conversion utilities.

2) Training

To launch the training curriculum, run:
```python
./training_curriculum.sh <train_dataset> <per_device_train_batch_size> <n_gpus>
```

3) Inference

To generate CadQuery predictions, run:
```python
python3 test.py --dataset <test_dataset> --checkpoint <checkpoint> --n_iters <n_iters> --outdir <outdir>
```
4) Evaluation

To compute evaluation metrics, run:
```python
python3 evaluate.py --dataset <test_dataset> --pred_dir <pred_dir>
```
5) Scan simulation

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
* Model: [kulibinai/cadreasoner](https://huggingface.co/kulibinai/cadreasoner)￼

Citation
```bibtex
@article{kabisov2026cadreasoner,
  title={CADReasoner: Iterative Program Editing for CAD Reverse Engineering},
  author={Kabisov, Soslan and Kirichuk, Vsevolod and Volkov, Andrey and Savrasov, Gennadii and Barannikov, Marina and Konushin, Anton and Kuznetsov, Andrey and Zhemchuzhnikov, Dmitrii},
  journal={arXiv preprint arXiv:2603.29847},
  year={2026}
}
```
