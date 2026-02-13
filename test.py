import multiprocessing
import os.path
import uuid
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from qwen_vl_utils import process_vision_info
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    GenerationConfig,
)

from dataset import CadReasonerImagesDataset, generation_collate_fn
from train_group import init_model
from utils import generate_meshes, split_list

warnings.filterwarnings("ignore", category=UserWarning, module='trimesh')


@dataclass
class DataSample:
    gt_mesh_path: str
    gt_py_path: str
    pred_mesh_path: str
    pred_py_path: str
    save_pred_path: str
    resample_idx: int


def generate_predictions_process(rank, checkpoint, samples, greedy, temperature, batch_size):
    model, processor = init_model(checkpoint)
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model.eval()

    dataset = CadReasonerImagesDataset(samples=samples, train=False, scale_gt=True, scale_pred=False)
    dataset.max_generated_code_len = 1500

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=generation_collate_fn,
        num_workers=2,
    )

    generation_config = GenerationConfig(
        max_new_tokens=1500,
        temperature=temperature,
        top_p=0.95,
        do_sample=True
    )

    if rank == 0:
        _iterator = tqdm(dataloader, desc="Code generation")
    else:
        _iterator = dataloader

    for batch in _iterator:
        batch_size_actual = len(batch['generated_code'])
        sample_indices = batch['index']

        messages = [
            [
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': batch['image'][i]},
                    {'type': 'text', 'text': batch['generated_code'][i]}
                ]}
            ] for i in range(batch_size_actual)
        ]

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        vision_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=texts,
            images=vision_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        )

        inputs = inputs.to(model.device)

        with torch.no_grad():
            if greedy:
                generated_ids = model.generate(
                    **inputs, max_new_tokens=1500,
                )
            else:
                generated_ids = model.generate(
                    **inputs, max_new_tokens=1500, generation_config=generation_config, top_k=1000, do_sample=True, temperature=temperature
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        for i in range(batch_size_actual):
            decoded_text = py_strings[i]
            sample = samples[sample_indices[i]]
            Path(sample.save_pred_path).parent.mkdir(parents=True, exist_ok=True)
            with open(f"{sample.save_pred_path}.py", "w") as f:
                f.write(decoded_text)

        del inputs, generated_ids, messages, vision_inputs, video_inputs


def generate_predictions(checkpoint, samples, greedy, temperature=1.2, batch_size=64):
    torch.cuda.empty_cache()

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("0 gpus")

    torch.multiprocessing.set_start_method("spawn", force=True)

    samples_per_gpu = split_list(samples, n_gpus)

    processes = []
    for rank in range(n_gpus):
        p = torch.multiprocessing.Process(
            target=generate_predictions_process,
            args=(rank, checkpoint, samples_per_gpu[rank], greedy, temperature, batch_size)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def create_samples(dataset_dir, outdir, iter_num, top_n, n_samples):
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    if iter_num == 1:
        samples = [
            DataSample(
                gt_py_path=None,
                gt_mesh_path=str(gt_mesh_path),
                pred_mesh_path="None",
                pred_py_path="None",
                save_pred_path=str(outdir / str(iter_num) / gt_mesh_path.stem / f"pred{resample_idx}"),
                resample_idx=resample_idx,
            ) for gt_mesh_path in dataset_dir.glob("*.stl") for resample_idx in range(n_samples)
        ]
    else:
        samples = []
        for gt_mesh_path in dataset_dir.glob("*.stl"):
            prev_iteration_pred_py_paths = list((outdir / str(iter_num - 1) / gt_mesh_path.stem).glob('*.py'))
            prev_iteration_pred_py_paths = sorted(prev_iteration_pred_py_paths, key=lambda x: float(x.stem) if is_float(x.stem) else float("inf"))
            prev_iteration_pred_py_paths = prev_iteration_pred_py_paths[:top_n]

            for pred_py_path in prev_iteration_pred_py_paths:
                for resample_idx in range(n_samples):
                    uuid4 = uuid.uuid4()
                    samples.append(
                        DataSample(
                            gt_py_path=None,
                            gt_mesh_path=str(gt_mesh_path),
                            pred_mesh_path=pred_py_path.with_suffix(".stl"),
                            pred_py_path=pred_py_path,
                            save_pred_path=str(outdir / str(iter_num) / gt_mesh_path.stem / f"{uuid4}"),
                            resample_idx=resample_idx,
                        )
                    )

    return samples


def compute_chamfer_distance(gt_mesh, pred_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


def get_cd(arg):
    gt_mesh_path, pred_mesh_path = arg
    cd = None

    try:
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        center = (pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0
        pred_mesh.apply_translation(-center)
        extent = np.max(pred_mesh.extents)
        if extent > 1e-7:
            pred_mesh.apply_scale(1.0 / extent)
        pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        gt_mesh = trimesh.load_mesh(gt_mesh_path)
        center = (gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0
        gt_mesh.apply_translation(-center)
        extent = np.max(gt_mesh.extents)
        if extent > 1e-7:
            gt_mesh.apply_scale(1.0 / extent)
        gt_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        cd = compute_chamfer_distance(gt_mesh, pred_mesh, n_points=30_000)
        cd = float(cd)
    except Exception as ex:
        print(ex)
        pass

    return dict(cd=cd, pred_mesh_path=pred_mesh_path)


def score_predictions_cd(samples):
    samples = [s for s in samples if Path(f"{s.save_pred_path}.stl").exists()]

    args = [(s.gt_mesh_path, f"{s.save_pred_path}.stl") for s in samples]

    with multiprocessing.Pool(processes=8) as pool:
        metrics = list(
            tqdm(pool.imap_unordered(get_cd, args), total=len(args), desc="Predictions scoring")
        )

    for m in metrics:
        if m["cd"] is None:
            continue
        score = m["cd"]
        pred_mesh_path = Path(m["pred_mesh_path"])
        pred_py_path = pred_mesh_path.with_suffix(".py")
        pred_mesh_path.rename(pred_mesh_path.with_stem(str(score)))
        pred_py_path.rename(pred_py_path.with_stem(str(score)))


def run(test_datasets, checkpoint, outdir, n_iters, top_n, n_samples, greedy):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in test_datasets:
        dataset_name = os.path.basename(dataset_dir)
        dataset_dir = Path(dataset_dir)
        for iter_num in range(1, n_iters + 1):
            print("---" * 10, dataset_name, f"iter_num = {iter_num}", "---" * 10)
            if greedy:
                samples = create_samples(dataset_dir=dataset_dir, outdir=outdir / dataset_name, iter_num=iter_num, top_n=top_n, n_samples=1)
                generate_predictions(checkpoint, samples, greedy=True)
            elif n_samples == 1:
                samples = create_samples(dataset_dir=dataset_dir, outdir=outdir / dataset_name, iter_num=iter_num, top_n=top_n, n_samples=n_samples)
                generate_predictions(checkpoint, samples, greedy=False, temperature=1.2)
            else:
                samples = create_samples(dataset_dir=dataset_dir, outdir=outdir / dataset_name, iter_num=iter_num, top_n=top_n, n_samples=n_samples)
                greedy_samples = [s for s in samples if s.resample_idx == 0]
                non_greedy_samples = [s for s in samples if s.resample_idx != 0]
                generate_predictions(checkpoint, greedy_samples, greedy=True)
                generate_predictions(checkpoint, non_greedy_samples, greedy=False, temperature=1.2)

            generate_meshes(
                py_strings_and_save_paths=[(f"{s.save_pred_path}.py", f"{s.save_pred_path}.stl") for s in samples],
                timeout=30,
            )

            score_predictions_cd(samples)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='The directory with .stl files.')
    parser.add_argument('--outdir', type=str, help='Directory to save predictions to.')
    parser.add_argument('--checkpoint', type=str, default="kulibinai/cadreasoner")
    parser.add_argument('--n_iters', type=int, default=5, help='Number of refinement iterations.')
    parser.add_argument('--greedy', default=False, type=lambda x: x.lower() == 'true',
                        help='Use greedy generation instead of generation with sampling.')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of best predictions from previous refinement iteration (when using sampling).')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samplings (when using sampling).')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    run(
        test_datasets=[args.dataset],
        checkpoint=args.checkpoint,
        outdir=args.outdir,
        n_iters=args.n_iters,
        top_n=args.top_n,
        n_samples=args.n_samples,
        greedy=args.greedy,
    )
