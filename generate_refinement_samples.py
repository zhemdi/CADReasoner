import argparse
import logging
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    GenerationConfig,
)

from dataset import CadRefineImagesDataset, generation_collate_fn, DataSample
from train_group import init_model
from utils import generate_meshes, split_list


warnings.filterwarnings("ignore", category=UserWarning, module='trimesh')


logger = None


def setup_logger(run_dir):
    global logger

    log_file = str(Path(run_dir) / "log.log")
    i = 0
    while os.path.exists(log_file):
        i += 1
        log_file = str(Path(run_dir) / f"log{i}.log")

    # Configure logging to write to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Use the logger
    logger = logging.getLogger()


def generate_for_refinement_iteration_worker(rank, samples, model_state_dict, results_dict, iteration_num, batch_size):
    model, processor = init_model("Qwen/Qwen2-VL-2B-Instruct")
    model.load_state_dict(model_state_dict)
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model.eval()

    dataset = CadRefineImagesDataset(samples=samples, scale_gt=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=generation_collate_fn,
        num_workers=2,
    )

    temperature = 1.9
    top_k = 4
    generation_config = GenerationConfig(
        max_new_tokens=1100,
        temperature=temperature,
        top_k=top_k,
        do_sample=True
    )

    refined_samples = []

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
            generated_ids = model.generate(
                **inputs, max_new_tokens=1100,
                do_sample=True, generation_config=generation_config, top_k=top_k, temperature=temperature,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        for i in range(batch_size_actual):
            sample = samples[sample_indices[i]]
            data_sample = DataSample(
                gt_py_path=sample.gt_py_path,
                gt_mesh_path=sample.gt_mesh_path,
                pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", sample.pred_mesh_path),
                pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", sample.pred_py_path),
            )
            refined_samples.append(data_sample)
            with open(data_sample.pred_py_path, "w") as f:
                f.write(py_strings[i])

        del inputs, generated_ids, messages, vision_inputs, video_inputs

    results_dict[rank] = refined_samples


def generate_for_refinement_iteration(model, samples, iteration_num, batch_size=64):
    torch.cuda.empty_cache()

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("0 gpus")

    torch.multiprocessing.set_start_method("spawn", force=True)
    manager = torch.multiprocessing.Manager()
    results_dict = manager.dict()

    model_state_dict = model.state_dict()

    samples_per_gpu = split_list(samples, n_gpus)

    # Spawn one process per GPU
    processes = []
    for rank in range(n_gpus):
        p = torch.multiprocessing.Process(
            target=generate_for_refinement_iteration_worker,
            args=(rank, samples_per_gpu[rank], model_state_dict, results_dict, iteration_num, batch_size)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    refinement_samples = [s for rank in range(n_gpus) for s in results_dict[rank]]

    return refinement_samples


def generate(groups: list[int], buffer_dir: str, dataset_dir: str, checkpoint: str):
    model, processor = init_model(checkpoint)
    model.to("cuda")

    dataset_dir = Path(dataset_dir)
    buffer_dir = Path(buffer_dir)

    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Start generating refinement samples.')

    for group in groups:
        logger.info("---" * 10 + str(group) + "---" * 10)

        train_mesh_buffer_dir = buffer_dir / str(group) / "train" / "pred_stl"
        train_py_buffer_dir = buffer_dir / str(group) / "train" / "pred_py"
        val_mesh_buffer_dir = buffer_dir / str(group) / "val" / "pred_stl"
        val_py_buffer_dir = buffer_dir / str(group) / "val" / "pred_py"

        train_mesh_buffer_dir.mkdir(parents=True, exist_ok=True)
        val_mesh_buffer_dir.mkdir(parents=True, exist_ok=True)

        train_py_buffer_dir.mkdir(parents=True, exist_ok=True)
        val_py_buffer_dir.mkdir(parents=True, exist_ok=True)

        train_dir = dataset_dir / str(group) / "train"
        train_samples = [
            DataSample(
                gt_py_path=str(gt_py_path),
                gt_mesh_path=str(gt_py_path.with_suffix(".stl")),
                pred_mesh_path=str(train_mesh_buffer_dir / f"{gt_py_path.stem}_it0.stl"),
                pred_py_path=str(train_py_buffer_dir / f"{gt_py_path.stem}_it0.py"),
            ) for gt_py_path in train_dir.glob("*.py")
        ]

        val_dir = dataset_dir / str(group) / "val"
        val_samples = [
            DataSample(
                gt_py_path=str(gt_py_path),
                gt_mesh_path=str(gt_py_path.with_suffix(".stl")),
                pred_mesh_path=str(val_mesh_buffer_dir / f"{gt_py_path.stem}_it0.stl"),
                pred_py_path=str(val_py_buffer_dir / f"{gt_py_path.stem}_it0.py"),
            ) for gt_py_path in val_dir.glob("*.py")
        ]

        refinement_train_samples = train_samples
        refinement_val_samples = val_samples

        for group_id in range(group):
            logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Code generation for train iteration {group_id + 1}')
            refinement_train_samples = generate_for_refinement_iteration(model, refinement_train_samples, group_id + 1)
            train_samples += refinement_train_samples
            logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Code generation for val iteration {group_id + 1}')
            refinement_val_samples = generate_for_refinement_iteration(model, refinement_val_samples, group_id + 1)
            val_samples += refinement_val_samples

        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Start of mesh generation.')
        generate_meshes(
            py_strings_and_save_paths=[(s.pred_py_path, s.pred_mesh_path) for s in train_samples]
        )
        generate_meshes(
            py_strings_and_save_paths=[(s.pred_py_path, s.pred_mesh_path) for s in val_samples]
        )
        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Mesh generation is finished.')

    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Generation of refinement samples is finished.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, type=int)
    parser.add_argument('--buffer_dir', required=True, type=str)
    parser.add_argument('--run_dir', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    setup_logger(args.run_dir)

    assert args.group > 0

    generate(
        groups=[int(args.group)],
        buffer_dir=args.buffer_dir,
        dataset_dir=args.dataset,
        checkpoint=args.checkpoint,
    )
