import torch.distributed as dist
import logging
import os
import random
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import warnings
from functools import partial
import gc
from tqdm import tqdm
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    GenerationConfig,
)

from cadrille import Cadrille as Qwen2VLForConditionalGeneration
import numpy as np
from dataset import CadRefineHybridDataset, generation_collate_fn, DataSample, collate_fn_for_sft

from qwen_vl_utils import process_vision_info

#from dataset_iso import CadRefineImagesDataset, generation_collate_fn, DataSample, collate_fn_for_sft
from utils import generate_meshes, split_list
warnings.filterwarnings("ignore", category=UserWarning, module='trimesh')


N_POINTS = 128
USE_PC = True
USE_IMG = False


logger = logging.getLogger(__name__)
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


def init_model(checkpoint):
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        resized_width=14 * 17 * 2,
        resized_height=14 * 17 * 4,
        padding_side="left",
        trust_remote_code=True,
        use_fast=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()

    return model, processor


class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.logging_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


def run_training(model, processor, train_dataset, eval_dataset, output_dir, defect=False):
    torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count()
    per_gpu_batch_size = 4
    per_device_train_batch_size = per_gpu_batch_size
    per_device_eval_batch_size = per_gpu_batch_size
    target_total_batch_size = 32
    gradient_accumulation_steps = max(1, target_total_batch_size // (per_gpu_batch_size * num_gpus))

    if USE_PC and USE_IMG:
        model_prefix = "HYBRID"
    elif USE_PC:
        model_prefix = "PC"
    else:
        model_prefix = "IMG"
    group = int(output_dir.split('/')[-1])
    eval_steps = 450
    epochs = 2
    print('EVAL_STEPS', eval_steps)
    print('EPOCHS', epochs)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_gpu_batch_size,
        per_device_eval_batch_size=per_gpu_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,
        #dataloader_persistent_workers=True,
        #dataloader_prefetch_factor=1,
        num_train_epochs=epochs,
        #max_steps=9,
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        warmup_steps=1000,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=100,
        # save_strategy="steps",
        # save_steps=3200,
        save_total_limit=2,
        # bf16=True,
        save_strategy="best",
        # save_steps=3200,
        evaluation_strategy="steps",
        eval_steps=eval_steps,  # 3200,
        load_best_model_at_end=True,
        report_to=None,
        dataloader_drop_last=True,
    )

    n_gpus = torch.cuda.device_count()

    _n = len(train_dataset) % (per_device_train_batch_size * n_gpus)
    for _ in range(_n):
        train_dataset.samples.pop()

    _n = len(eval_dataset) % (per_device_eval_batch_size * n_gpus)
    for _ in range(_n):
        eval_dataset.samples.pop()

    assert len(train_dataset) % n_gpus == 0
    assert len(train_dataset) % per_device_train_batch_size == 0
    assert len(eval_dataset) % n_gpus == 0
    assert len(eval_dataset) % per_device_eval_batch_size == 0

    print("Initializing the Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_fn_for_sft, processor=processor, use_pc=USE_PC, use_img=USE_IMG, n_points=N_POINTS),
        tokenizer=processor,
        callbacks=[PrintToFileCallback()],
    )

    print("--- Training started ---")
    trainer.train()
    print("--- Training finished ---")

    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    print(f"Final model saved to {final_path}")
    torch.cuda.empty_cache()
    return trainer





def make_refine_iteration(model, processor, samples, iteration_num, use_pc=True, use_img=False, n_points=N_POINTS, batch_size=64):
    torch.cuda.empty_cache()

    # Reuse the current distributed context
    if dist.is_initialized():
        n_gpus = dist.get_world_size()
        rank = dist.get_rank()
    else:
        n_gpus = 1
        rank = 0

    samples_per_gpu = split_list(samples, n_gpus)
    my_samples = samples_per_gpu[rank]

    apply_augs = True if iteration_num == 1 else False
    dataset = CadRefineHybridDataset(
        samples=my_samples,
        n_points=n_points,
        use_pc=use_pc,
        use_img=use_img,
        apply_augs=apply_augs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=generation_collate_fn,
        num_workers=0,
    )

    temperature = 1.2
    generation_config = GenerationConfig(
        max_new_tokens=1500,
        temperature=temperature,
        top_p=0.95,
        do_sample=True
    )

    refined_samples = []

    if rank == 0:
        _iterator = tqdm(dataloader, desc="Generating target code")
    else:
        _iterator = dataloader

    n_total_points = n_points * 2

    for batch in _iterator:
        batch_size_actual = len(batch['generated_code'])
        sample_indices = batch['index']

        point_clouds_batch = []
        if use_pc and 'point_cloud' in batch:
            point_clouds_batch = [torch.tensor(pc) for pc in batch['point_cloud']]

        messages = []
        for i in range(batch_size_actual):
            user_content = []

            # Add the image when requested
            if use_img and 'target_image' in batch:
                user_content.append({'type': 'image', 'image': batch['target_image'][i]})

            # Add the text
            user_content.append({'type': 'text', 'text': batch['generated_code'][i]})

            messages.append([{'role': 'user', 'content': user_content}])

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        # Prepend pad tokens only when point clouds are used
        if use_pc:
            n_total_points = n_points * 2
            points_inputs = ''.join(n_total_points * [processor.tokenizer.pad_token])
            texts = [points_inputs + text for text in texts]

        vision_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=vision_inputs if use_img else None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        if use_pc:
            inputs['point_clouds'] = torch.stack(point_clouds_batch).to(model.device)
            inputs['is_pc'] = torch.tensor([True] * batch_size_actual, dtype=torch.bool).to(model.device)
        else:
            inputs['is_pc'] = torch.tensor([False] * batch_size_actual, dtype=torch.bool).to(model.device)

        if use_img:
            inputs['is_img'] = torch.tensor([True] * batch_size_actual, dtype=torch.bool).to(model.device)
        else:
            inputs['is_img'] = torch.tensor([False] * batch_size_actual, dtype=torch.bool).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=1100, generation_config=generation_config,
                top_k=1000, do_sample=True, temperature=temperature
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        for i in range(batch_size_actual):
            decoded_text = py_strings[i]
            sample = my_samples[sample_indices[i]]
            data_sample = DataSample(
                gt_py_path=sample.gt_py_path,
                gt_mesh_path=sample.gt_mesh_path,
                pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", sample.pred_mesh_path),
                pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", sample.pred_py_path),
            )
            refined_samples.append(data_sample)
            Path(data_sample.pred_py_path).parent.mkdir(parents=True, exist_ok=True)   # !!!!!!
            with open(data_sample.pred_py_path, "w") as f:
                f.write(decoded_text)

        del inputs, generated_ids, messages, vision_inputs, video_inputs

    # Gather the results from all GPUs
    #if dist.is_initialized():
    #    # every GPU wrote its own files, synchronize here
    #    dist.barrier()

    #    # gather all refined_samples on rank 0
    #    if rank == 0:
    #        all_refined_samples = refined_samples.copy()
    # Rebuild the full list from all GPUs
    #        for r in range(1, n_gpus):
    # Take the samples handled by this rank
    #            all_refined_samples.extend([
    #                DataSample(
    #                    gt_py_path=s.gt_py_path,
    #                    gt_mesh_path=s.gt_mesh_path,
    #                    pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", s.pred_mesh_path),
    #                    pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", s.pred_py_path),
    #                ) for s in samples_per_gpu[r]
    #            ])
    #        refined_samples = all_refined_samples

    #    dist.barrier()
    if dist.is_initialized():
        # First every rank writes its own files
        dist.barrier()

        # Then rebuild the full list on ALL ranks
        all_refined_samples = []
        for r in range(n_gpus):
            samples_for_rank = [
                DataSample(
                    gt_py_path=s.gt_py_path,
                    gt_mesh_path=s.gt_mesh_path,
                    pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", s.pred_mesh_path),
                    pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", s.pred_py_path),
                ) for s in samples_per_gpu[r]
            ]
            all_refined_samples.extend(samples_for_rank)

        refined_samples = all_refined_samples
        dist.barrier()

    return refined_samples


def dump_training_samples(run_dir, group, train_samples, n_samples=100):
    dir = run_dir / "buffer" / str(group) / "some_train_samples"
    dir.mkdir(parents=True, exist_ok=False)

    train_dataset = CadRefineHybridDataset(
        samples=train_samples,
        n_points=N_POINTS,
        use_pc=USE_PC,
        use_img=USE_IMG,
        apply_augs=False
    )

    for _ in range(n_samples):
        idx = random.randint(0, len(train_samples))
        sample = train_dataset[idx]
        if sample is None:
            continue

        sample_dir = dir / str(idx)
        sample_dir.mkdir(parents=True, exist_ok=True)

        with open(sample_dir / "target_code.py", "w") as f:
            f.write(sample["target_code"])

        with open(sample_dir / "generated_code.py", "w") as f:
            f.write(sample["generated_code"])

        if USE_PC and "point_cloud" in sample:
            np.save(str(sample_dir / "input_point_cloud.npy"), sample["point_cloud"])

        if USE_IMG and "target_image" in sample:
            sample["target_image"].save(str(sample_dir / "input_image.png"))

    #if USE_IMG and train_dataset.plotter is not None:
    #train_dataset.plotter.close()
    #train_dataset.plotter = None

    del train_dataset
    gc.collect()


def run_curriculum(groups: list[int], run_dir: str, dataset_dir: str, checkpoint: str,
                   continue_do_generate_code: bool = True, continue_do_generate_meshes: bool = True,
                   defect: bool = False):
    model, processor = init_model(checkpoint)

    #model.to("cuda")
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model initialized')

    dataset_dir = Path(dataset_dir)
    run_dir = Path(run_dir)

    for group in groups:
        print("---" * 10 + str(group) + "---" * 10)

        train_mesh_buffer_dir = run_dir / "buffer" / str(group) / "train" / "pred_stl"
        train_py_buffer_dir = run_dir / "buffer" / str(group) / "train" / "pred_py"
        val_mesh_buffer_dir = run_dir / "buffer" / str(group) / "val" / "pred_stl"
        val_py_buffer_dir = run_dir / "buffer" / str(group) / "val" / "pred_py"

        if not dist.is_initialized() or dist.get_rank() == 0:
            if continue_do_generate_meshes:
                train_mesh_buffer_dir.mkdir(parents=True, exist_ok=True)
                val_mesh_buffer_dir.mkdir(parents=True, exist_ok=True)

            if continue_do_generate_code:
                train_py_buffer_dir.mkdir(parents=True, exist_ok=True)
                val_py_buffer_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

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

        refined_train_samples = train_samples
        refined_val_samples = val_samples

        train_len = len(train_samples)
        val_len = len(val_samples)

        for group_id in range(group):
            if continue_do_generate_code:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: make_refine_iteration for train iteration {group_id + 1}')
                refined_train_samples = make_refine_iteration(model, processor, refined_train_samples, group_id + 1, use_pc=USE_PC, use_img=USE_IMG)
                train_samples += refined_train_samples
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: make_refine_iteration for val iteration {group_id + 1}')
                refined_val_samples = make_refine_iteration(model, processor, refined_val_samples, group_id + 1, use_pc=USE_PC, use_img=USE_IMG)
                val_samples += refined_val_samples
            else:
                iteration_num = group_id + 1

                # train
                iteration_samples = [None] * train_len
                for i in range(train_len):
                    sample = train_samples[i]
                    iteration_sample = DataSample(
                        gt_py_path=sample.gt_py_path,
                        gt_mesh_path=sample.gt_mesh_path,
                        pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", sample.pred_mesh_path),
                        pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", sample.pred_py_path),
                    )
                    iteration_samples[i] = iteration_sample
                train_samples += iteration_samples

                # train
                iteration_samples = [None] * val_len
                for i in range(val_len):
                    sample =val_samples[i]
                    iteration_sample = DataSample(
                        gt_py_path=sample.gt_py_path,
                        gt_mesh_path=sample.gt_mesh_path,
                        pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", sample.pred_mesh_path),
                        pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", sample.pred_py_path),
                    )
                    iteration_samples[i] = iteration_sample
                val_samples += iteration_samples

        if continue_do_generate_meshes:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Mesh generation started')
            generate_meshes(
                py_strings_and_save_paths=[(s.pred_py_path, s.pred_mesh_path) for s in train_samples]
            )
            generate_meshes(
                py_strings_and_save_paths=[(s.pred_py_path, s.pred_mesh_path) for s in val_samples]
            )
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Meshes generated')

        if dist.is_initialized():
            if dist.get_rank() == 0:
                dump_training_samples(run_dir, group, train_samples)
            dist.barrier()  # everyone waits here
        else:
            dump_training_samples(run_dir, group, train_samples)
        #dump_training_samples(run_dir, group, train_samples)

        train_dataset = CadRefineHybridDataset(
            samples=train_samples,
            n_points=N_POINTS,
            use_pc=USE_PC,
            use_img=USE_IMG,
            apply_augs=False
        )
        eval_dataset = CadRefineHybridDataset(
            samples=val_samples,
            n_points=N_POINTS,
            use_pc=USE_PC,
            use_img=USE_IMG,
            apply_augs=False,
            noise_scale_pc=None,
        )
        print('EVAL DATASET', len(eval_dataset))
        print(f"Found {len(train_dataset)} training samples.")
        print(f"Found {len(eval_dataset)} validation samples.")

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Training started')

        trainer = run_training(model, processor, train_dataset, eval_dataset, output_dir=f"{run_dir}/model/{group}", defect=defect)

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Training finished')

        model = trainer.model

        del trainer
        torch.cuda.empty_cache()
        del train_samples, val_samples, refined_train_samples, refined_val_samples
        del train_dataset, eval_dataset
        gc.collect()

        continue_do_generate_code = True
        continue_do_generate_meshes = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, type=str,
                        help='Training dataset root, split into curriculum groups (see data/README.md).')
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-VL-2B-Instruct',
                        help='Model to start from. Pass a run checkpoint to resume the curriculum.')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Directory for logs and checkpoints. Defaults to runs/<experiment>/<experiment>_<timestamp>.')
    parser.add_argument('--groups', type=int, nargs='+', default=[0, 1, 2],
                        help='Curriculum groups to train on, in order.')
    parser.add_argument('--n_points', type=int, default=128, help='Number of points in the point cloud.')
    parser.add_argument('--use_pc', type=lambda x: x.lower() == 'true', default=True,
                        help='Use point clouds (true/false).')
    parser.add_argument('--use_img', type=lambda x: x.lower() == 'true', default=False,
                        help='Use images (true/false). Set both flags to true for the cross-modality model.')
    parser.add_argument('--skip_generate_code', action='store_true',
                        help='Skip code generation for the first group (when resuming a run).')
    parser.add_argument('--skip_generate_meshes', action='store_true',
                        help='Skip mesh generation for the first group (when resuming a run).')
    args = parser.parse_args()

    if not args.use_pc and not args.use_img:
        parser.error('At least one of --use_pc / --use_img must be true.')

    global N_POINTS, USE_PC, USE_IMG
    N_POINTS = args.n_points
    USE_PC = args.use_pc
    USE_IMG = args.use_img

    return args


if __name__ == "__main__":
    args = parse_args()

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        local_rank = 0
        is_distributed = False

    run_dir = args.run_dir
    if run_dir is None:
        if USE_PC and USE_IMG:
            experiment_name = 'cadreasoner_cm'
        elif USE_PC:
            experiment_name = 'cadreasoner_pc'
        else:
            experiment_name = 'cadreasoner_img'

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        run_dir = os.path.join("runs", experiment_name, f"{experiment_name}_{timestamp}")

    if local_rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    if is_distributed:
        dist.barrier()

    if local_rank == 0:
        log_path = Path(run_dir) / "all_logs.log"
        sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stderr = sys.stdout
        setup_logger(run_dir)

    if is_distributed:
        dist.barrier()

    run_curriculum(
        groups=args.groups,
        run_dir=run_dir,
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        continue_do_generate_code=not args.skip_generate_code,
        continue_do_generate_meshes=not args.skip_generate_meshes,
        defect=False
    )

    if is_distributed:
        dist.destroy_process_group()