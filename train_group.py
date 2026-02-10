import argparse
import gc
import logging
import os
import re
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from dataset import CadRefineImagesDataset, generation_collate_fn, DataSample, collate_fn_for_sft

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


def init_model(checkpoint: str):
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
        trust_remote_code=True,
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


def run_training(
    model,
    processor,
    train_dataset,
    eval_dataset,
    output_dir,
    per_device_train_batch_size,
):
    torch.cuda.empty_cache()

    target_total_batch_size = 32
    num_gpus = torch.cuda.device_count()
    gradient_accumulation_steps = max(1, round(target_total_batch_size / (per_device_train_batch_size * num_gpus)))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=1,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        warmup_steps=1000,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=1000,
        save_total_limit=2,
        save_strategy="best",
        evaluation_strategy="steps",
        eval_steps=0.099,
        load_best_model_at_end=True,
        report_to=None,
    )

    n_gpus = torch.cuda.device_count()

    _n = len(train_dataset) % (per_device_train_batch_size * n_gpus)
    for _ in range(_n):
        train_dataset.samples.pop()

    _n = len(eval_dataset) % (per_device_train_batch_size * n_gpus)
    for _ in range(_n):
        eval_dataset.samples.pop()

    assert len(train_dataset) % n_gpus == 0
    assert len(train_dataset) % per_device_train_batch_size == 0
    assert len(eval_dataset) % n_gpus == 0
    assert len(eval_dataset) % per_device_train_batch_size == 0

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_fn_for_sft, processor=processor),
        tokenizer=processor,
        callbacks=[PrintToFileCallback()],
    )

    trainer.train()

    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    logger.info(f"Final model is saved to: {final_path}")
    torch.cuda.empty_cache()
    return trainer


def get_samples(dataset_dir, buffer_dir, group):
    train_mesh_buffer_dir = buffer_dir / str(group) / "train" / "pred_stl"
    train_py_buffer_dir = buffer_dir / str(group) / "train" / "pred_py"
    val_mesh_buffer_dir = buffer_dir / str(group) / "val" / "pred_stl"
    val_py_buffer_dir = buffer_dir / str(group) / "val" / "pred_py"

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

    for group_id in range(group):
        iteration_num = group_id + 1

        iteration_samples = [
            DataSample(
                gt_py_path=sample.gt_py_path,
                gt_mesh_path=sample.gt_mesh_path,
                pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", sample.pred_mesh_path),
                pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", sample.pred_py_path),
            ) for sample in train_samples
        ]
        train_samples += iteration_samples

        iteration_samples = [
            DataSample(
                gt_py_path=sample.gt_py_path,
                gt_mesh_path=sample.gt_mesh_path,
                pred_mesh_path=re.sub(r'(\d+)\.stl$', f"{iteration_num}.stl", sample.pred_mesh_path),
                pred_py_path=re.sub(r'(\d+)\.py$', f"{iteration_num}.py", sample.pred_py_path),
            ) for sample in val_samples
        ]
        val_samples += iteration_samples

    return train_samples, val_samples


def dump_training_samples(run_dir, group, train_samples, n_samples=100):
    if n_samples <= 0:
        return

    dir = run_dir / "buffer" / str(group) / "some_train_samples"
    dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CadRefineImagesDataset(samples=train_samples, scale_gt=True)

    dataloader = DataLoader(
        train_dataset,
        batch_size=n_samples,
        shuffle=True,
        drop_last=False,
        collate_fn=generation_collate_fn,
        num_workers=1,
        prefetch_factor=1,
    )

    for batch in dataloader:
        for idx, target_code, generated_code, input_image in \
                zip(batch["index"], batch["target_code"], batch["generated_code"], batch["image"]):
            sample_dir = dir / str(idx)
            sample_dir.mkdir(parents=True, exist_ok=True)

            with open(sample_dir / "target_code.py", "w") as f:
                f.write(target_code)

            with open(sample_dir / "generated_code.py", "w") as f:
                f.write(generated_code)

            input_image.save(str(sample_dir / "input_image.png"))
        break

    del train_dataset
    gc.collect()


def run_curriculum(
    groups: list[int],
    run_dir: str,
    buffer_dir: str,
    dataset_dir: str,
    checkpoint: str,
    per_device_train_batch_size: int,
):
    model, processor = init_model(checkpoint)
    model.to("cuda")

    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model is initialized.')

    dataset_dir = Path(dataset_dir)
    run_dir = Path(run_dir)
    buffer_dir = Path(buffer_dir)

    for group in groups:
        logger.info("---" * 10 + str(group) + "---" * 10)

        train_samples, val_samples = get_samples(dataset_dir=dataset_dir, buffer_dir=buffer_dir, group=group)

        train_dataset = CadRefineImagesDataset(samples=train_samples, scale_gt=True)
        eval_dataset = CadRefineImagesDataset(samples=val_samples, scale_gt=True)

        logger.info(f"Length of train dataset: {len(train_dataset)}.")
        logger.info(f"Length of val dataset: {len(eval_dataset)}.")

        # dump_training_samples(run_dir, group, train_samples)

        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Start training.')

        trainer = run_training(
            model, processor, train_dataset, eval_dataset,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=os.path.join(run_dir, "model", str(group)),
         )

        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Training finished.')

        model = trainer.model

        del trainer
        torch.cuda.empty_cache()
        del train_samples, val_samples
        del train_dataset, eval_dataset
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, type=int)
    parser.add_argument('--buffer_dir', required=True, type=str)
    parser.add_argument('--run_dir', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--per_device_train_batch_size', required=True, type=int, default=3)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    setup_logger(args.run_dir)

    run_curriculum(
        groups=[int(args.group)],
        run_dir=args.run_dir,
        buffer_dir=args.buffer_dir,
        dataset_dir=args.dataset,
        checkpoint=args.checkpoint,
        per_device_train_batch_size=args.per_device_train_batch_size,
    )
