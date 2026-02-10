#!/bin/bash

set -e  # Exit on any error

echo "Usage: $0 <dataset> <per_device_train_batch_size> <n_gpus>"

run_dir="$PWD/run"
buffer_dir="$run_dir/buffer"
dataset=$1
per_device_train_batch_size=$2
n_gpus=$3

echo "========================================="
echo "Starting training pipeline"
echo "Run directory: $run_dir"
echo "Buffer directory: $buffer_dir"
echo "Dataset: $dataset"
echo "per_device_train_batch_size: $per_device_train_batch_size"
echo "n GPUs: $n_gpus"
echo "========================================="

# Group 0
echo ""--- Processing Group 0 ---""
echo "Training model on group 0..."
checkpoint="Qwen/Qwen2-VL-2B-Instruct"
torchrun --standalone --nproc_per_node="$n_gpus" train_group.py --group="0" --run_dir="$run_dir" --buffer_dir="$buffer_dir" --dataset="$dataset" --checkpoint="$checkpoint" --per_device_train_batch_size="$per_device_train_batch_size" 2>&1 | tee "$run_dir/torchrun_log_0.log" || true


# Group 1
echo "--- Processing Group 1 ---"
echo "Generating code and meshes for group 1..."
checkpoint="${run_dir}/model/0/final"
python3 generate_refinement_samples.py --group="1" --run_dir="$run_dir" --buffer_dir="$buffer_dir" --dataset="$dataset" --checkpoint="$checkpoint"
echo "Training model on group 1..."
torchrun --standalone --nproc_per_node="$n_gpus" train_group.py --group="1" --run_dir="$run_dir" --buffer_dir="$buffer_dir" --dataset="$dataset" --checkpoint="$checkpoint" --per_device_train_batch_size="$per_device_train_batch_size" 2>&1 | tee "$run_dir/torchrun_log_0.log" || true


# Group 2
echo "--- Processing Group 2 ---"
echo "Generating code and meshes for group 2..."
checkpoint="${run_dir}/model/1/final"
python3 generate_refinement_samples.py --group="2" --run_dir="$run_dir" --buffer_dir="$buffer_dir" --dataset="$dataset" --checkpoint="$checkpoint"
echo "Training model on group 2..."
torchrun --standalone --nproc_per_node="$n_gpus" train_group.py --group="2" --run_dir="$run_dir" --buffer_dir="$buffer_dir" --dataset="$dataset" --checkpoint="$checkpoint" --per_device_train_batch_size="$per_device_train_batch_size" 2>&1 | tee "$run_dir/torchrun_log_0.log" || true


echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
