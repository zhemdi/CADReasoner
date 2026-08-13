import multiprocessing
import os.path
import uuid
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import warnings

from tqdm import tqdm
import torch



from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    # Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    GenerationConfig,
)
from cadrille import Cadrille as Qwen2VLForConditionalGeneration


from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

from dataset import CadRefineHybridDataset, generation_collate_fn
from utils import generate_meshes, split_list


warnings.filterwarnings("ignore", category=UserWarning, module='trimesh')


def resolve_dataset_dir(dataset: str) -> tuple[Path, str]:
    """
    Returns:
        dataset_dir: local directory containing .stl files
        dataset_name: name used for output subfolder
    """
    dataset_path = Path(dataset)

    # local directory
    if dataset_path.exists():
        return dataset_path, dataset_path.name

    # Hugging Face dataset repo id
    if "/" in dataset:
        local_dir = snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            allow_patterns=["*.stl"],
        )
        return Path(local_dir), dataset.split("/")[-1]

    raise ValueError(
        f"Dataset '{dataset}' is neither an existing local directory nor a valid HF dataset repo id."
    )


N_POINTS = None
USE_PC = None
USE_IMG = None
DEFECT = None


@dataclass
class DataSample:
    gt_mesh_path: str
    gt_py_path: str
    pred_mesh_path: str
    # pred_py_str: str | None = None
    pred_py_path: str
    save_pred_path: str
    resample_idx: int


def init_model(checkpoint):
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        # min_pixels=256 * 28 * 28,
        # max_pixels=1280 * 28 * 28,
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
        # trust_remote_code=True,
        # device_map='auto'
    )#.to('cuda')

    return model, processor


def generate_predictions_process(rank, checkpoint, samples, greedy, temperature, batch_size,
                                 n_points, use_pc, use_img, defect):
    model, processor = init_model(checkpoint)
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model.eval()

    N_POINTS = n_points
    USE_PC = use_pc
    USE_IMG = use_img
    DEFECT = defect


    dataset = CadRefineHybridDataset(
        samples=samples,
        train=False,
        n_points = N_POINTS,
        use_pc = USE_PC,
        use_img = USE_IMG,
        apply_augs = False,
        noise_scale_pc = None# if DEFECT else 0.01,
    )
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

    refined_samples = []

    if rank == 0:
        _iterator = tqdm(dataloader, desc="Generating target code")
    else:
        _iterator = dataloader

    for batch in _iterator:
        batch_size_actual = len(batch['generated_code'])
        sample_indices = batch['index']

        point_clouds_batch = []
        if USE_PC and 'point_cloud' in batch:
            point_clouds_batch = [torch.tensor(pc) for pc in batch['point_cloud']]

        messages = []
        for i in range(batch_size_actual):
            user_content = []

            # Add the image when requested
            if USE_IMG and 'target_image' in batch:
                user_content.append({'type': 'image', 'image': batch['target_image'][i]})

            # Add the text
            user_content.append({'type': 'text', 'text': batch['generated_code'][i]})

            messages.append([{'role': 'user', 'content': user_content}])

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        if USE_PC:
            n_total_points = N_POINTS * 2
            points_inputs = ''.join(n_total_points * [processor.tokenizer.pad_token])
            texts = [points_inputs + text for text in texts]

        vision_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=texts,
            images=vision_inputs if USE_IMG else None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        if USE_PC:
            inputs['point_clouds'] = torch.stack(point_clouds_batch).to(model.device)
            inputs['is_pc'] = torch.tensor([True] * batch_size_actual, dtype=torch.bool).to(model.device)
        else:
            inputs['is_pc'] = torch.tensor([False] * batch_size_actual, dtype=torch.bool).to(model.device)

        if USE_IMG:
            inputs['is_img'] = torch.tensor([True] * batch_size_actual, dtype=torch.bool).to(model.device)
        else:
            inputs['is_img'] = torch.tensor([False] * batch_size_actual, dtype=torch.bool).to(model.device)

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

        # Free the batch tensors
        del inputs, generated_ids, messages, vision_inputs, video_inputs
        # torch.cuda.empty_cache()


def generate_predictions(checkpoint, samples, greedy, temperature=1.2, batch_size=64):
    torch.cuda.empty_cache()

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("0 gpus")

    # Setup multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    samples_per_gpu = split_list(samples, n_gpus)

    # Spawn one process per GPU
    processes = []
    for rank in range(n_gpus):
        p = torch.multiprocessing.Process(
            target=generate_predictions_process,
            args=(rank, checkpoint, samples_per_gpu[rank], greedy, temperature, batch_size,
                  N_POINTS, USE_PC, USE_IMG, DEFECT)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def create_samples(dataset_dir, outdir, iter_num, n_samples):
    # iter_num - what predict
    # out_dataset_dir = Path(outdir) / os.path.basename(dataset_dir)
    # pred_py_dir = out_dataset_dir / "pred_py"
    # pred_stl_dir = out_dataset_dir / "pred_stl"
    # pred_py_dir.mkdir(parents=True, exist_ok=False)
    # pred_stl_dir.mkdir(parents=True, exist_ok=False)
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
            prev_iteration_pred_py_paths = prev_iteration_pred_py_paths[:n_samples]

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


def score_predictions_vis_metric(samples):
    score_samples = []
    for sample in samples:
        if Path(f"{sample.save_pred_path}.stl").exists():
            score_samples.append(
                DataSample(
                    gt_py_path=None,
                    gt_mesh_path=sample.gt_mesh_path,
                    pred_mesh_path=f"{sample.save_pred_path}.stl",
                    pred_py_path=f"{sample.save_pred_path}.py",
                    save_pred_path=None,
                    resample_idx=sample.resample_idx,
                )
            )

    dataset = CadRefineHybridDataset(samples=score_samples, apply_augs=False, train=False)
    dataset.max_generated_code_len = 1500

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        collate_fn=generation_collate_fn,
        num_workers=2,
    )

    for batch in tqdm(dataloader, desc="Scoring predictions"):
        batch_size_actual = len(batch['generated_code'])
        sample_indices = batch['index']
        for i in range(batch_size_actual):
            sample = score_samples[sample_indices[i]]
            if not Path(sample.pred_mesh_path).exists():
                continue

            vis = batch['target_image'][i]
            image = np.array(vis)[: 14 * 17 * 3, ...]
            image = image.astype(dtype=np.int64)
            diff = image[..., 0] - image[..., 1]
            score = (diff ** 2).sum()

            Path(sample.pred_mesh_path).rename(Path(sample.pred_mesh_path).with_stem(str(score)))
            Path(sample.pred_py_path).rename(Path(sample.pred_py_path).with_stem(str(score)))
            vis.save(str(Path(sample.pred_py_path).with_name(f"{score}.png"))) ########################################


def compute_chamfer_distance(gt_mesh, pred_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


def get_cd(arg):
    gt_mesh_path, pred_mesh_path = arg
    cd = None

    try:  # apply_transform fails for some reason; or mesh path can not exist
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        center = (pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0
        pred_mesh.apply_translation(-center)
        extent = np.max(pred_mesh.extents)
        if extent > 1e-7:
            pred_mesh.apply_scale(1.0 / extent)
        pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
        # pred_mesh = trimesh.load_mesh(pred_mesh_path)
        # pred_mesh.apply_scale(1.0 / 200)
        # pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        gt_mesh = trimesh.load_mesh(gt_mesh_path)

        cd = compute_chamfer_distance(gt_mesh, pred_mesh, n_points=50_000)
        cd = float(cd)
    except Exception as ex:
        # print(ex)
        pass

    return dict(cd=cd, pred_mesh_path=pred_mesh_path)


def score_predictions_cd(samples):
    samples = [s for s in samples if Path(f"{s.save_pred_path}.stl").exists()]

    args = [(s.gt_mesh_path, f"{s.save_pred_path}.stl") for s in samples]

    with multiprocessing.Pool(processes=8) as pool:
        metrics = list(
            tqdm(pool.imap_unordered(get_cd, args), total=len(args), desc="Scoring predictions")
        )

    for m in metrics:
        if m["cd"] is None:
            continue
        score = m["cd"]
        pred_mesh_path = Path(m["pred_mesh_path"])
        pred_py_path = pred_mesh_path.with_suffix(".py")
        pred_mesh_path.rename(pred_mesh_path.with_stem(str(score)))
        pred_py_path.rename(pred_py_path.with_stem(str(score)))


def run(test_datasets, checkpoint, outdir, n_iters, n_samples):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset in test_datasets:
        dataset_dir, dataset_name = resolve_dataset_dir(dataset)
        for iter_num in range(1, n_iters + 1):
            print("###" * 8, dataset_name, f"iter_num = {iter_num}", "###" * 8)
            samples = create_samples(dataset_dir=dataset_dir, outdir=outdir / dataset_name, iter_num=iter_num, n_samples=n_samples)
            greedy_samples = [s for s in samples if s.resample_idx == 0]
            non_greedy_samples = [s for s in samples if s.resample_idx != 0]
            generate_predictions(checkpoint, greedy_samples, greedy=True)
            generate_predictions(checkpoint, non_greedy_samples, greedy=False, temperature=1.2)

            generate_meshes(
                py_strings_and_save_paths=[(f"{s.save_pred_path}.py", f"{s.save_pred_path}.stl") for s in samples]
            )

            score_predictions_cd(samples)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='A directory with .stl files, or a Hugging Face dataset repo id.')
    parser.add_argument('--checkpoint', type=str, default='kulibinai/cadreasoner-pc',
                        help='kulibinai/cadreasoner-pc (point cloud) or kulibinai/cadreasoner-cm (cross-modality).')
    parser.add_argument('--outdir', required=True, type=str, help='Directory to save predictions to.')
    parser.add_argument('--n_iters', type=int, default=3, help='Number of refinement iterations.')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samplings per shape.')
    parser.add_argument('--n_points', type=int, default=128,
                        help='Number of points in the point cloud. Must match the value used in training.')
    parser.add_argument('--use_pc', required=True, type=lambda x: x.lower() == 'true', help='Use point cloud (true/false)')
    parser.add_argument('--use_img', required=True, type=lambda x: x.lower() == 'true', help='Use images (true/false)')
    args = parser.parse_args()

    if not args.use_pc and not args.use_img:
        parser.error('At least one of --use_pc / --use_img must be true.')

    global N_POINTS, USE_PC, USE_IMG, DEFECT
    N_POINTS = args.n_points
    USE_PC = args.use_pc
    USE_IMG = args.use_img
    DEFECT = None

    return args


if __name__ == '__main__':
    args = parse_args()

    run(
        test_datasets=[args.dataset],
        checkpoint=args.checkpoint,
        outdir=args.outdir,
        n_iters=args.n_iters,
        n_samples=args.n_samples,
    )
