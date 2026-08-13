import os.path
from dataclasses import dataclass

import trimesh
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from pc_utils import make_pc_far
from cadrille import find_assistant_content_sublist_indexes
from visualization import Plotter


def normalize_to_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mn, mx = mesh.bounds
    center = (mn + mx) * 0.5
    scale = (mx - mn).max() / 2.0
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    return mesh

def normalize_pred_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_scale(1.0 / 100.0)
    return mesh



@dataclass
class DataSample:
    gt_mesh_path: str
    gt_py_path: str
    pred_mesh_path: str
    pred_py_path: str


class CadRefineHybridDataset(Dataset):
    """
    Hybrid dataset supporting both point clouds and images.

    Args:
        samples: list of DataSample
        n_points: number of points in the point cloud (only used when use_pc=True)
        use_pc: whether to use point clouds
        use_img: whether to use images
        apply_augs: whether to augment the images (only used when use_img=True)
        thresh_percentile: percentile used to filter points (only used when use_pc=True)
    """
    def __init__(
            self,
            samples: list[DataSample],
            n_points: int = 128,
            use_pc: bool = True,
            use_img: bool = True,
            apply_augs: bool = False,
            thresh_percentile: int = 90,
            train=True,
            noise_scale_pc=0.01,
    ):
        super().__init__()

        if not use_pc and not use_img:
            raise ValueError("At least one of use_pc / use_img must be True")

        self.samples = samples
        self.max_generated_code_len = 1200
        self.n_points_per_cloud = n_points
        self.thresh_percentile = thresh_percentile
        self.use_pc = use_pc
        self.use_img = use_img
        self.apply_augs = apply_augs
        self.train = train
        self.noise_scale_pc = noise_scale_pc

        # The Plotter is only instantiated when images are needed
        self.plotter = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, train=True):
        sample = self.samples[index]

        if self.train:
            with open(sample.gt_py_path, 'r', encoding='utf-8') as f:
                target_code = f.read()
        else:
            target_code = None

        result = {
            'target_code': target_code,
            'generated_code': None,
            'index': index,
        }

        # Point cloud generation
        if self.use_pc:
            try:
                # Load the ground-truth mesh
                gt_mesh = trimesh.load_mesh(sample.gt_mesh_path)
                gt_mesh = self._augment_pc(gt_mesh)
                gt_mesh = normalize_to_unit_cube(gt_mesh)

                pred_mesh = None
                try:
                    pred_mesh = trimesh.load_mesh(sample.pred_mesh_path)
                    if not isinstance(pred_mesh, trimesh.Trimesh) or pred_mesh.vertices.shape[0] == 0:
                        pred_mesh = None
                    elif pred_mesh is not None:
                        pred_mesh = normalize_pred_mesh(pred_mesh)
                except Exception:
                    pred_mesh = None

                point_cloud = make_pc_far(
                    gt_mesh,
                    pred_mesh,
                    m_each=self.n_points_per_cloud,
                    thresh_percentile=self.thresh_percentile
                )
                result['point_cloud'] = point_cloud

            except Exception as e:
                print(f"Point cloud generation failed for {sample.gt_mesh_path}: {e}")
                return None

        # Image generation
        if self.use_img:
            try:
                if self.plotter is None:
                    self.plotter = Plotter()

                if os.path.exists(sample.pred_mesh_path):
                    try:
                        target_image = self.plotter.get_img(
                            sample.gt_mesh_path,
                            sample.pred_mesh_path,
                            apply_augs=self.apply_augs
                        )
                    except Exception as e:
                        print("Error in visualization:", e)
                        self.plotter.reload()
                        target_image = self.plotter.get_img(sample.gt_mesh_path, None, apply_augs=self.apply_augs)
                else:
                    target_image = self.plotter.get_img(
                        sample.gt_mesh_path,
                        None,
                        apply_augs=self.apply_augs
                    )
                result['target_image'] = target_image

            except Exception as e:
                print(f"Visualization failed for {sample.gt_mesh_path}: {e}")
                if self.plotter is not None:
                    self.plotter.reload()
                return None

        # Load the previously generated code
        generated_code = "import cadquery as cq\n"
        if os.path.exists(sample.pred_py_path):
            with open(sample.pred_py_path, 'r', encoding='utf-8') as f:
                generated_code = f.read()

        if len(generated_code) > self.max_generated_code_len:
            generated_code = generated_code[:self.max_generated_code_len]

        result['generated_code'] = generated_code

        return result

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh


def collate_fn_for_sft(batch, processor, use_pc=True, use_img=True, n_points=128):
    """
    Hybrid collate function for training.

    Args:
        batch: batch of samples
        processor: processor used for tokenization
        use_pc: whether to use point clouds
        use_img: whether to use images
        n_points: number of points per point cloud
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("The whole batch is None!")

    messages = []
    point_clouds_batch = []
    vision_messages = []

    is_pc = [1 if use_pc else 0] * len(batch)
    is_img = [1 if use_img else 0] * len(batch)

    for i, item in enumerate(batch):
        # Assemble the chat messages
        user_content = []

        # Add the image when requested
        if use_img and 'target_image' in item:
            user_content.append({'type': 'image', 'image': item['target_image']})

        # Add the text
        user_content.append({'type': 'text', 'text': item['generated_code']})

        message = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': item['target_code']}
            ]}
        ]
        messages.append(message)
        vision_messages.append(message)

        # Collect point clouds when requested
        if use_pc and 'point_cloud' in item:
            pc_target = torch.tensor(item['point_cloud'])
            point_clouds_batch.append(pc_target)

    # Apply the chat template
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]

    # Prepend pad tokens that the point cloud embeddings will be scattered into
    if use_pc:
        n_total_points = n_points * 2
        points_inputs = ''.join(n_total_points * [processor.tokenizer.pad_token])
        texts = [points_inputs + text for text in texts]

    # Process the vision inputs
    vision_inputs, video_inputs = process_vision_info(vision_messages)

    # Tokenize
    inputs = processor(
        text=texts,
        images=vision_inputs if use_img else None,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    # Add the point cloud tensors to the batch
    if use_pc:
        inputs['point_clouds'] = torch.stack(point_clouds_batch)
        inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)

    # Add the image flag
    if use_img:
        inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)

    # Build the labels
    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)
    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    inputs['labels'] = labels_ids

    if 'is_pc' not in inputs or inputs['is_pc'] is None:
        inputs['is_pc'] = torch.tensor([False] * len(batch), dtype=torch.bool)

    if 'is_img' not in inputs or inputs['is_img'] is None:
        inputs['is_img'] = torch.tensor([False] * len(batch), dtype=torch.bool)

    return inputs


def generation_collate_fn(batch):
    """
    Collate a batch for the generation stage.
    Supports both point clouds and images.
    """
    batch = [item for item in batch if item is not None]

    collated_batch = {
        'target_code': [],
        'generated_code': [],
        'index': []
    }

    has_pc = batch and 'point_cloud' in batch[0]
    has_img = batch and 'target_image' in batch[0]

    if has_pc:
        collated_batch['point_cloud'] = []
    if has_img:
        collated_batch['target_image'] = []

    for item in batch:
        collated_batch['target_code'].append(item['target_code'])
        collated_batch['generated_code'].append(item['generated_code'])
        collated_batch['index'].append(item['index'])

        if has_pc:
            collated_batch['point_cloud'].append(item['point_cloud'])
        if has_img:
            collated_batch['target_image'].append(item['target_image'])

    collated_batch['is_pc'] = torch.tensor([1 if has_pc else 0] * len(batch), dtype=torch.bool)
    collated_batch['is_img'] = torch.tensor([1 if has_img else 0] * len(batch), dtype=torch.bool)

    return collated_batch
