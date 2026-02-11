from dataclasses import dataclass
import os.path

import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from visualization import Plotter


@dataclass
class DataSample:
    gt_mesh_path: str
    gt_py_path: str
    pred_mesh_path: str
    pred_py_path: str


class CadReasonerImagesDataset(Dataset):
    def __init__(self, samples: list[DataSample], train=True, scale_gt=False, scale_pred=False):
        super().__init__()
        self.samples = samples
        self.plotter = None
        self.max_generated_code_len = 1200
        self.train = train
        self.scale_gt = scale_gt
        self.scale_pred = scale_pred

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.plotter is None:
            self.plotter = Plotter(scale_gt=self.scale_gt, scale_pred=self.scale_pred)

        try:
            if sample.pred_py_path and os.path.exists(sample.pred_mesh_path) and os.path.exists(sample.pred_py_path):
                try:
                    image = self.plotter.get_img(sample.gt_mesh_path, sample.pred_mesh_path)
                except Exception as e:
                    print("Error in visualization:", e)
                    self.plotter.reload()
                    image = self.plotter.get_img(sample.gt_mesh_path, None)
            else:
                image = self.plotter.get_img(sample.gt_mesh_path, None)
        except Exception as e:
            print("Error in visualization:", e)
            self.plotter.reload()
            return None

        generated_code = "import cadquery as cq\n"
        if sample.pred_py_path and os.path.exists(sample.pred_py_path):
            with open(sample.pred_py_path, 'r', encoding='utf-8') as f:
                generated_code = f.read()

        if len(generated_code) > self.max_generated_code_len:
            generated_code = generated_code[:self.max_generated_code_len]

        if self.train:
            with open(sample.gt_py_path, 'r', encoding='utf-8') as f:
                target_code = f.read()
        else:
            target_code = None

        return {
            'target_code': target_code,
            'generated_code': generated_code,
            'image': image,
            'index': index,
        }


def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn_for_sft(batch, processor):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    messages = []
    for item in batch:
        message = [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': item['image']},
                {'type': 'text', 'text': item['generated_code']}
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': item['target_code']}
            ]}
        ]
        messages.append(message)

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0] + 2:begin_end_indexs[1] + 1] = \
                ids_list[begin_end_indexs[0] + 2:begin_end_indexs[1] + 1]
        labels_list.append(label_ids)
    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    inputs['labels'] = labels_ids

    return inputs


def generation_collate_fn(batch):
    batch = [item for item in batch if item is not None]

    collated_batch = {
        'target_code': [],
        'generated_code': [],
        'image': [],
        'index': []
    }

    for item in batch:
        collated_batch['target_code'].append(item['target_code'])
        collated_batch['generated_code'].append(item['generated_code'])
        collated_batch['image'].append(item['image'])
        collated_batch['index'].append(item['index'])

    return collated_batch
