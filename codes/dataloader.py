""" Dataloader utilities for training, validation, and testing. """

import os
from argparse import Namespace
import json
import numpy as np

import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.ops import masks_to_boxes
from skimage import io as sio


def data_transform(mode):
    '''
    Transform for training and validation datasets.
    '''

    if mode == 'train':
        return v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=0.3),
            v2.RandomGrayscale(p=0.2),
            v2.ToDtype(torch.float32, scale=True)
        ])

    return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])


def resize(img):
    '''
    Resize the image if its height or width > 1024
    '''
    h, w = img.shape[:2]
    resize_flag = 0
    if max(h, w) > 1024:
        resize_flag = 1
        scale = 1024 / max(h, w)
        w = int(w * scale)
        h = int(h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return resize_flag, img, w, h

class MedicalDataset(Dataset):
    '''
    Dataset for training and validation.
    '''

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.datas_path = sorted([os.path.join(root, dir) for dir in os.listdir(root)])


    def __len__(self):
        return len(self.datas_path)


    def __getitem__(self, idx):
        data_path = self.datas_path[idx]

        img = cv2.imread(os.path.join(data_path, "image.tif"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_flag, img, width, height = resize(img)

        if self.transforms:
            img = self.transforms(img)

        masks = []
        labels = []

        paths = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.startswith("class") and f.endswith(".tif")
        ])


        for path in paths:
            cls = int(os.path.basename(path).replace("class", "").replace(".tif", ""))
            mask = sio.imread(path)
            if resize_flag == 1:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                binary_mask = (mask == inst_id).astype(np.uint8)
                masks.append(binary_mask)
                labels.append(cls)

        if len(masks) == 0:
            return self.__getitem__((idx + 1) % len(self))
        else:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = masks_to_boxes(masks)
            labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": masks.sum(dim=(1, 2)).float(),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64)
        }

        return img, target

class TestDataset(Dataset):
    '''
    Dataset for inference.
    '''

    def __init__(self, root: str, map_json: str):
        self.root = root
        self.filename = sorted(os.listdir(self.root))

        with open(map_json, "r", encoding='utf-8') as f:
            info_list = json.load(f)

        self.name_to_id = {
            info["file_name"]: info["id"]
            for info in info_list
        }


    def __len__(self):
        return len(self.filename)


    def __getitem__(self, idx):
        image_id = self.name_to_id.get(self.filename[idx])
        data = os.path.join(self.root, self.filename[idx])

        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, image_id


def collate_fn(batch):
    '''
    Collate function.
    '''
    return tuple(zip(*batch))

def dataloader(args: Namespace, mode: str) -> DataLoader:
    """
    Create dataloader based on the mode: train, val, or test.

    Args:
        args (Namespace): Command-line arguments containing data_path and batch_size.
        mode (str): Mode of the data loader ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader for the corresponding dataset.
    """

    dataset = None
    shuffle = False

    if mode in ['train', 'valid']:
        data_path = os.path.join(args.data_path, mode)
        transform = data_transform(mode)
        dataset = MedicalDataset(data_path, transforms=transform)
        if mode == 'train':
            shuffle = True
    elif mode == 'test':
        data_path = os.path.join(args.data_path, mode)
        map_json = os.path.join(args.data_path, "test_image_name_to_ids.json")
        dataset = TestDataset(data_path, map_json)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=6,
        pin_memory=True,
        collate_fn=collate_fn
    )
