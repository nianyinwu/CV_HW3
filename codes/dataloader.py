""" Dataloader utilities for training, validation, and testing. """

import os
import glob
from argparse import Namespace
import json
import numpy as np

import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
# import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
from skimage import io as sio


def data_transform(mode):
    '''
    Transform for training and validation datasets.
    '''
    # resize = v2.ScaleJitter(target_size=1024, antialias=True, resize_tolerance=0)
    if mode == 'train':
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            # v2.RandomAdjustSharpness(sharpness_factor=10, p=0.8),
            # v2.ToImage(),
            # v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    return None
    # v2.Compose([
    #     v2.RandomAdjustSharpness(sharpness_factor=10, p=1),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    # ])

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
        # cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


        masks = []
        labels = []

        paths = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.startswith("class") and f.endswith(".tif")
        ])


        for path in paths:
            cls = int(os.path.basename(path).replace("class", "").replace(".tif", ""))
            # print(os.path.basename(path), cls)
            mask = sio.imread(path)
            if resize_flag == 1:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            # mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

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

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

class TestDataset(Dataset):
    '''
    Dataset for inference.
    '''

    def __init__(self, root: str, map_json: str, transforms=None):
        self.root = root
        self.filename = sorted(os.listdir(self.root))
        self.transforms = transforms

        with open(map_json, "r", encoding='utf-8') as f:
            info_list = json.load(f)

        self.name_to_id = {
            info["file_name"]: info["id"]
            for info in info_list
        }


    def __len__(self):
        return len(self.filename)


    def __getitem__(self, idx):
        # print(self.filename[idx])
        image_id = self.name_to_id.get(self.filename[idx])
        data = os.path.join(self.root, self.filename[idx])

        image = cv2.imread(data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, image_id


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
        dataset = MedicalDataset(data_path, transforms=None)
        if mode == 'train':
            shuffle = True
    elif mode == 'test':
        data_path = os.path.join(args.data_path, mode)
        # transform = data_transform(mode)
        map_json = os.path.join(args.data_path, "test_image_name_to_ids.json")
        dataset = TestDataset(data_path, map_json, transforms=None)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=6,
        pin_memory=True,
        collate_fn=collate_fn
    )
