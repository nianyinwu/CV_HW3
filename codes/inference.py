""" Inference script for digit recognition """

import json
import argparse
import warnings
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from utils import tqdm_bar

import torch
from model import get_model
from dataloader import dataloader
from pycocotools import mask as mask_utils


# ignore warnings
warnings.filterwarnings('ignore')

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument(
        '--device',
        type=str,
        choices=["cuda", "cpu"],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='../datas',
        help='Path to input data'
    )
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        default='./best.pth',
        help='Path to model weights'
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./saved_model',
        help='the path of save the training model'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    return parser.parse_args()

def test(
    args: argparse.Namespace,
    test_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> List[Tuple[str, str]]:
    """
    Perform inference on the test set.

    Returns:
        List of tuples (image_name, predicted_class)
    """

    test_model.eval()
    test_model.to(args.device)

    results = []

    with torch.no_grad():
        for images, image_ids in (pbar := tqdm(data_loader, ncols=120)):
            images = [img.to(args.device) for img in images]
            outputs = test_model(images)

            for image_id, output in zip(image_ids, outputs):

                for box, label, score, mask in zip(
                    output["boxes"], output["labels"], output["scores"], output["masks"]
                ):

                    # Bounding box (COCO format: [x, y, w, h])
                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    # Binary mask → RLE encoding
                    binary_mask = (mask[0] > 0.5).cpu().numpy().astype("uint8")
                    # binary_mask = (mask.squeeze(0) > 0.5).cpu().numpy().astype("uint8")
                    arr = np.asfortranarray(binary_mask)
                    rle = mask_utils.encode(arr)
                    rle["counts"] = rle["counts"].decode("utf-8")  # for JSON export

                    results.append({
                        "image_id": image_id,
                        "bbox": bbox,
                        "score": float(score),
                        "category_id": int(label),
                        "segmentation": rle
                    })

                tqdm_bar('Test', pbar)

    return results


def make_json(save_path: str, predictions: list[tuple[str, str]]) -> None:
    """
    Generate prediction JSON file.
    """

    predictions = sorted(predictions, key=lambda x: x["image_id"])
    with open(f"{save_path}/test-results.json", "w", newline='', encoding='utf-8') as file:
        json.dump(predictions, file)

    print('Save test-result.json !!!')

if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    test_loader = dataloader(opt, 'test')

    # Load model
    model = get_model(num_classes=5)
    model.load_state_dict(torch.load(opt.weights))

    # Run inference
    pred_json = test(opt, model, test_loader)

    make_json(opt.save_path, pred_json)
