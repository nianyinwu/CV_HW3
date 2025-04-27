"""Evaluate a Instance Segmentation model """

import torch
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from torch.cuda.amp import autocast

def evaluation(
    device: torch.device,
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    gt_json: str,
    threshold: float
) -> tuple[torch.Tensor, float]:
    """
    Evaluate the model on the validation dataset.

    Args :
        device : Evaluation device.
        model : Trained model to evaluate.
        valid_loader : DataLoader for the validation set.
        gt_json : The path of ground truth annotation file.
        threshold : The score threshold to filter prediction.

    Returns:
        mean_ap : Mean Average Precision.
        acc : Accuracy.
    """

    # Evaluation mAP, Accuracy
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in tqdm(valid_loader, ncols=120):
            images = [img.to(device) for img in images]

            with autocast():
                outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = target["image_id"].item()

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
                        "category_id": int(label),
                        "bbox": bbox,
                        "segmentation": rle,
                        "score": float(score)
                    })

    if not results:
        print("No valid predictions above threshold.")
        return 0.0

    coco_gt = COCO(gt_json)
    coco_pred = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_ap = coco_eval.stats[0]  # AP@[IoU=0.50:0.95]
    ap50  = coco_eval.stats[1]

    return mean_ap, ap50
