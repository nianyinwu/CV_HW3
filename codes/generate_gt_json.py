""" Generate ground truth annotation json file """
import os
import json
import numpy as np
from skimage import io as sio
from pycocotools import mask as mask_utils
from tqdm import tqdm

def generate_gt_json(root_dir, save_path, category_ids):
    """
    Generate ground truth annotation json file 
    """

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": f"class{cid}"} for cid in category_ids]
    }

    ann_id = 1
    image_id = 0

    sample_dirs = sorted(os.listdir(root_dir))
    for sample_name in tqdm(sample_dirs, desc="Generating GT JSON"):
        sample_path = os.path.join(root_dir, sample_name)
        image_path = os.path.join(sample_path, "image.tif")

        if not os.path.exists(image_path):
            continue

        image = sio.imread(image_path)
        height, width = image.shape[:2]

        coco["images"].append({
            "id": image_id,
            "file_name": f"{sample_name}/image.tif",
            "width": width,
            "height": height
        })

        for fname in os.listdir(sample_path):
            if not fname.startswith("class") or not fname.endswith(".tif"):
                continue

            cls_id = int(fname.replace("class", "").replace(".tif", ""))
            mask = sio.imread(os.path.join(sample_path, fname))

            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                binary_mask = (mask == inst_id).astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                # mask encode
                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                # bbox
                pos = np.where(binary_mask)
                xmin = int(np.min(pos[1]))
                xmax = int(np.max(pos[1]))
                ymin = int(np.min(pos[0]))
                ymax = int(np.max(pos[0]))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "segmentation": rle,
                    "area": int(binary_mask.sum()),
                    "bbox": bbox,
                    "iscrowd": 0
                })

                ann_id += 1

        image_id += 1

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"Saved COCO GT to {save_path} (images={len(coco['images'])}, anns={len(coco['annotations'])})")

if __name__ == "__main__":
    generate_gt_json(
        root_dir="./val_split",
        save_path="gt.json",
        category_ids=list(range(1, 5))
    )
