import os
import argparse
import json
from tqdm import tqdm

"""
toate PubLayNet COCO annotations intr-un "content_box" / image.

    python prepare_publaynet.py `
        --coco data/raw/publaynet/labels/publaynet/train.json `
        --classes text title list `
        --out data/annotations_publaynet_train.json
"""

def load_coco_annotations(path):
    with open(path, 'r') as f:
        coco = json.load(f)
    return coco


def get_category_ids(coco, target_names):
    name_to_id = {c['name']: c['id'] for c in coco['categories']}
    ids = []
    for name in target_names:
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            raise ValueError(f"Category '{name}' not found in COCO categories")
    return set(ids)


def aggregate_bboxes(coco, target_ids):
    # Map image_id - filename & dimensiuni
    images = {img['id']: img for img in coco['images']}
    # bboxes / image
    img_to_boxes = {img_id: [] for img_id in images}

    for ann in coco['annotations']:
        if ann['category_id'] in target_ids:
            img_to_boxes[ann['image_id']].append(ann['bbox'])  # [x, y, w, h]

    results = []
    for img_id, bboxes in tqdm(img_to_boxes.items(), desc='Aggregating bboxes'):
        if not bboxes:
            continue
        # Compute aggregated box
        x_mins = [b[0] for b in bboxes]
        y_mins = [b[1] for b in bboxes]
        x_maxs = [b[0] + b[2] for b in bboxes]
        y_maxs = [b[1] + b[3] for b in bboxes]
        x_min, y_min = min(x_mins), min(y_mins)
        x_max, y_max = max(x_maxs), max(y_maxs)
        w = x_max - x_min
        h = y_max - y_min
        img_info = images[img_id]
        results.append({
            'file_name': img_info['file_name'],
            'bbox': [x_min, y_min, w, h],
            'width': img_info['width'],
            'height': img_info['height'],
            'label': 'content_box'
        })
    return results


def main():
    parser = argparse.ArgumentParser(description='Pregateste PubLayNet pt content_box detection')
    parser.add_argument('--coco', required=True,
                        help='Path to COCO JSON annotations file')
    parser.add_argument('--classes', nargs='+', default=['text', 'title', 'list'],
                        help='nume categorii')
    parser.add_argument('--out', required=True,
                        help='Output path pt JSON annotations')
    args = parser.parse_args()

    coco = load_coco_annotations(args.coco)
    # target category IDs
    target_ids = get_category_ids(coco, args.classes)
    # aggregate
    aggregated = aggregate_bboxes(coco, target_ids)
    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved {len(aggregated)} content_box annotations to {args.out}")


if __name__ == '__main__':
    main()
