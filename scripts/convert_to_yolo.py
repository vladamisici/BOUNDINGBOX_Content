import os
import argparse
import json
import shutil

def yolo_format(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    return [x_center / img_w, y_center / img_h, w / img_w, h / img_h]

def process_split(json_path, split_name, image_root, out_images, out_labels, class_id=0):
    # Load annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # prepare output dirs
    img_out_dir = os.path.join(out_images, split_name)
    lbl_out_dir = os.path.join(out_labels, split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for ann in annotations:
        fname = ann['file_name']
        bbox = ann['bbox']  # [x, y, w, h]
        img_w = ann['width']
        img_h = ann['height']

        # compute YOLO coords
        yolo_box = yolo_format(bbox, img_w, img_h)

        # build source image path by split subfolder
        src_img = os.path.join(image_root, split_name, fname)
        dst_img = os.path.join(img_out_dir, fname)
        if not os.path.exists(src_img):
            print(f"WARNING: image not found: {src_img}")
            continue
        shutil.copy(src_img, dst_img)

        # write label file
        label_fname = os.path.splitext(fname)[0] + '.txt'
        lbl_path = os.path.join(lbl_out_dir, label_fname)
        with open(lbl_path, 'w') as lf:
            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in yolo_box)
            lf.write(line + "\n")

    print(f"Processed {len(annotations)} items for split '{split_name}'")

def main():
    parser = argparse.ArgumentParser(
        description='Convert aggregated JSON to YOLO labels and organize images')
    parser.add_argument('--train-json', required=True,
                        help='Aggregated JSON for train split')
    parser.add_argument('--val-json', required=True,
                        help='Aggregated JSON for val split')
    parser.add_argument('--image-root', required=True,
                        help='Root folder containing subfolders for train/val images')
    parser.add_argument('--out-images', default='data/images',
                        help='Output root for train/val images')
    parser.add_argument('--out-labels', default='data/labels',
                        help='Output root for train/val labels')
    parser.add_argument('--class-id', type=int, default=0,
                        help='YOLO class ID for content_box')
    args = parser.parse_args()

    # Process both splits
    process_split(args.train_json, 'train', args.image_root,
                  args.out_images, args.out_labels, args.class_id)
    process_split(args.val_json,   'val',   args.image_root,
                  args.out_images, args.out_labels, args.class_id)

if __name__ == '__main__':
    main()
