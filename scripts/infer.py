import os
import argparse
from ultralytics import YOLO
import cv2
import glob
from utils import draw_boxes


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with YOLOv8 content-box detector')
    parser.add_argument('--weights', required=True,
                        help='Path to model weights (.pt)')
    parser.add_argument('--source', required=True,
                        help='Image or directory to run inference on')
    parser.add_argument('--output', default='runs/inference',
                        help='Directory to save annotated images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    os.makedirs(args.output, exist_ok=True)

    # Collect image paths
    paths = []
    if os.path.isdir(args.source):
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.tif']:
            paths.extend(sorted(glob.glob(os.path.join(args.source, ext))))
    else:
        paths = [args.source]

    for img_path in paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # If it’s grayscale (H×W or H×W×1), convert to BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        results = model.predict(img, conf=args.conf)
        # extract boxes
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        labels = [f"{int(cls)} {conf:.2f}" for cls, conf in zip(results[0].boxes.cls.cpu(), results[0].boxes.conf.cpu())]
        # draw
        out_img = draw_boxes(img, boxes, labels)
        # save
        fname = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.output, fname), out_img)

    print(f"Saved annotated images to {args.output}")

if __name__ == '__main__':
    main()
