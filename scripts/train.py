import argparse
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for content-box detection')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--weights', default='yolov8m.pt',
                        help='Pretrained weights (Ultralytics model)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--imgsz', type=int,
                        help='Image size (overrides config)')
    parser.add_argument('--batch', type=int,
                        help='Batch size (overrides config)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args if provided
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.imgsz:
        cfg['imgsz'] = args.imgsz
    if args.batch:
        cfg['batch'] = args.batch

    # Initialize model
    model = YOLO(args.weights)

    # Train
    model.train(
        data=args.config,
        epochs=cfg.get('epochs', 50),
        imgsz=cfg.get('imgsz', 640),
        batch=cfg.get('batch', 16),
        device=0,
        project=cfg.get('project', 'models/checkpoints'),
        name=cfg.get('name', 'content_box_exp'),
        exist_ok=True
    )

if __name__ == '__main__':
    main()
