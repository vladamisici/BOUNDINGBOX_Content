# Content-Box-Detector

## 1. Structura proiect
```
content-box-detector/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                # PubLayNet COCO + partituri
│   ├── images/             # train/ val/ pentru YOLO
│   └── labels/             # train/ val/ `.txt` YOLO
├── scripts/
│   ├── download_publaynet.py
│   ├── prepare_publaynet.py
│   ├── convert_to_yolo.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
├── models/
│   ├── checkpoints/        # `best.pt`, `last.pt`
│   └── exports/            # ONNX, TensorRT
└── .gitignore
```

![Structura proiect](docs/structure.png)

---

## 2. Instalare dependinte
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# sau: source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## 3. Download PubLayNet
```bash
python scripts/download_publaynet.py \
  --parts full labels \
  --output data/raw/publaynet
```

![Download PubLayNet](docs/download.png)

---

## 4. Pregatire anotari
```bash
python scripts/prepare_publaynet.py \
  --coco data/raw/publaynet/labels/train.json \
  --classes text title list figure table \
  --out data/annotations_publaynet_train.json
```

![Agregare bounding-box](docs/prepare.png)

---

## 5. Convertire in format YOLO
```bash
python scripts/convert_to_yolo.py \
  --train-json data/annotations_publaynet_train.json \
  --val-json   data/annotations_publaynet_val.json \
  --image-root data/raw/publaynet/full \
  --out-images data/images \
  --out-labels data/labels
```

![YOLO labels](docs/convert.png)

---

## 6. Antrenare model
```bash
python scripts/train.py --config config.yaml --device 0
```

![Training](docs/train.png)

---

## 7. Inferenta
```bash
python scripts/infer.py \
  --weights models/checkpoints/content_box_exp/weights/best.pt \
  --source data/images/val \
  --output runs/inference \
  --conf 0.25
```

![Inferenta](docs/infer.png)

---

## 8. Export model
```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx')
