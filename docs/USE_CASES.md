# Use cases and format/model status

| Use case | Format | Model | Status |
|---|---|---|---|
| Object detection | YOLO txt + data.yaml | RT-DETR r18vd (COCO pretrained) | verified (GPU run, apple-orange val 0.873 / test 0.825) |
| Object detection | COCO json (incl. Roboflow export) | RT-DETR | tested (fixtures); not user-run |
| Image classification | image folder | CNN (tiny PyTorch) | tested |
| Tabular | csv | RandomForest | tested |
| Segmentation | — | — | planned (no trainer) |
| RT-DETR-L, ResNet-50 | — | — | blocked (`TODO(user)` URL/sha256/licence) |
