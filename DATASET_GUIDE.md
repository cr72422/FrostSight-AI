# Dataset and Evaluation Guide

## First version: no custom dataset needed

The application uses a pretrained YOLO model to let you build and test the complete image-inference workflow. Use several images you are permitted to use. Record the image, confidence threshold, predictions, inference time, and observed mistakes.

A general-purpose pretrained model may label a package as another object class or miss it. Do not interpret detections as validated package counts.

## Optional package-specific training/evaluation

1. Browse a dataset catalog such as Roboflow Universe's package-detection use cases:
   https://universe.roboflow.com/use-cases/package-detection
2. Open the specific dataset page and inspect class names, bounding-box quality, split sizes, and license.
3. Download annotations in a YOLO-compatible format only if the license allows your intended use.
4. Organize files according to the dataset export's documented structure.
5. Create `data.yaml` using the actual local paths and class names. Do not copy the example configuration without checking it.
6. Ensure train, validation, and test images are separate and there is no data leakage.
7. Use a detector configured for the dataset's classes. A general pretrained model may need fine-tuning before package-specific evaluation.
8. Run `python evaluate.py` after `data.yaml` is correctly configured and the model is compatible.
9. Record the metrics and limitations in the README.

## Example `data.yaml` (edit to match your dataset)

```yaml
path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: package
```

This is illustrative only. Multi-class datasets need all class names and correct annotation IDs. Paths depend on the actual export.

## Metrics

- Precision: proportion of predicted detections that match labeled objects.
- Recall: proportion of labeled objects that the model detects.
- mAP50: mean average precision at an intersection-over-union threshold of 0.50.
- mAP50-95: averaged across multiple IoU thresholds.
- Inference time: elapsed prediction time; report hardware and whether model startup is included.

Never invent metric values. The app's confidence scores are not accuracy metrics.
