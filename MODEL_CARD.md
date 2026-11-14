# Model card — Dogs vs Cats

## Intended use

Educational binary image classification of cat and dog photos. The prediction command resizes and normalizes an image automatically. The model is not an object detector and should not be used to establish that an arbitrary image contains an animal.

## Model and training provenance

- Backbone: ImageNet-pretrained MobileNetV2, frozen.
- Head chosen using validation loss: regularized_head.
- Full local training set: 17,498 photos plus 4,375 training-only augmented views.
- Validation: 3,748 photos; reserved test: 3,754 photos.
- Original course VGG16-style CNN implemented separately and checked on a small initial subset.
- Two candidate heads; ModelCheckpoint and EarlyStopping monitored validation loss.
- Test metrics were not used for architecture, checkpoint, or threshold selection.

## Input and output

Input: float32 RGB arrays shaped (batch,200,200,3), normalized to [0,1]. Exported preprocessing maps these to the pretrained backbone’s expected range. Output: two softmax probabilities, ordered cat then dog. Prediction uses argmax.

## Evaluation

Reserved-test accuracy: **98.48%**. ROC AUC: **0.99881**. Detailed precision, recall, F1, confusion matrix and per-image predictions are in `reports/`.

## Limitations

The dataset is balanced and consists of pet photos; results need not transfer to different cameras, backgrounds, drawings, other animals or out-of-distribution inputs. The model always returns cat or dog. Confidence can be wrong and has not been calibrated. Frozen ImageNet pretraining provides most visual features; this is transfer learning, not full end-to-end training from scratch. Only two head configurations were compared.

## Reproducibility

Seed 42, pinned dependencies, content-hash split groups and source scripts are provided. Exact floating-point results may vary by hardware. The final model checksum is in `reports/final_model_metadata.json`.
