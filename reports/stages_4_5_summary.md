# Stages 4 and 5 — completed

- Full split sizes: train=17498, validation=3748, test=3754.
- Content grouping kept six exact duplicate photos in one split each; zero path or content-hash overlap between splits.
- All 25,000 photos decoded through the TensorFlow loader as normalized float32 RGB 200x200 images.
- Validation loading is deterministic; augmentation is training-only.
- Full requested CNN: 13 convolution layers, five max pools, Flatten, Dense4096, Dense4096, Dense2 softmax.
- Parameters: 107005762.
- Initial run: 2 epochs, 64 balanced training images, 32 balanced validation images, batch size 2.
- Plain SGD and an unzipped Keras checkpoint reduce memory use. A first zipped-save attempt was killed; the completed run used the revised saving method.
- Checkpoint: models/initial_cnn; 428119065 bytes; reload and probability checks passed.
- Final subset training accuracy: 0.4375. Final subset validation accuracy: 0.5.
- These small-subset metrics are a pipeline check, not useful final classifier performance or test accuracy.
- Test images used for fitting, model selection, or predictions: 0.
- Notebook: src/02_preprocessing_and_initial_training.ipynb (executed without cell errors).

Next: full-data training, validation-driven optimization, and final test-set evaluation.
