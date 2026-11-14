"""Memory-bounded tf.data image loading and training-only augmentation."""
from pathlib import Path
import csv, random
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (200, 200)
CLASS_NAMES = ('cat', 'dog')
SEED = 42


def records_for_split(split, limit=None):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    with (ROOT / 'data/processed/split_manifest.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['split'] == split]
    if not rows:
        raise ValueError('No records; run src/build_splits.py first.')
    if limit is not None:
        if limit < 2 or limit > len(rows):
            raise ValueError('Subset size must be between two and the split size.')
        selected = []
        for label in range(2):
            group = [r for r in rows if int(r['label']) == label]
            random.Random(SEED + label + (0 if split == 'train' else 100)).shuffle(group)
            selected.extend(group[:limit//2 + (1 if label == 0 and limit % 2 else 0)])
        rows = selected
    return rows


def preprocess_image(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, method='bilinear', antialias=True)
    image = tf.clip_by_value(tf.cast(image, tf.float32) / 255.0, 0.0, 1.0)
    return image, tf.cast(label, tf.int32)


def make_dataset(split, batch_size=4, limit=None, shuffle=None):
    rows = records_for_split(split, limit)
    paths = [str(ROOT / r['path']) for r in rows]
    labels = [int(r['label']) for r in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle is None:
        shuffle = split == 'train'
    if shuffle:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    dataset = dataset.map(preprocess_image, num_parallel_calls=2, deterministic=True)
    return dataset.batch(batch_size).prefetch(1)


def make_augmentation():
    # Keras augmentation layers leave validation/test/inference inputs unchanged.
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal', seed=SEED),
        tf.keras.layers.RandomRotation(0.05, fill_mode='reflect', seed=SEED+1),
        tf.keras.layers.RandomZoom(0.10, fill_mode='reflect', seed=SEED+2),
    ], name='training_only_augmentation')
