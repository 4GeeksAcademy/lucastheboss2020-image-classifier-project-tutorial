"""Validate all splits through TensorFlow before starting model training."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from pathlib import Path
from collections import Counter
import csv, hashlib, json
import numpy as np
import tensorflow as tf
from data_pipeline import ROOT, make_dataset, make_augmentation


def validate_pipeline():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    summary=json.loads((ROOT/'reports/split_summary.json').read_text())
    manifest=ROOT/summary['manifest']
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==summary['manifest_sha256']
    with manifest.open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert all((ROOT/r['path']).is_file() for r in rows)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        left=[r for r in rows if r['split']==a];right=[r for r in rows if r['split']==b]
        assert not {r['path'] for r in left}&{r['path'] for r in right}
        assert not {r['sha256'] for r in left}&{r['sha256'] for r in right}
    results={}
    first_validation=None
    for split in ('train','validation','test'):
        count=0;labels_seen=Counter();minimum=1.0;maximum=0.0
        for images,labels in make_dataset(split,batch_size=32,shuffle=False):
            assert tuple(images.shape[1:])==(200,200,3)
            assert images.dtype==tf.float32 and labels.dtype==tf.int32
            tf.debugging.assert_all_finite(images,'Nonfinite pixel')
            lo=float(tf.reduce_min(images));hi=float(tf.reduce_max(images))
            assert 0.0<=lo<=hi<=1.0
            minimum=min(minimum,lo);maximum=max(maximum,hi)
            count+=int(images.shape[0]);labels_seen.update(labels.numpy().tolist())
            if split=='validation' and first_validation is None:
                first_validation=images[:4].numpy()
        assert count==summary['splits'][split]['count']
        assert labels_seen[0]==summary['splits'][split]['class_counts']['cat']
        assert labels_seen[1]==summary['splits'][split]['class_counts']['dog']
        results[split]={'decoded_images':count,'pixel_min':minimum,'pixel_max':maximum,'shape':[200,200,3],'class_counts':dict(labels_seen)}
        print(f'TENSORFLOW DECODE PASS: {split}: {count} images, RGB 200x200, pixels in [0,1]',flush=True)
    augmentation=make_augmentation()
    unchanged=augmentation(first_validation,training=False).numpy()
    np.testing.assert_array_equal(unchanged,first_validation)
    changed=augmentation(first_validation,training=True).numpy()
    assert changed.shape==first_validation.shape and np.isfinite(changed).all()
    assert not np.array_equal(changed,first_validation)
    second=next(iter(make_dataset('validation',batch_size=32,shuffle=False)))[0][:4].numpy()
    np.testing.assert_array_equal(first_validation,second)
    report={'decoded_images':sum(r['decoded_images'] for r in results.values()),'splits':results,'path_overlap':0,'content_hash_overlap':0,'validation_deterministic':True,'augmentation_changes_training_input':True,'augmentation_disabled_for_inference':True,'model_evaluated_on_test':False}
    (ROOT/'reports/pipeline_validation.json').write_text(json.dumps(report,indent=2)+chr(10))
    print('DATA PIPELINE VALIDATED; no test-set model evaluation performed.',flush=True)
    return report


if __name__=='__main__':validate_pipeline()
