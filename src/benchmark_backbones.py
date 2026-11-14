import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import gc,json,time
from pathlib import Path
import tensorflow as tf
from data_pipeline import ROOT,make_dataset

def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    images=next(iter(make_dataset('train',batch_size=8,limit=32)))[0]
    results={}
    for name,constructor,preprocess in [('vgg16',tf.keras.applications.VGG16,lambda x:tf.keras.applications.vgg16.preprocess_input(x*255.0)),('mobilenet_v2',tf.keras.applications.MobileNetV2,lambda x:x*2.0-1.0)]:
        print('BENCHMARK',name,flush=True)
        base=constructor(include_top=False,weights='imagenet',input_shape=(200,200,3),pooling='avg')
        base.trainable=False
        values=preprocess(images)
        base(values,training=False).numpy()
        started=time.monotonic()
        for _ in range(2):features=base(values,training=False).numpy()
        seconds=time.monotonic()-started
        per_image=seconds/16
        results[name]={'seconds_per_image':per_image,'estimated_train_validation_feature_minutes':per_image*(17498+3748)/60,'parameters':base.count_params(),'feature_dimension':int(features.shape[-1])}
        print(name,json.dumps(results[name]),flush=True)
        del base
        tf.keras.backend.clear_session();gc.collect()
    (ROOT/'reports/compute_benchmark.json').write_text(json.dumps(results,indent=2)+chr(10))
    print('BENCHMARK_COMPLETE',flush=True)

if __name__=='__main__':main()
