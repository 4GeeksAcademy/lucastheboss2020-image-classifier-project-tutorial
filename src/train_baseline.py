"""Train the assignment's VGG16-style CNN on an explicit initial subset."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse, gc, json, math, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from data_pipeline import ROOT, SEED, IMAGE_SIZE, CLASS_NAMES, make_dataset, make_augmentation, records_for_split


def build_model():
    model=tf.keras.Sequential(name='vgg16_style_dogs_cats_200')
    model.add(tf.keras.Input(shape=(*IMAGE_SIZE,3)))
    model.add(make_augmentation())
    for block,(filters,repetitions) in enumerate([(64,2),(128,2),(256,3),(512,3),(512,3)],1):
        for convolution in range(1,repetitions+1):
            model.add(tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu',name=f'block{block}_conv{convolution}'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,name=f'block{block}_pool'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(4096,activation='relu',name='fc1'))
    model.add(tf.keras.layers.Dense(4096,activation='relu',name='fc2'))
    model.add(tf.keras.layers.Dense(2,activation='softmax',name='class_probabilities'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'],jit_compile=False)
    return model


class Progress(tf.keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs=None):
        print(f'INITIAL TRAINING epoch {epoch+1}/{self.params["epochs"]}',flush=True)
    def on_train_batch_end(self,batch,logs=None):
        if batch%2==0:
            print(f'  batch {batch+1}: loss={logs["loss"]:.5f}, accuracy={logs["accuracy"]:.3f}',flush=True)


class DirectoryCheckpoint(tf.keras.callbacks.Callback):
    """Stream the best complete Keras checkpoint to disk without an in-memory ZIP."""
    def __init__(self,path):
        super().__init__();self.path=Path(path);self.best=float('inf')
    def on_epoch_end(self,epoch,logs=None):
        loss=float((logs or {}).get('val_loss',float('inf')))
        if loss<self.best:
            print(f'Saving unzipped checkpoint for epoch {epoch+1}: val_loss={loss:.5f}',flush=True)
            self.model.save(str(self.path),zipped=False)
            self.best=loss


def train(epochs=2,batch_size=2,train_limit=64,validation_limit=32,full_data=False):
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    assert (ROOT/'reports/pipeline_validation.json').exists(),'Validate the pipeline first.'
    if epochs < 1 or batch_size < 1:raise ValueError('Epochs and batch size must be positive.')
    if full_data:train_limit=validation_limit=None
    run_prefix='full_data' if full_data else 'initial'
    train_rows=records_for_split('train',train_limit);validation_rows=records_for_split('validation',validation_limit)
    assert not {r['sha256'] for r in train_rows}&{r['sha256'] for r in validation_rows}
    train_data=make_dataset('train',batch_size=batch_size,limit=train_limit)
    validation_data=make_dataset('validation',batch_size=batch_size,limit=validation_limit,shuffle=False)
    model=build_model()
    assert model.input_shape==(None,200,200,3) and model.output_shape==(None,2)
    assert sum(isinstance(l,tf.keras.layers.Conv2D) for l in model.layers)==13
    assert [l.units for l in model.layers if isinstance(l,tf.keras.layers.Dense)]==[4096,4096,2]
    text=[];model.summary(print_fn=lambda line,**kw:text.append(line))
    (ROOT/('reports/model_summary.txt' if not full_data else 'reports/full_data_model_summary.txt')).write_text(chr(10).join(text)+chr(10))
    model_dir=ROOT/'models';model_dir.mkdir(exist_ok=True)
    checkpoint=model_dir/('full_data_cnn' if full_data else 'initial_cnn')
    if checkpoint.exists():raise FileExistsError('Checkpoint already exists; choose a new run directory before overwriting.')
    run={'run_kind':'full_data' if full_data else 'initial_subset_pipeline_check','seed':SEED,'epochs_requested':epochs,'batch_size':batch_size,'training_images_used':len(train_rows),'validation_images_used':len(validation_rows),'test_images_used':0,'class_mapping':{'cat':0,'dog':1},'input_shape':[200,200,3],'output_units':2,'output_activation':'softmax','loss':'sparse_categorical_crossentropy','optimizer':'SGD(lr=0.001,momentum=0.0)','parameters':model.count_params(),'architecture_note':'Matches the course Conv2D/MaxPool/Flatten/Dense4096/Dense4096/Dense2 architecture, with a consistent 200x200 input and training-only augmentation. No pretrained weights.','checkpoint':str(checkpoint.relative_to(ROOT)),'interpretation':'Small initial run to verify the complete training/save/reload pipeline. These subset metrics are not final test accuracy or a completed optimization study.'}
    (ROOT/f'reports/{run_prefix}_training_config.json').write_text(json.dumps(run,indent=2)+chr(10))
    (ROOT/f'reports/{run_prefix}_training_subset.json').write_text(json.dumps({'train':[r['path'] for r in train_rows],'validation':[r['path'] for r in validation_rows]},indent=2)+chr(10))
    print(json.dumps(run,indent=2),flush=True)
    started=time.monotonic()
    history=model.fit(train_data,validation_data=validation_data,epochs=epochs,verbose=2,shuffle=False,callbacks=[Progress(),tf.keras.callbacks.TerminateOnNaN(),tf.keras.callbacks.CSVLogger(str(ROOT/f'reports/{run_prefix}_training_history.csv')),DirectoryCheckpoint(checkpoint)])
    values={key:[float(x) for x in value] for key,value in history.history.items()}
    assert all(math.isfinite(x) for value in values.values() for x in value),'Nonfinite training metrics'
    (ROOT/f'reports/{run_prefix}_training_history.json').write_text(json.dumps(values,indent=2)+chr(10))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epoch_numbers=list(range(1,len(values['loss'])+1))
    for ax,key,title in zip(axes,['loss','accuracy'],['Loss','Accuracy']):
        ax.plot(epoch_numbers,values[key],marker='o',label='train subset')
        ax.plot(epoch_numbers,values['val_'+key],marker='o',label='validation subset')
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_xticks(epoch_numbers);ax.legend();ax.grid(alpha=.25)
    fig.suptitle(f'Initial pipeline check only: {len(train_rows)} train / {len(validation_rows)} validation images')
    fig.tight_layout();fig.savefig(ROOT/f'reports/figures/{run_prefix}_training_history.png',dpi=130);plt.close(fig)
    sample=next(iter(validation_data))[0]
    del model,history
    tf.keras.backend.clear_session();gc.collect()
    reloaded=tf.keras.models.load_model(checkpoint,compile=False)
    probabilities=reloaded(sample,training=False).numpy()
    assert probabilities.shape==(int(sample.shape[0]),2) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1),1.0,atol=1e-5)
    result={**run,'epochs_completed':len(values['loss']),'elapsed_seconds':round(time.monotonic()-started,2),'history':values,'best_validation_epoch':int(np.argmin(values['val_loss']))+1,'checkpoint_bytes':sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),'checkpoint_reload_passed':True,'test_set_evaluated':False}
    (ROOT/f'reports/{run_prefix}_training_results.json').write_text(json.dumps(result,indent=2)+chr(10))
    print('INITIAL_TRAINING_COMPLETE: checkpoint reload and probability checks passed; test set not evaluated.',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--train-limit',type=int,default=64)
    parser.add_argument('--validation-limit',type=int,default=32)
    parser.add_argument('--full-data',action='store_true')
    a=parser.parse_args()
    train(a.epochs,a.batch_size,a.train_limit,a.validation_limit,a.full_data)
