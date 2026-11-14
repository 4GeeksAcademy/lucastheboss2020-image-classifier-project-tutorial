"""Full-data transfer-learning comparison with validation-only model selection."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,csv,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,log_loss,roc_auc_score
from data_pipeline import ROOT,SEED,make_dataset,make_augmentation,records_for_split


def backbone_for(name):
    constructor=tf.keras.applications.MobileNetV2 if name=='mobilenet_v2' else tf.keras.applications.VGG16
    model=constructor(weights='imagenet',include_top=False,input_shape=(200,200,3),pooling='avg')
    model.trainable=False
    return model


def preprocess(images,name):
    return images*2.0-1.0 if name=='mobilenet_v2' else tf.keras.applications.vgg16.preprocess_input(images*255.0)


def cache_features(base,name,split,augment_fraction=0.0):
    rows=records_for_split(split)
    cache_dir=ROOT/'data/processed'/('features_'+name);cache_dir.mkdir(parents=True,exist_ok=True)
    manifest_hash=hashlib.sha256((ROOT/'data/processed/split_manifest.csv').read_bytes()).hexdigest()
    extra_indices=[i for i in range(len(rows)) if split=='train' and augment_fraction>0 and i%round(1/augment_fraction)==0]
    n=len(rows);d=int(base.output_shape[-1]);total=n+len(extra_indices)
    meta_path=cache_dir/(split+'_metadata.json')
    signature={'backbone':name,'weights':'imagenet','split':split,'original_images':n,'augmented_views':len(extra_indices),'rows':total,'dimensions':d,'manifest_sha256':manifest_hash,'input_size':[200,200],'seed':SEED}
    xp=cache_dir/(split+'_features.npy');yp=cache_dir/(split+'_labels.npy')
    if meta_path.exists() and json.loads(meta_path.read_text())==signature and xp.exists() and yp.exists():
        print('Using verified cached features:',split,flush=True)
        return np.load(xp,mmap_mode='r'),np.load(yp),signature
    features=np.lib.format.open_memmap(xp,mode='w+',dtype='float32',shape=(total,d))
    labels=np.lib.format.open_memmap(yp,mode='w+',dtype='int32',shape=(total,))
    augmentation=make_augmentation() if extra_indices else None
    @tf.function(input_signature=[tf.TensorSpec([None,200,200,3],tf.float32)])
    def forward(images):return base(preprocess(images,name),training=False)
    original_offset=0;aug_offset=n;last_print=0;started=time.monotonic()
    print(f'FEATURE EXTRACTION {split}: {n} original images + {len(extra_indices)} training-only augmented views',flush=True)
    for images,batch_labels in make_dataset(split,batch_size=16,shuffle=False):
        size=int(images.shape[0]);out=forward(images).numpy()
        features[original_offset:original_offset+size]=out
        labels[original_offset:original_offset+size]=batch_labels.numpy()
        if augmentation is not None:
            positions=[j for j in range(size) if (original_offset+j)%round(1/augment_fraction)==0]
            if positions:
                augmented=augmentation(tf.gather(images,positions),training=True)
                count=len(positions)
                features[aug_offset:aug_offset+count]=forward(augmented).numpy()
                labels[aug_offset:aug_offset+count]=tf.gather(batch_labels,positions).numpy();aug_offset+=count
        original_offset+=size
        if original_offset-last_print>=512 or original_offset==n:
            print(f'{split}: {original_offset}/{n} images ({time.monotonic()-started:.0f}s)',flush=True);last_print=original_offset
    assert original_offset==n and aug_offset==total
    expected=np.array([int(r['label']) for r in rows],dtype='int32')
    np.testing.assert_array_equal(labels[:n],expected)
    assert np.isfinite(features).all()
    features.flush();labels.flush()
    meta_path.write_text(json.dumps(signature,indent=2)+chr(10))
    print('FEATURE CACHE COMPLETE',split,flush=True)
    return features,np.asarray(labels),signature


def feature_dataset(x,y,training=False):
    ds=tf.data.Dataset.from_tensor_slices((np.asarray(x),np.asarray(y)))
    if training:ds=ds.shuffle(len(y),seed=SEED,reshuffle_each_iteration=True)
    options=tf.data.Options();options.threading.private_threadpool_size=2;options.threading.max_intra_op_parallelism=1
    return ds.with_options(options).batch(128).prefetch(1)


def train_heads(train_x,train_y,val_x,val_y):
    from sklearn.metrics import log_loss,accuracy_score
    specifications=[{'name':'linear_head','units':0,'dropout':0.0,'lr':0.001},{'name':'regularized_head','units':128,'dropout':0.35,'lr':0.0003}]
    trials=[];histories={};folder=ROOT/'models';folder.mkdir(exist_ok=True)
    for index,spec in enumerate(specifications):
        tf.keras.utils.set_random_seed(SEED+index)
        head=tf.keras.Sequential(name=spec['name']);head.add(tf.keras.Input(shape=(int(train_x.shape[1]),)))
        if spec['units']:
            head.add(tf.keras.layers.Dense(spec['units'],activation='relu'))
            head.add(tf.keras.layers.Dropout(spec['dropout'],seed=SEED))
        head.add(tf.keras.layers.Dense(2,activation='softmax'))
        head.compile(optimizer=tf.keras.optimizers.Adam(spec['lr']),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        checkpoint=folder/(spec['name']+'.keras')
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True,verbose=1),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,min_delta=0.0001,restore_best_weights=True,verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=0.000001,verbose=1),tf.keras.callbacks.CSVLogger(str(ROOT/'reports'/(spec['name']+'_history.csv')))]
        print('TRAINING FULL-DATA HEAD',json.dumps(spec),flush=True)
        started=time.monotonic()
        history=head.fit(feature_dataset(train_x,train_y,True),validation_data=feature_dataset(val_x,val_y),epochs=25,shuffle=False,callbacks=callbacks,verbose=2)
        best=tf.keras.models.load_model(checkpoint,compile=False)
        probabilities=best.predict(feature_dataset(val_x,val_y),verbose=0)
        trial={**spec,'epochs_completed':len(history.history['loss']),'best_epoch':int(np.argmin(history.history['val_loss']))+1,'validation_log_loss':float(log_loss(val_y,probabilities,labels=[0,1])),'validation_accuracy':float(accuracy_score(val_y,probabilities.argmax(axis=1))),'parameters':best.count_params(),'elapsed_seconds':round(time.monotonic()-started,2),'checkpoint':str(checkpoint.relative_to(ROOT))}
        trials.append(trial);histories[spec['name']]={k:[float(v) for v in values] for k,values in history.history.items()}
        (ROOT/'reports'/(spec['name']+'_history.json')).write_text(json.dumps(histories[spec['name']],indent=2)+chr(10))
        print('TRIAL COMPLETE',json.dumps(trial),flush=True)
        del head,best,history;gc.collect()
    selected=min(trials,key=lambda t:t['validation_log_loss'])
    report={'selection_rule':'Lowest full-validation-set log loss; test labels and metrics not used for selection.','trials':trials,'selected':selected,'test_evaluated_during_selection':False}
    (ROOT/'reports/optimization_results.json').write_text(json.dumps(report,indent=2)+chr(10))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,h in histories.items():
        epochs=range(1,len(h['loss'])+1)
        axes[0].plot(epochs,h['loss'],label=name+' train');axes[0].plot(epochs,h['val_loss'],linestyle='--',label=name+' validation')
        axes[1].plot(epochs,h['accuracy'],label=name+' train');axes[1].plot(epochs,h['val_accuracy'],linestyle='--',label=name+' validation')
    for ax,title in zip(axes,['Cross-entropy loss','Accuracy']):ax.set_title(title);ax.set_xlabel('Epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Full training split — validation-driven head comparison');fig.tight_layout();fig.savefig(ROOT/'reports/figures/optimization_history.png',dpi=140);plt.close(fig)
    with (ROOT/'reports/optimization_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(trials[0]));writer.writeheader();writer.writerows(trials)
    return selected


def export_model(base,name,selected,val_features):
    head=tf.keras.models.load_model(ROOT/selected['checkpoint'],compile=False)
    inputs=tf.keras.Input(shape=(200,200,3),name='normalized_rgb_image')
    if name=='mobilenet_v2':
        prepared=tf.keras.layers.Rescaling(2.0,offset=-1.0,name='mobilenet_preprocessing')(inputs)
    else:
        bgr=tf.keras.ops.flip(inputs,axis=-1)
        prepared=tf.keras.layers.Rescaling(255.0,offset=[-103.939,-116.779,-123.68],name='vgg_preprocessing')(bgr)
    outputs=head(base(prepared,training=False))
    model=tf.keras.Model(inputs,outputs,name=name+'_dogs_cats_final')
    target=ROOT/'models/final_classifier.keras';model.save(target)
    loaded=tf.keras.models.load_model(target,compile=False)
    batch=next(iter(make_dataset('validation',batch_size=4,shuffle=False)))[0]
    actual=loaded(batch,training=False).numpy();expected=head(np.asarray(val_features[:4]),training=False).numpy()
    np.testing.assert_allclose(actual,expected,rtol=0.0005,atol=0.00005)
    np.testing.assert_allclose(actual.sum(axis=1),1.0,atol=0.00001)
    metadata={'backbone':name,'pretraining':'ImageNet','frozen_backbone':True,'selected_head':selected['name'],'model_path':'models/final_classifier.keras','input_shape':[200,200,3],'input_range':[0,1],'class_names':['cat','dog'],'parameters':loaded.count_params(),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'reload_prediction_equivalence_passed':True}
    (ROOT/'reports/final_model_metadata.json').write_text(json.dumps(metadata,indent=2)+chr(10))
    print('FINAL MODEL EXPORTED AND RELOADED',json.dumps(metadata),flush=True)
    return head,metadata


def evaluate_reserved_test(base,name,head,selected):
    assert (ROOT/'reports/optimization_results.json').exists()
    assert not json.loads((ROOT/'reports/optimization_results.json').read_text())['test_evaluated_during_selection']
    test_x,test_y,test_meta=cache_features(base,name,'test',0.0)
    probabilities=head.predict(feature_dataset(test_x,test_y),verbose=0)
    predictions=probabilities.argmax(axis=1)
    matrix=confusion_matrix(test_y,predictions,labels=[0,1])
    metrics={'test_images':len(test_y),'accuracy':float(accuracy_score(test_y,predictions)),'log_loss':float(log_loss(test_y,probabilities,labels=[0,1])),'roc_auc':float(roc_auc_score(test_y,probabilities[:,1])),'confusion_matrix':matrix.tolist(),'classification_report':classification_report(test_y,predictions,labels=[0,1],target_names=['cat','dog'],output_dict=True,zero_division=0),'selected_head':selected['name'],'selection_locked_before_test':True,'threshold_policy':'Argmax of two softmax class probabilities; no test-set threshold tuning.'}
    (ROOT/'reports/test_metrics.json').write_text(json.dumps(metrics,indent=2)+chr(10))
    rows=records_for_split('test')
    with (ROOT/'reports/test_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['path','true_label','predicted_label','cat_probability','dog_probability','correct'])
        for row,y,p,pr in zip(rows,test_y,predictions,probabilities):writer.writerow([row['path'],int(y),int(p),float(pr[0]),float(pr[1]),bool(y==p)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax)
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center',fontsize=15,color='white' if value>matrix.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['cat','dog'],yticklabels=['cat','dog'],xlabel='Predicted label',ylabel='True label',title=f'Reserved test set: accuracy {metrics["accuracy"]:.2%}')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/confusion_matrix.png',dpi=150);plt.close(fig)
    correct=np.flatnonzero(test_y==predictions);wrong=np.flatnonzero(test_y!=predictions)
    display_indices=list(correct[test_y[correct]==0][:4])+list(correct[test_y[correct]==1][:4])+list(wrong[test_y[wrong]==0][:2])+list(wrong[test_y[wrong]==1][:2])
    fig,axes=plt.subplots(3,4,figsize=(12,10))
    for ax in axes.flat:ax.axis('off')
    for ax,index in zip(axes.flat,display_indices):
        with Image.open(ROOT/rows[index]['path']) as image:ax.imshow(image.convert('RGB'))
        ok=test_y[index]==predictions[index]
        ax.set_title(f'True: {["cat","dog"][test_y[index]]}; predicted: {["cat","dog"][predictions[index]]}'+chr(10)+f'confidence {probabilities[index].max():.1%}',color='green' if ok else 'red',fontsize=9)
    fig.suptitle('Reserved test predictions: correct cases and mistakes');fig.tight_layout();fig.savefig(ROOT/'reports/figures/test_prediction_examples.png',dpi=130);plt.close(fig)
    print('RESERVED TEST EVALUATION COMPLETE',json.dumps({k:v for k,v in metrics.items() if k!='classification_report'}),flush=True)
    return metrics


def main(name='mobilenet_v2',reproduce=False):
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED);tf.config.experimental.enable_op_determinism()
    if (ROOT/'reports/test_metrics.json').exists() and not reproduce:
        raise FileExistsError('Existing final evaluation protected. Use --reproduce only to rerun the fixed pipeline.')
    if reproduce and (ROOT/'reports/final_model_metadata.json').exists():
        previous=json.loads((ROOT/'reports/final_model_metadata.json').read_text())
        assert previous['backbone']==name, 'Reproduction must use the original backbone; do not change architecture using a seen test split.'
    started=time.monotonic();base=backbone_for(name)
    train_x,train_y,train_meta=cache_features(base,name,'train',0.25)
    val_x,val_y,val_meta=cache_features(base,name,'validation',0.0)
    feature_report={'backbone':name,'train':train_meta,'validation':val_meta,'augmentation_policy':'One seeded augmented view for every fourth training manifest record; no validation/test augmentation. Both head trials use exactly the same training views.','test_features_accessed_before_selection':False}
    (ROOT/'reports/feature_extraction.json').write_text(json.dumps(feature_report,indent=2)+chr(10))
    selected=train_heads(train_x,train_y,val_x,val_y)
    head,metadata=export_model(base,name,selected,val_x)
    metrics=evaluate_reserved_test(base,name,head,selected)
    (ROOT/'reports/final_run_summary.json').write_text(json.dumps({'backbone':name,'elapsed_seconds':round(time.monotonic()-started,2),'selected_head':selected['name'],'test_accuracy':metrics['accuracy'],'training_unique_images':train_meta['original_images'],'training_augmented_views':train_meta['augmented_views'],'validation_images':val_meta['original_images'],'test_images':metrics['test_images'],'model_reload_passed':True,'model_sha256':metadata['sha256']},indent=2)+chr(10))
    print('STAGES_6_AND_7_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--backbone',choices=['mobilenet_v2','vgg16'],default='mobilenet_v2')
    parser.add_argument('--reproduce',action='store_true',help='Rerun the same predefined pipeline when results already exist; not for test-set tuning.')
    args=parser.parse_args();main(args.backbone,args.reproduce)
