"""Run the saved classifier on one local cat/dog image."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import argparse,json
import numpy as np
import tensorflow as tf
try:
    from .data_pipeline import ROOT,preprocess_image
except ImportError:
    from data_pipeline import ROOT,preprocess_image


def predict_image(image_path,model_path=None):
    path=Path(image_path).expanduser().resolve()
    target=Path(model_path) if model_path else ROOT/'models/final_classifier.keras'
    if not path.is_file():raise FileNotFoundError('Image file does not exist: '+str(path))
    if not target.is_file():raise FileNotFoundError('Saved model is missing. Download the repository model or complete training first.')
    image,_=preprocess_image(tf.constant(str(path)),tf.constant(0))
    model=tf.keras.models.load_model(target,compile=False)
    probabilities=np.asarray(model(image[None,...],training=False))[0]
    if probabilities.shape!=(2,) or not np.isfinite(probabilities).all():raise ValueError('Model did not return two finite class probabilities.')
    np.testing.assert_allclose(probabilities.sum(),1.0,atol=1e-5)
    label=int(probabilities.argmax())
    return {'predicted_class':['cat','dog'][label],'confidence':float(probabilities[label]),'probabilities':{'cat':float(probabilities[0]),'dog':float(probabilities[1])}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,help='Path to an image file.')
    parser.add_argument('--model',default=str(ROOT/'models/final_classifier.keras'))
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    try:result=predict_image(args.image,args.model)
    except (FileNotFoundError,ValueError,tf.errors.OpError) as error:parser.error(str(error))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
