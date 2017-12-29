import tensorflow as tf
import pandas as pd
import numpy as np
import math
import scipy.io.wavfile as wav
from datetime import datetime
from python_speech_features import mfcc


def convert_single_example(data, label0, label1):
    fs, audio = wav.read(data)
    feature = mfcc(audio, samplerate=fs, numcep=13)       
    label = np.column_stack((label0, label1)).astype(float)
    return feature, label


def _floats_feature(arr):
    arr = np.reshape(arr, (-1))
    return tf.train.Feature(float_list=tf.train.FloatList(value=arr))
    
    
def save_tfrecord(catalog, tfrecord_filename):
    # TODO: shuffle with random seed
    df = pd.read_csv(catalog)
    writer = tf.python_io.TFRecordWriter('data/'+tfrecord_filename)
    for i in range(df.shape[0]):
        if i % 100 == 0 and i != 0:
            print('transfered {}'.format(i))
        x, y = convert_single_example(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2])
        example = tf.train.Example(features=tf.train.Features(feature={
                                   'source': _floats_feature(x),
                                   'target': _floats_feature(y)}))
        writer.write(example.SerializeToString())
    writer.close()
    
    
if __name__ == '__main__':
    start = datetime.now()
    
    save_tfrecord("data/train.csv", "train.tfrecords")
    save_tfrecord("data/validation.csv", "val.tfrecords")
    
    print('Data Processing: {}'.format(str(datetime.now()-start)))

    
