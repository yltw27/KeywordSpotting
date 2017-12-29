#!/usr/bin/python3
import tensorflow as tf
import numpy as np
from array import array
from python_speech_features import mfcc
from ctypes import *
from contextlib import contextmanager
import pyaudio
import os
import pygame
import wave
from struct import pack
from datetime import datetime
import sys
import scipy.io.wavfile as wav
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if len(sys.argv) == 1:
    checkpoint_path = './ckpt/2017-11-08/'
    target_ckpt_file = 'kws-20171108-1605-300'

else:
    checkpoint_path = sys.argv[1]
    target_ckpt_file = sys.argv[2]


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


def inference(data):
    test_x = graph.get_tensor_by_name("input_x:0")
    test_y_true = graph.get_tensor_by_name("input_y:0")    
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    b = np.zeros((1, 2))
    feed_dict = {test_x: data, test_y_true: b, keep_prob: 1.0}
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    result = sess.run(op_to_restore, feed_dict)   
    return result


def run():
    os.system('arecord -d 3 -r 16000 -f S16_LE test.wav')
    fs, audio = wav.read('test.wav')

    feature = mfcc(np.asarray(audio), samplerate=16000).reshape(1, -1)

    start_point = datetime.utcnow()
    inf = inference(feature)
    end_point = datetime.utcnow()

    if inf == 1:
        print('\n{} {}'.format(inf, end_point - start_point))
        print('How can I help you?')
        os.system('aplay sounds/spell.wav\n')

    else:
        print(inf)


if __name__ == '__main__':
    start = datetime.now()
    sess = tf.Session()
#    with tf.device("/cpu:0"):
#    print(checkpoint_path + target_ckpt_file + '.meta')
    saver = tf.train.import_meta_graph(checkpoint_path + target_ckpt_file + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    graph = tf.get_default_graph()
    
    print('Warm up: {}'.format(datetime.now() - start))
    while True: 
        run()

