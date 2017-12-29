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
    checkpoint_path = './ckpt/2017-10-25/'  # wrong~10%
    target_ckpt_file = 'kws_v1-20171025-1757-200' 
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
 #   pygame.mixer.init()

#    with noalsaerr():
#        stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
#    stream_save = []
#    r_temp = array('h')  # save wav.

#    stream_record = stream.read(48000)
#    stream_save.extend(stream_record)
#    r_temp.extend(array('h', stream_record))

#    stream.stop_stream()
#    stream.close()

    os.system('sudo arecord -d 3 -r 16000 -f S16_LE ./test.wav')
    #os.system('sudo arecord -D plughw:1,0 -d 3 -r 16000 -f S16_LE ./test.wav')
    # os.system("aplay ./test.wav")
    fs, audio = wav.read('./test.wav')

    feature = mfcc(np.asarray(audio), samplerate=16000).reshape(1, -1)
#    time_step = feature.shape[0]
#    feature = feature.reshape(1, -1)
#    # normalization
#    feature_mean = np.mean(feature)
#    feature_std = np.std(feature)
#    feature = [(x - feature_mean) / feature_std for x in feature]
#    feature = np.array(feature)

    # sleep(0.1)
    start_point = datetime.utcnow()
    inf = inference(feature)
    end_point = datetime.utcnow()

    if inf == 1:
        print('\n{} {}'.format(inf, end_point - start_point))
        print('How can I help you?')
        os.system('aplay ./sounds/spell.wav\n')
        #pygame.mixer.music.load("./sounds/spell.wav")
        #pygame.mixer.music.play(0)
        #while pygame.mixer.music.get_busy():
        #    pygame.event.poll()

    else:
        print(inf)
    
    # sleep(1)


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
    # for i in range(20): 
        run()
#        sleep(2)

