# KeywordSpotting

Keyword Spotting using Convolutional Neural Network

Reference: 
Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting
https://arxiv.org/abs/1703.05390

stream.py
 - record .wav files in 3 seconds 
 - save in "data/name"
 - the format of filename: 
    20171231_NAME_0000_0.wav (the last number = 0, without keyword)
    20171231_NAME_0012_1.wav (the last number = 1, with keyword)

create_catalog.py
 - read all .wav in data/ and create 2 catalogs of train.csv and validaiotn.csv
 
mfcc_tfrecord.py
 - convert each .wav file in the certain catalog and save as .tfrecords
 - enhance training efficiency

train_nontfrecord.py
 - train model with raw .wav files
 - save training result in graph/
 - save model in ckpt/
 
train_tfrecord.py
 - train model with .tfrecords
 - save training result in graph/
 - save model in ckpt/

inference.py
 - edit line 21-22 to your own checkpoint(.meta) path 
 - continueously create(record) a test.wav(3 seconds) and do inference
