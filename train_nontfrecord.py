import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import pandas as pd
import numpy as np

start = datetime.now()

# basic setting
feature, time_step = 13, 299
learning_rate = 0.0001
keep_prob = 1.0
#default_stddev = 0.046875
n_hidden_1 = 8
n_hidden_2 = 8
n_hidden_3 = 16
n_hidden_4 = 16
#n_hidden_5 = 16
#n_hidden_6 = 16
fc_1 = 32
stride = 2

epoch = 100
batch_size = 8
ckpt_keep = 1

win1 = 20
win2 = 20
win3 = 10
win4 = 10
#win5 = 5
#win6 = 5

# make checkpoint folder
ckpt_folder = start.strftime("%Y-%m-%d")
try:
    os.mkdir('ckpt/'+ckpt_folder)
except FileExistsError:
    pass
checkpoint_dir = 'ckpt/'+ckpt_folder


def mfcc_single(data, label0, label1):
    fs, audio = wav.read(data)
    feature = mfcc(audio, samplerate=fs)  # , numcep=26)
    label = np.asarray([label0, label1]).astype(float)
    return feature, label


def generate_batch():
    df_train = pd.read_csv('./data/train.csv')
    df_val = pd.read_csv('./data/validation.csv')
    batch_x, batch_y, val_x, val_y = [], [], [], []
    temp_x, temp_y = [], []
    for i in range(df_train.shape[0]):
        data = df_train.iloc[i, 0]
        label0, label1 = df_train.iloc[i, 1], df_train.iloc[i, 2]
        feature, label = mfcc_single(data, label0, label1)
#        print(feature.shape, df_train.iloc[i, 0])        
        temp_x.append(feature)
        temp_y.append(label)
        if (i % batch_size == (batch_size-1) and i != 0) or batch_size == 1:        
            print('training data processing: {}'.format(i+1))           
            temp_x = np.vstack(temp_x).reshape(batch_size, -1)
            temp_y = np.vstack(temp_y).reshape(batch_size, 2)
            batch_x.append(temp_x)
            batch_y.append(temp_y)
            temp_x, temp_y = [], []
    # for last samples (< batch_size)
    if df_train.shape[0] % batch_size == 0:
        pass
    else:
        x = df_train.shape[0] % batch_size
        temp_x = np.vstack(temp_x).reshape(x, -1)
        temp_y = np.vstack(temp_y).reshape(x, 2)
        batch_x.append(temp_x)
        batch_y.append(temp_y)
    batch_num = len(batch_x)
    print('training data prepared, batch_size: {}, batch_num: {}'.format(batch_x[0].shape, batch_num))
    
    temp_x, temp_y = [], []
    for i in range(df_val.shape[0]):
        if (i+1) % 100 == 0:
            print('validation data processing: {}'.format(i)) 
        data = df_val.iloc[i, 0]
        label0, label1 = df_train.iloc[i, 1], df_train.iloc[i, 2]
        feature, label = mfcc_single(data, label0, label1)
        temp_x.append(feature)
        temp_y.append(label)
    x = df_val.shape[0]
    temp_x = np.vstack(temp_x).reshape(x, -1)
    temp_y = np.vstack(temp_y).reshape(x, 2)
    val_x.append(temp_x)
    val_y.append(temp_y) 
    print('validation data prepared, batch_size: {}'.format(val_x[0].shape))
    
    return batch_x, batch_y, batch_num, val_x, val_y


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, stride, 1, 1], padding='VALID')


def weight_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
#    initial = tf.truncated_normal(shape, stddev=default_stddev)
    return tf.Variable(initial, name=name)
    
    
def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
#    initial = tf.truncated_normal(shape, stddev=default_stddev)
    return tf.Variable(initial, name=name)


# Graph Creation
def deepnn(x):
    # Reshape to use within a convolutional neural net.
    x = tf.reshape(x, [-1, time_step, feature, 1])

     # First convolutional layer
    W_conv1 = weight_variable([win1, 1, 1, n_hidden_1], name='w1')
    b_conv1 = bias_variable([n_hidden_1], name="b1")
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1, name='h1')
    h_conv1 = tf.nn.dropout(h_conv1, keep_prob, name='layer1')

    # Second convolutional layer
    W_conv2 = weight_variable([win2, 1, n_hidden_1, n_hidden_2], name='w2')
    b_conv2 = bias_variable([n_hidden_2], name="b2")
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2, name='h2')
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob, name='layer2')

    # Third convolutional layer
    W_conv3 = weight_variable([win3, 1, n_hidden_2, n_hidden_3], name='w3')
    b_conv3 = bias_variable([n_hidden_3], name="b3")
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3, name='h3')
    h_conv3 = tf.nn.dropout(h_conv3, keep_prob, name='layer3')

    # Fourth convolutional layer
    W_conv4 = weight_variable([win4, 1, n_hidden_3, n_hidden_4], name='w4')
    b_conv4 = bias_variable([n_hidden_4], name="b4")
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4, name='h4')
    h_conv4 = tf.nn.dropout(h_conv4, keep_prob, name='layer4')
    
#    W_conv5 = weight_variable([win5, 1, n_hidden_4, n_hidden_5], name='w5')
#    b_conv5 = bias_variable([n_hidden_5], name="b5")
#    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5, name='h5')
#    h_conv5 = tf.nn.dropout(h_conv5, keep_prob, name='layer5')
#    
#    W_conv6 = weight_variable([win6, 1, n_hidden_5, n_hidden_6], name='w6')
#    b_conv6 = bias_variable([n_hidden_6], name="b6")
#    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6, name='h6')
#    h_conv6 = tf.nn.dropout(h_conv6, keep_prob, name='layer6')
    
    last_shape = h_conv4.get_shape().as_list()
    print('\nlast shape: {}\n'.format(last_shape))
    last_shape = last_shape[1] * last_shape[2] * last_shape[3]
    
    # Fully connected layer 1
    W_fc1 = weight_variable([last_shape, fc_1], name='w_fc1')
    b_fc1 = bias_variable([fc_1], name="b_fc1")
    
    h_pool_flat = tf.reshape(h_conv4, [-1, last_shape], name='hp_flat')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1, name='h_fc1')

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#    # Map the 1024 features to classes, one for each digit
    W_fc2 = weight_variable([fc_1, 2], name='w_fc2')
    b_fc2 = bias_variable([2], name="b_fc2")
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv
   

def train():
    train_accuracies, val_accuracies, x_range, train_loss, val_loss = [], [], [], [], []
    tf.reset_default_graph()
    
    batch_x, batch_y, batch_num, val_x, val_y = generate_batch()
    print('training data prepared, data processing duration: {}'.format(str(datetime.now()-start)))
    
    x = tf.placeholder(tf.float32, [None, time_step * feature], name='input_x')
    y_true = tf.placeholder(tf.float32, [None, 2], name='input_y')
    keep_prob_tf = tf.placeholder(tf.float32, name='keep_prob')
    
    y_predict = deepnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-4,
                                       name='AdamOptimizer').minimize(cost)

    y_model = tf.argmax(y_predict, 1, name='op_to_restore')
    y_real = tf.argmax(y_true, 1)
    correct_prediction = tf.equal(y_model, y_real)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
    
    init_op = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)
        saver = tf.train.Saver(max_to_keep=5)
       
        for i in range(epoch):
            batch_acc, batch_loss = [], []
#            batch_acc_val, batch_loss_val = [], []
        
            for j in range(batch_num):
                loss, _, acc = sess.run([cost, optimizer, accuracy],
                                        feed_dict={x: batch_x[j], y_true: batch_y[j], keep_prob_tf: keep_prob})
                batch_acc.append(acc)
                batch_loss.append(loss)                                                               
            
            val_loss_ep, val_acc_ep = sess.run([cost, accuracy],
                                               feed_dict={x: val_x[0], y_true: val_y[0], keep_prob_tf: 1.0})

            loss_avg = sum(batch_loss) / len(batch_loss)
            acc_avg = sum(batch_acc) / len(batch_acc)
            
            save_path = saver.save(sess, checkpoint_dir + "/kws_v1-" +
                                   datetime.now().strftime("%Y%m%d-%H%M"),
                                   global_step=i+1)
            print("=" * 60)
            print("Epoch:", i+1)
            x_range.append(i+1)
            train_accuracies.append(acc_avg)
            val_accuracies.append(val_acc_ep)
            train_loss.append(loss_avg)
            val_loss.append(val_loss_ep)
                    
            print("Training   Accuracy = {:.3f} %   Training   Loss = {:.6f}".format(acc_avg*100, loss_avg) + 
                  "\nValidation Accuracy = {:.3f} %   Validation Loss = {:.6f}".format(val_acc_ep*100, val_loss_ep))

            print('Checkpoint:', save_path)
            
        # print number of parameters
        print()
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            print(variable)
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('\nTotal parameters: {}\n'.format(total_parameters))

    return train_accuracies, val_accuracies, x_range, train_loss, val_loss


if __name__ == '__main__':
    a = datetime.now()
    train_accuracies, val_accuracies, x_range, loss_epoch, loss_epoch_val = train()
    b = datetime.now()
    training_time = b - a
    print('training duration:', str(training_time))

    plt.figure(1)
    plt.subplot(211)  # the first one of 2x1
    plt.plot(x_range, train_accuracies, 'black', label='Training Accuracy')
    plt.plot(x_range, val_accuracies, '-r', label='Validation Accuracy')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.1)
    plt.xlabel('Epoch')

    plt.subplot(212)  # the second one og 2x1
    plt.plot(x_range, loss_epoch, 'black', label='Training Loss')
    plt.plot(x_range, loss_epoch_val, '-g', label='Validation Loss')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymin=-0.1)
    plt.savefig('graph/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '_nontfrecord.png')
    plt.show()


