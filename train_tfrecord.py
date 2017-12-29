import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
from datetime import datetime

start = datetime.now()

# basic setting
feature, time_step = 13, 299
train_filename = "data/train.tfrecords"
val_filename = "data/val.tfrecords"

learning_rate = 0.0001
keep_prob = 1.0
default_stddev = 0.046875
n_hidden_1 = 32
n_hidden_2 = 64
n_hidden_3 = 128
n_hidden_4 = 256

epoch = 200
batch_size = 8
ckpt_keep = 1

samples = sum(1 for _ in tf.python_io.tf_record_iterator(train_filename))
batch_size_val = sum(1 for _ in tf.python_io.tf_record_iterator(val_filename))
training_steps = int(np.ceil(samples/batch_size))

min_after_dequeue = 2

# make checkpoint folder
ckpt_folder = start.strftime("%Y-%m-%d")
try:
    os.mkdir('ckpt/'+ckpt_folder)
except OSError:
    pass
checkpoint_dir = 'ckpt/'+ckpt_folder


# Graph Creation
def deepnn(x):

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

#    def max_pool_2x2(x, name):
#        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=default_stddev)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name):
        # initial = tf.constant(0.0, shape=shape)
        initial = tf.truncated_normal(shape, stddev=default_stddev)
        return tf.Variable(initial, name=name)

    # Reshape to use within a convolutional neural net.
    # Input shape: [batch_size, n_steps, n_input, 1]
    x_image = tf.reshape(x, [-1, time_step, feature, 1])

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 1, n_hidden_1], name='w1')
    b_conv1 = bias_variable([n_hidden_1], name="b1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='h1')
    # Pooling layer - downsamples by 2X.
#    h_pool1 = max_pool_2x2(h_conv1, name='hp1')
    layer_1 = tf.nn.dropout(h_conv1, keep_prob, name='layer1')

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, n_hidden_1, n_hidden_2], name='w2')
    b_conv2 = bias_variable([n_hidden_2], name="b2")
    h_conv2 = tf.nn.relu(conv2d(layer_1, W_conv2) + b_conv2, name='h2')
    # Second pooling layer.
#    h_pool2 = max_pool_2x2(h_conv2, name='hp2')
    layer_2 = tf.nn.dropout(h_conv2, keep_prob, name='layer2')

    # Third convolutional layer
    W_conv3 = weight_variable([5, 5, n_hidden_2, n_hidden_3], name='w3')
    b_conv3 = bias_variable([n_hidden_3], name="b3")
    h_conv3 = tf.nn.relu(conv2d(layer_2, W_conv3) + b_conv3, name='h3')
    # Third pooling layer.
#    h_pool3 = max_pool_2x2(h_conv3, name='hp3')
    layer_3 = tf.nn.dropout(h_conv3, keep_prob, name='layer3')

    # Fourth convolutional layer
    W_conv4 = weight_variable([5, 5, n_hidden_3, n_hidden_4], name='w4')
    b_conv4 = bias_variable([n_hidden_4], name="b4")
    h_conv4 = tf.nn.relu(conv2d(layer_3, W_conv4) + b_conv4, name='h4')
    # Fourth pooling layer.
#    h_pool4 = max_pool_2x2(h_conv4, name='hp4')
    layer_4 = tf.nn.dropout(h_conv4, keep_prob, name='layer4')

    # Fifth convolutional layer
    #W_conv5 = weight_variable([5, 5, n_hidden_4, n_hidden_5], name='w5')
    #b_conv5 = bias_variable([n_hidden_5], name="b5")
    #h_conv5 = tf.nn.relu(conv2d(layer_4, W_conv5) + b_conv5, name='h5')
    # Fifth pooling layer.
    #h_pool5 = max_pool_2x2(h_conv5, name='hp5')
    #layer_5 = tf.nn.dropout(h_pool5, keep_prob, name='layer5')
    
    last_shape = layer_4.get_shape().as_list()
    print('\nlast shape: {}\n'.format(last_shape))
    last_shape = last_shape[1] * last_shape[2] * last_shape[3]

    # Fully connected layer 1
    W_fc1 = weight_variable([last_shape, 1024], name='w_fc1')
    b_fc1 = bias_variable([1024], name="b_fc1")
    h_pool_flat = tf.reshape(layer_4, [-1, last_shape], name='hp_flat')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1, name='h_fc1')

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to classes, one for each digit
    W_fc2 = weight_variable([1024, 2], name='w_fc2')
    b_fc2 = bias_variable([2], name="b_fc2")
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv


def restore_tfrecord(filename, mode):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None, seed=4567)
    reader = tf.TFRecordReader()
    print(reader.num_records_produced())
    print(reader.num_work_units_completed())
    
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'source': tf.FixedLenFeature([feature*time_step], tf.float32),
                                           'target': tf.FixedLenFeature([2], tf.float32),
                                       }) 

    src = features['source']
    tar = features['target']
    
    if mode == 'val':
        bz = batch_size_val
    else:
        bz = batch_size        
    
    capacity = min_after_dequeue+3*batch_size
    source, target = tf.train.shuffle_batch([src, tar],
                                            batch_size=bz, 
                                            num_threads=3, 
                                            capacity=capacity,
                                            min_after_dequeue=min_after_dequeue)
    return source, target


def train():
    a = datetime.now()
    train_accuracies, val_accuracies, x_range, loss_epoch, loss_epoch_val = [], [], [], [], []
    tf.reset_default_graph()
    
    train_x, train_y = restore_tfrecord(train_filename, mode='train')
    val_x, val_y = restore_tfrecord(val_filename, mode='val')
    print('training data prepared, data processing duration: {}'.format(str(datetime.now()-start)))
    
    x = tf.placeholder(tf.float32, [None, feature * time_step], name='input_x')
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
        saver = tf.train.Saver(max_to_keep=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        acc_step, loss_step = [], []
        
        try:
#            while not coord.should_stop():
            for i in range(training_steps * epoch):
                input_x, input_y = sess.run([train_x, train_y])
                loss, _, acc_eval = sess.run([cost, optimizer, accuracy],
                                             feed_dict={x: input_x, y_true: input_y, keep_prob_tf: keep_prob})
                acc_step.append(acc_eval)
                loss_step.append(loss)                                                                                             

                # Print an overview fairly often.
                if i % training_steps == 0:
                    loss_avg = sum(loss_step) / len(loss_step)
                    acc_avg = sum(acc_step) / len(acc_step)
                    input_x, input_y = sess.run([val_x, val_y])
                    loss_val, val_acc = sess.run([cost, accuracy],
                                        feed_dict={x: input_x, y_true: input_y, keep_prob_tf: 1.0})
                    ep = int(i/training_steps)
                    save_path = saver.save(sess, checkpoint_dir + "/kws-" +
                                           datetime.now().strftime("%Y%m%d-%H%M"),
                                           global_step=ep+1)
                    print("=" * 50)
                    print("Epoch:", ep+1)
                    x_range.append(ep+1)
                    train_accuracies.append(acc_avg)
                    val_accuracies.append(val_acc)
                    loss_epoch.append(loss_avg)
                    loss_epoch_val.append(loss_val)
                    
                    print("Training   Accuracy = {:.3f} %   Training   Loss = {:.6f}".format(acc_avg*100, loss_avg) + 
                          "\nValidation Accuracy = {:.3f} %   Validation Loss = {:.6f}".format(val_acc*100, loss_val))
                    print('Checkpoint:', save_path)
                
#                    # simple early stop  # using "from collections import deque"
                    es_train, es_val = train_accuracies[-5:], val_accuracies[-5:]
                    stop_point = sum([es_train[i] >= es_val[i] for i in range(len(es_train))])
                    if stop_point >= 4 and loss_avg < 0.005 and es_val[-1]*100 >= 90.0:
                        print('\nEarly Stop at epoch {}, loss_avg={:.4f}'.format(ep+1, loss_avg))
                        break

                    acc_step, loss_step = [], []

            b = datetime.now()
            training_time = b - a
            print('training duration: {}\n'.format(str(training_time)))
            
            # print number of parameters
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

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
    
    return train_accuracies, val_accuracies, x_range, loss_epoch, loss_epoch_val, save_path


if __name__ == '__main__':
    train_accuracies, validation_accuracies, x_range, loss_epoch, loss_epoch_val, save_path = train()  

    plt.figure(1)
    plt.subplot(211)  # the first one of 2x1
    plt.plot(x_range, train_accuracies, 'black', label='Training Accuracy')
    plt.plot(x_range, validation_accuracies, '-r', label='Validation Accuracy')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.1)
    plt.xlabel('Epoch')

    plt.subplot(212)  # the second one og 2x1
    plt.plot(x_range, loss_epoch, 'black', label='Training Loss')
    plt.plot(x_range, loss_epoch_val, '-g', label='Validation Loss')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymin=-0.1)
    plt.savefig('graph/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '_tfrecord.png')
    plt.show()
   

