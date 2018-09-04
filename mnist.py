import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time 
from datetime import timedelta
import math
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

weight_matrix_size1 = 5 #5x5 pixcels
depth1 = 16  #16 depth

weight_matrix_size2 = 5 #5x5 pixcels
depth2 = 32  #32 depth

fully_conn_layer = 256 #neuros at end of fully connected layer

#Data dimensions

#We have an input image of 28 x 28 dimensions
img_size = 28

# We have a one hot encoded matrix of length 28*28 = 784
img_size_flat = img_size * img_size

#Shape of the image represented by
img_shape = (img_size,img_size)

#Number of channels in the input image
num_channels = 1

#Number of output classes to be trained on
num_classes = 10

def weight_matrix(dimensions):
    return tf.Variable(tf.truncated_normal(shape = dimensions, stddev=0.1))
def biases_matrix(length):
    return tf.Variable(tf.constant(0.1,shape=[length]))

#Helper functions for ConvNet

def convolutional_layer(input, #The images
                  depth, #channels of the image
                  no_filters, #number of filters in the output
                  weight_matrix_size):
    
    dimensions = [weight_matrix_size,weight_matrix_size, depth, no_filters]
    
    weights = weight_matrix(dimensions)
    
    biases = biases_matrix(length=no_filters)
    
    layer = tf.nn.conv2d(input=input,
                        filter= weights,
                        strides=[1, 1, 1, 1], #stride 2
                        padding='SAME') #input size = output size
    layer += biases
    
    layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    #Passing the pooled layer into ReLU Activation function
    layer = tf.nn.relu(layer)
    
    return layer , weights

# Helper function for Flattening the layer

def flatten_layer(layer):
    
    layer_shape = layer.get_shape()
    
    num_features = layer_shape[1:4].num_elements()
        
    layer_flat = tf.reshape(layer,[-1,num_features])
    
    return layer_flat, num_features

#Helper functions for activation and fully connected

def fully_connected(input,num_inputs,
                num_outputs,
                use_relu = True):
    weights = weight_matrix([num_inputs,num_outputs])
    
    biases = biases_matrix(length= num_outputs)
    
    layer = tf.matmul(input,weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer

#Placeholder variables

x = tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')

x_image = tf.reshape(x, [-1,img_size,img_size,num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'y_true')

y_true_cls = tf.argmax(y_true, axis=1)

# Setting up the network

layer_conv1 , weights_conv1 = convolutional_layer(input = x_image,
                                            depth = num_channels,
                                            weight_matrix_size = weight_matrix_size1,
                                            no_filters = depth1)

#layer_conv1 shape = (-1,14,14,16) and dtype = float32

layer_conv2 , weights_conv2 = convolutional_layer(input = layer_conv1,
                                            depth = depth1,
                                            weight_matrix_size = weight_matrix_size2,
                                            no_filters = depth2)
#layer_conv2 = shape=(?, 7, 7, 36) dtype=float32

#Flattening the layer

layer_flat , num_features = flatten_layer(layer_conv2)

#Fully connected layers

layer_fc1 = fully_connected(input = layer_flat,
                        num_inputs = num_features,
                        num_outputs = fully_conn_layer,
                        use_relu = True)

layer_fc2 = fully_connected(input = layer_fc1,
                        num_inputs = fully_conn_layer,
                        num_outputs = num_classes,
                        use_relu = False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred , axis =1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#optimizing cost function

optimizer = tf.train.AdamOptimizer(learning_rate= 1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#TensorFlow session 
config = tf.ConfigProto(
        device_count = {'cpu': 0}
    )
session = tf.Session(config=config)

session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0

accuracy_ = tf.summary.scalar('accuracy_value', accuracy)
loss_ = tf.summary.scalar('loss_value', cost)

def optimize(num_iterations):
    
    global total_iterations

    start_time = time.time()
    
    summary_op = tf.summary.merge_all()

    file_writer = tf.summary.FileWriter('./tf_mnist.logs', session.graph)
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        session.run(optimizer, feed_dict=feed_dict_train)
        
        acc_value = session.run(accuracy_, feed_dict=feed_dict_train)
        loss_value = session.run(loss_, feed_dict=feed_dict_train)
        file_writer.add_summary(acc_value, i)
        file_writer.add_summary(loss_value, i)
        
        
        if i % 100 == 0:
            
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            
            print(msg.format(i + 1, acc))
            
            total_iterations += num_iterations
        
        
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
optimize(num_iterations=10000)
