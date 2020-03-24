import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)



