import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def data_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, Y_train, X_test, Y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    return X_train, Y_train, X_test, Y_test

def main():
    x_adv_mnist_fsgm = np.load(os.path.join('data','x_adv_mnist_fsgm.npy'))
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.Session()
    print(sess.run(tf.reduce_max(tf.norm(mnist.test.images - x_adv_mnist_fsgm, axis = 1))))


if __name__ == "__main__":
    main()