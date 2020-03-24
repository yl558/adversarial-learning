import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import keras
from cleverhans.utils_tf import model_train, model_eval
import cleverhans.attacks_tf as attack
from cleverhans.attacks import SaliencyMapMethod
import scipy.misc

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

# define classifier
layers_cls = [784, 1000, 500, 10]
n = len(layers_cls)
w, b = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('classifier'):
    for i in range(1, n):
        w[i] = weight_variable([layers_cls[i - 1], layers_cls[i]])
        b[i] = bias_variable([layers_cls[i]])

def classifier(x):
    n = len(layers_cls)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            h[i] = z[i]
    logits = h[n - 1]
    return logits

std = [0, 0, 0, 0]
c = [0, 40, 100, 30]
def classifier_n(x):
    n = len(layers_cls)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            perturb = tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            #perturb = tf.sign(perturb) * 0.5
            #perturb_n = tf.clip_by_norm(perturb, 4, 1)
            z[i] += perturb
            z[i] = tf.clip_by_value(z[i], 0, 1)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], c[i], 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], c[i], 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = z[i]
    logits = h[n - 1]
    return logits

# define fake image generator
noise_len = 100
layers_gan = [784 + noise_len, 1000, 784]
s = layers_gan
n = len(layers_gan)
w2, b2 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('generator'):
    for i in range(1, n):
        w2[i] = weight_variable([s[i - 1], s[i]])
        b2[i] = bias_variable([s[i]])

def generator(x, noise, max_perturb = 0.3):
    n = len(layers_gan)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            z[i] = tf.concat([z[i], noise], 1)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w2[i]) + b2[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w2[i]) + b2[i]
            h[i] = tf.nn.tanh(z[i])
    
    #h[n - 1] = tf.clip_by_norm(h[n - 1], max_perturb, axes = 1)
    h[n - 1] = tf.clip_by_value(h[n - 1], -max_perturb, max_perturb)
    x1 = x + h[n - 1]
    x1 = tf.clip_by_value(x1, 0, 1)
    out = x1
    return out

def save_images(images, size, path):
    img = images
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w] = image
    return scipy.misc.imsave(path, merge_img)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def softmax_loss(label, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

def accuracy(label, logits):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float32))

def get_acc(x, label):
    y = classifier(x)
    return accuracy(label, y)


def diff(x1, x2):
    return tf.reduce_mean(tf.norm(x1 - x2, axis = 1))

def main():
    random_seed = 1024
    train_size, test_size = 55000, 10000
    batch_size = 100
    learning_rate = 0.05
    epochs = 5
    steps = epochs * 550
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sess = tf.InteractiveSession(config=config)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    tf.set_random_seed(random_seed)

    x = tf.placeholder(tf.float32, [None, 784])  # input
    label = tf.placeholder(tf.float32, [None, 10])  # true label

    noise = tf.placeholder(tf.float32, [None, 100]) # noise vector
    y_target = tf.placeholder(tf.float32, [None, 10]) # target label

    x_perturb = x + tf.random_normal(shape = tf.shape(x), mean=0.0, stddev= 0.5, dtype=tf.float32)
    x_perturb = tf.clip_by_value(x_perturb, 0, 1)

    x1 = tf.placeholder(tf.float32, [None, 784])
    x2 = tf.placeholder(tf.float32, [None, 784])
    
    y_n = classifier_n(x)
    y = classifier(x)
    y_perturb = classifier(x_perturb)

    # gan 
    x_gan = generator(x, noise)
    y_gan = classifier(x_gan)
    
    loss_cls = softmax_loss(label, y_n) #+ softmax_loss(label, y_gan)
    loss_gan = - softmax_loss(label, y_gan)

    all_vars = tf.trainable_variables()
    c_vars = [var for var in all_vars if 'classifier' in var.name]
    g_vars = [var for var in all_vars if 'generator' in var.name]
    train_op_classifier = GradientDescentOptimizer(learning_rate = learning_rate) \
        .minimize(loss_cls, var_list = c_vars, global_step = global_step)
    train_op_generator = GradientDescentOptimizer(learning_rate = 0.05) \
        .minimize(loss_gan, var_list = g_vars, global_step = global_step)

    #fgsm 
    x_fgsm = attack.fgsm(x, y, eps = 0.2, clip_min=0, clip_max=1)
    y_fgsm = classifier(x_fgsm)
    # random fgsm
    x_fgsm_rd = attack.fgsm(x_perturb, y_perturb, eps = 0.2, clip_min=0, clip_max=1)
    y_fgsm_rd = classifier(x_fgsm_rd)
    # jsma
    jsma = SaliencyMapMethod(classifier, back='tf', sess=sess)
    
    # train
    saver = tf.train.Saver()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x_fgsm_mnist = np.load(os.path.join('data','x_fgsm_mnist.npy'))
    x_gan_mnist = np.load(os.path.join('data','x_gan_mnist.npy'))
    x_jsma_mnist_1 = np.load(os.path.join('data','x_jsma_mnist_1.npy'))

    y_target_batch = np.zeros((100, 10), dtype=np.float32) 
    y_target_batch[:, 0] = 1.0
    y_target_test = np.zeros((10000, 10), dtype=np.float32) 
    y_target_test[:, 0] = 1.0

    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        acc = {} 
        print('train classifier')
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            noise_d = sample_Z(batch_size, 100)
            f_dict = {x: batch[0], label: batch[1], noise: noise_d, y_target: y_target_batch}
            sess.run(train_op_classifier, feed_dict=f_dict)
            #for j in range(1):
                #sess.run(train_op_generator, feed_dict=f_dict)
            if t % 550 == 0:
                epoch = int(t / 550)
                acc['benign'] = sess.run(get_acc(x, label), feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                acc['pre fgsm'] = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_mnist, label: mnist.test.labels})
                acc['pre gan'] = sess.run(get_acc(x, label), feed_dict={x: x_gan_mnist, label: mnist.test.labels})
                acc['pre jsma 1'] = sess.run(get_acc(x, label), feed_dict={x: x_jsma_mnist_1, label: mnist.test.labels[0:100,]})

                x_fgsm_d = sess.run(x_fgsm, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                acc['fgsm'] = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_d, label: mnist.test.labels})
                

                x_fgsm_rd_d = sess.run(x_fgsm_rd, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                acc['fgsm_rd'] = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_rd_d, label: mnist.test.labels})
                
                print(epoch, acc)
        '''
        print('train gan')
        for t in range(1, 550 * 10 + 1):
            batch = mnist.train.next_batch(batch_size)
            f_dict = {x: batch[0], label: batch[1], noise: sample_Z(batch_size, 100), y_target: y_target_batch}
            sess.run(train_op_generator, feed_dict=f_dict)
            if t % 550 == 0:
                epoch = int(t / 550)
                batch = mnist.test.next_batch(batch_size)
                f_dict = {x: batch[0], label: batch[1], noise: sample_Z(batch_size, 100), y_target: y_target_batch}
                x_gan_data = sess.run(x_gan, feed_dict=f_dict)
                acc_gan = sess.run(get_acc(x, label), feed_dict={x: x_gan_data, label: batch[1]})
                print(epoch, acc_gan)

        x_fgsm_d = sess.run(x_fgsm, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
        acc['fgsm'] = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_d, label: mnist.test.labels})

        x_gan_d = sess.run(x_gan, feed_dict={x: mnist.test.images ,label: mnist.test.labels\
            , noise: sample_Z(10000, 100), y_target: y_target_test})
        acc['gan'] = sess.run(get_acc(x, label), feed_dict={x: x_gan_d ,label: mnist.test.labels\
            , noise: sample_Z(10000, 100), y_target: y_target_test})
        '''
        jsma_params = {'theta': 1., 'gamma': 0.1,'nb_classes': 10, 'clip_min': 0.,'clip_max': 1., 'targets': y,\
            'y_val': y_target_batch}
        x_jsma_1_d = jsma.generate_np(mnist.test.images[0:100,], **jsma_params)
        acc['jsma 1'] = sess.run(get_acc(x, label), feed_dict={x: x_jsma_1_d, label: mnist.test.labels[0:100,]})

        print(acc['jsma 1'])
        
    
if __name__ == "__main__":
    main()