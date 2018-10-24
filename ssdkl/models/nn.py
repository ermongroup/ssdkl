import os

import tensorflow as tf
import numpy as np

import ssdkl.models.layers as layers


class NN_UCI():
    """
    Builds computation graph for a small neural net for UCI experiments.
    """
    def __init__(self, dim=10, l2loss=0.0005, train_phase=True,
                 trainable=True, name='nn', embedding_dimension=2):
        with tf.name_scope(name):
            self._setup_nn(dim, l2loss, train_phase, trainable,
                embedding_dimension)

    def _setup_nn(self, dim, l2loss, train_phase, trainable,
                  embedding_dimension):
        """
        Sets up the neural net.
        """
        self.X = tf.placeholder(
            dtype=tf.float32, shape=[None, dim], name='X')
        self.y = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name='y')

        self.hidden1 = layers.fully_connected(
            input_layer=self.X, out_size=100, l2loss=l2loss,
            train_phase=train_phase, trainable=trainable, name='nn_hidden1')
        self.hidden2 = layers.fully_connected(
            input_layer=self.hidden1, out_size=50, l2loss=l2loss,
            train_phase=train_phase, trainable=trainable, name='nn_hidden2')
        self.hidden3 = layers.fully_connected(
            input_layer=self.hidden2, out_size=50, l2loss=l2loss,
            train_phase=train_phase, trainable=trainable, name='nn_hidden3')
        self.embeddings = layers.fully_connected(
            input_layer=self.hidden3, out_size=embedding_dimension, activation_fn=tf.identity, l2loss=l2loss,
            train_phase=train_phase, trainable=trainable, name='nn_embeddings')

        self.predictions = tf.squeeze(
            layers.regression(
                input_layer=self.embeddings, out_size=1, l2loss=l2loss,
                train_phase=train_phase, trainable=False),
            name='nn_regression')
        self.l2_loss = tf.mul(2.0, tf.nn.l2_loss(
            self.predictions - tf.squeeze(self.y)), name='l2_loss')
        self.mse = tf.div(
            self.l2_loss, tf.to_float(tf.shape(self.y)[0]), name='mse')

######################################################################


class CNN():
    """
    Builds computation graph for CNN.
    """
    def __init__(
        self, images, n_conv6=300, n_conv7=30, device='/gpu:0',
            train_phase=True, dropout=False):
        self.n_conv6 = n_conv6
        self.n_conv7 = n_conv7
        self.device = device
        self.set_up_graph(images, train_phase, dropout)

    def set_up_graph(self, images, train_phase, dropout):
        """
        Builds computation graph.
        """
        with tf.device(self.device):
            self.images = images
            self.conv1 = layers.conv2d(
                self.images, 11, 64, 4, pad='VALID',
                bias_init=tf.constant_initializer(value=0.1), l2loss=0.0005,
                train_phase=train_phase, trainable=True, name='conv1')
            self.norm1 = layers.lrn(
                self.conv1, local_size=5, alpha=0.0005, beta=0.75, bias=2.0,
                name='norm1')
            self.pool1 = layers.max_pool(
                self.norm1, 3, 2, pad='SAME', name='pool1')
            self.conv2 = layers.conv2d(
                self.pool1, 5, 256, 1, pad='SAME',
                bias_init=tf.constant_initializer(value=0.1),
                l2loss=0.0005, train_phase=train_phase, trainable=True,
                name='conv2')
            self.norm2 = layers.lrn(
                self.conv2, local_size=5, alpha=0.0005, beta=0.75, bias=2.0,
                name='norm2')
            self.pool2 = layers.max_pool(
                self.norm2, 3, 2, pad='VALID', name='pool2')
            self.conv3 = layers.conv2d(
                self.pool2, 3, 256, 1, pad='SAME',
                bias_init=tf.constant_initializer(value=0.1), l2loss=0.0005,
                train_phase=train_phase, trainable=True, name='conv3')
            self.conv4 = layers.conv2d(
                self.conv3, 3, 256, 1, pad='SAME',
                bias_init=tf.constant_initializer(value=0.1), l2loss=0.0005,
                train_phase=train_phase, trainable=True, name='conv4')
            self.conv5 = layers.conv2d(
                self.conv4, 3, 256, 1, pad='SAME',
                bias_init=tf.constant_initializer(value=0.1), l2loss=0.0005,
                train_phase=train_phase, trainable=True, name='conv5')
            self.pool5 = layers.max_pool(
                self.conv5, 3, 2, pad='SAME', name='pool5')
            self.conv6 = layers.conv2d(
                self.pool5, 6, self.n_conv6, 6, pad='VALID',
                bias_init=tf.constant_initializer(value=0.1), l2loss=0.0005,
                train_phase=train_phase, trainable=True, name='conv6')
            if dropout:
                self.dropout6 = layers.dropout(
                    self.conv6, prob=0.5, name='dropout6')
                self.conv7 = layers.conv2d(
                    self.dropout6, 1, self.n_conv7, 1, pad='VALID',
                    bias_init=tf.constant_initializer(value=0.1),
                    l2loss=0.0005, train_phase=train_phase, trainable=True,
                    name='conv7')
            else:
                self.conv7 = layers.conv2d(
                    self.conv6, 1, self.n_conv7, 1, pad='VALID',
                    bias_init=tf.constant_initializer(value=0.1),
                    l2loss=0.0005, train_phase=train_phase, trainable=True,
                    name='conv7')
            # Extract features
            self.pool7 = layers.average_pool(
                self.conv7, 2, 1, pad='VALID', name='pool7')
            self.features = tf.squeeze(
                self.pool7, squeeze_dims=[1, 2], name='features')


######################################################################


class CNN_MNIST():
    """
    Builds computation graph for CNN for MNIST.
    """
    def __init__(
        self, dim=28, l2loss=0.0005, train_phase=True, trainable=True,
            name='cnn'):
        dim = int(np.sqrt(dim))
        with tf.name_scope(name):
            self._setup_cnn(dim, l2loss, train_phase, trainable)

    def _setup_cnn(self, dim, l2loss, train_phase, trainable):
        """
        Builds computation graph.
        """
        self.X = tf.placeholder(
            dtype=tf.float32, shape=[None, dim * dim], name='X_flat')
        self.X_image = tf.reshape(self.X, shape=[-1, dim, dim, 1], name='X')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
        # Convolution and pool layers
        self.conv1 = layers.conv2d(
            self.X_image, 5, 32, 1, l2loss=l2loss, train_phase=train_phase,
            trainable=trainable, name='cnn_hidden1')
        self.pool1 = layers.max_pool(self.conv1, 2, 2, name='pool1')
        self.conv2 = layers.conv2d(
            self.pool1, 5, 64, 1, l2loss=l2loss, train_phase=train_phase,
            trainable=trainable, name='cnn_hidden2')
        self.pool2 = layers.max_pool(self.conv2, 2, 2, name='pool2')
        self.conv3 = layers.conv2d(
            self.pool2, 3, 64, 1, l2loss=l2loss, train_phase=train_phase,
            trainable=trainable, name='cnn_hidden3')
        self.pool3 = layers.max_pool(self.conv3, 2, 2, name='pool3')
        self.conv4 = layers.conv2d(
            self.pool3, 3, 64, 1, l2loss=l2loss, train_phase=train_phase,
            trainable=trainable, name='cnn_hidden4')
        self.pool4 = layers.max_pool(self.conv4, 2, 2, name='pool4')
        # Fully connected
        self.pool4_flat = tf.reshape(self.pool4, shape=[-1, 2 * 2 * 64])
        self.embeddings = layers.fully_connected2(
            self.pool4_flat, 256, l2loss=l2loss, train_phase=train_phase,
            trainable=trainable, name='cnn_embeddings')
        self.predictions = tf.squeeze(
            layers.regression(
                self.embeddings, out_size=1, l2loss=l2loss,
                train_phase=train_phase, trainable=False),
            name='cnn_regression')
        self.l2_loss = tf.mul(2.0, tf.nn.l2_loss(
            self.predictions - tf.squeeze(self.y)), name='ls_loss')
        self.mse = tf.div(
            self.l2_loss, tf.to_float(tf.shape(self.y)[0]), name='mse')

######################################################################


def weight_variable(shape, name='weights'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

######################################################################


def initialize_from_numpy(path, sess, exclude=set()):
    """
    Initializes parameter values from Numpy arrays.
    """
    for var in tf.trainable_variables():
        toks = var.name.split('/')
        layer = toks[0]
        if layer in exclude:
            continue
        print(var.name)
        print(var.get_shape())
        if 'conv' in var.name:
            if 'weights' in toks[1]:
                sess.run(var.assign(tf.convert_to_tensor(np.load(
                    os.path.join(path, layer + '_filters.npy')).transpose(
                    (2, 3, 1, 0)))))
            if 'bias' in toks[1]:
                sess.run(var.assign(tf.convert_to_tensor(np.load(
                    os.path.join(path, layer + '_bias.npy')))))

######################################################################
