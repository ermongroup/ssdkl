import tensorflow as tf
import math

######################################################################


def lrn(input_layer, local_size, alpha, beta, bias=1.0, name='lrn'):
    """Local Response Normalization"""
    return tf.nn.local_response_normalization(
        input_layer, depth_radius=(local_size // 2),
        alpha=(float(alpha) / local_size), beta=beta, bias=bias, name=name)


def max_pool(input_layer, size, stride, pad='SAME', name='pool'):
    return tf.nn.max_pool(
        input_layer, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],
        padding=pad, name=name)


def average_pool(input_layer, size, stride, pad='SAME', name='pool'):
    return tf.nn.avg_pool(
        input_layer, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],
        padding=pad, name=name)


def conv2d(input_layer, size, depth, stride, pad='SAME',
           activation_fn=tf.nn.relu, bias_init=tf.constant_initializer(0.1),
           l2loss=None, train_phase=True, trainable=True, name='conv'):
    full_size = [size, size, int(input_layer.get_shape()[3]), depth]
    patch_size = full_size[0] * full_size[1]
    weight_initializer = xavier_init(
        full_size[2] * patch_size, full_size[3] * patch_size)
    reuse = not train_phase
    with tf.variable_scope(name, reuse=reuse):
        filt = tf.get_variable(
            name+'_weights', shape=full_size, initializer=weight_initializer,
            trainable=trainable)
        if l2loss is not None:
            weight_decay = tf.mul(
                tf.nn.l2_loss(filt), l2loss, name='weight_decay')
            tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(
            input_layer, filt, [1, stride, stride, 1], padding=pad)
        conv_biases = tf.get_variable(
            name+'_bias', [full_size[-1]], initializer=bias_init,
            trainable=trainable)
        bias = tf.nn.bias_add(conv, conv_biases)
        return activation_fn(bias)


def dropout(input_layer, prob, name='dropout'):
    '''prob should be a placeholder'''
    return tf.nn.dropout(input_layer, prob, name=name)


def fully_connected(input_layer, out_size, activation_fn=tf.nn.relu,
                    bias_init=tf.constant_initializer(0.1), l2loss=None,
                    train_phase=True, trainable=True, name='fc'):
    reuse = not train_phase
    full_size = [int(input_layer.get_shape()[1]), out_size]
    weight_initializer = xavier_init(full_size[0], full_size[1])
    with tf.variable_scope(name, reuse=reuse):
            W = tf.get_variable(
                name+'_W', shape=full_size, initializer=weight_initializer,
                trainable=trainable)
            b = tf.get_variable(
                name+'_b', shape=full_size[-1], initializer=bias_init,
                trainable=trainable)
            if l2loss is not None:
                    wd = tf.mul(tf.nn.l2_loss(W), l2loss, name='weight_decay')
                    tf.add_to_collection('losses', wd)
            mult = tf.matmul(input_layer, W)
            plus_b = tf.nn.bias_add(mult, b)
            return activation_fn(plus_b)


def fully_connected2(input_layer, out_size,
                     bias_init=tf.constant_initializer(0.1), l2loss=None,
                     train_phase=True, trainable=True, name='fc'):
    reuse = not train_phase
    full_size = [int(input_layer.get_shape()[1]), out_size]
    weight_initializer = xavier_init(full_size[0], full_size[1])
    with tf.variable_scope(name, reuse=reuse):
            W = tf.get_variable(
                name+'_W', shape=full_size, initializer=weight_initializer,
                trainable=trainable)
            b = tf.get_variable(
                name+'_b', shape=full_size[-1], initializer=bias_init,
                trainable=trainable)
            if l2loss is not None:
                    wd = tf.mul(tf.nn.l2_loss(W), l2loss, name='weight_decay')
                    tf.add_to_collection('losses', wd)
            mult = tf.matmul(input_layer, W)
            plus_b = tf.nn.bias_add(mult, b)
            return plus_b


def regression(input_layer, out_size=1, bias_init=tf.constant_initializer(0.1),
               l2loss=None, train_phase=True, trainable=True,
               name='regression'):
    reuse = not train_phase
    full_size = [int(input_layer.get_shape()[1]), out_size]
    weight_initializer = xavier_init(full_size[0], full_size[1])
    with tf.variable_scope(name, reuse=reuse):
            Beta = tf.get_variable(
                name+'_Beta', shape=full_size, initializer=weight_initializer,
                trainable=trainable)
            if l2loss is not None:
                    wd = tf.mul(tf.nn.l2_loss(Beta), l2loss,
                                name='weight_decay')
                    tf.add_to_collection('losses', wd)
            mult = tf.matmul(input_layer, Beta)
            return mult

######################################################################


def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
        Understanding the difficulty of training deep feedforward neural
        networks. International conference on artificial intelligence and
        statistics.
    Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
        An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

######################################################################
