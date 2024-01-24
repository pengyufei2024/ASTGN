import tf_utils
import tensorflow as tf

def ASTGN(X, TE, SE, P, Q, T, L, K, d, bn, bn_decay, is_training):
    D = K * d
    # input
    X = tf.expand_dims(X, axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P :]
    # encoder
    for _ in range(L):
        X = STAttentionModule(X, STE_P, K, d, bn, bn_decay, is_training)
    # transformAttention layer
    X = transformAttention(
        X, STE_P, STE_Q, K, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X = STAttentionModule(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units = [D, 1], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training,
        use_bias = True, drop = 0.1)
    return tf.squeeze(X, axis = 3)



def FC(x, units, activations, bn, bn_decay, is_training, use_bias = True, drop = None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = tf_utils.dropout(x, drop = drop, is_training = is_training)
        x = tf_utils.conv2d(
            x, output_dims = num_unit, kernel_size = [1, 1], stride = [1, 1],
            padding = 'VALID', use_bias = use_bias, activation = activation,
            bn = bn, bn_decay = bn_decay, is_training = is_training)
    return x

def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    SE = tf.expand_dims(tf.expand_dims(SE, axis = 0), axis = 0)
    SE = FC(
        SE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    dayofweek = tf.one_hot(TE[..., 0], depth = 7)
    timeofday = tf.one_hot(TE[..., 1], depth = T)
    TE = tf.concat((dayofweek, timeofday), axis = -1)
    TE = tf.expand_dims(TE, axis = 2)
    TE = FC(
        TE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return tf.add(SE, TE)

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    D = K * d
    X = tf.concat((X, STE), axis = -1)
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X

def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask = True):
    D = K * d
    X = tf.concat((X, STE), axis = -1)
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    query = tf.transpose(query, perm = (0, 2, 1, 3))
    key = tf.transpose(key, perm = (0, 2, 3, 1))
    value = tf.transpose(value, perm = (0, 2, 1, 3))
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape = (num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
        mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype = tf.bool)
        attention = tf.compat.v2.where(
            condition = mask, x = attention, y = -2 ** 15 + 1)
    attention = tf.nn.softmax(attention, axis = -1)
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    XS = FC(
        HS, units = D, activations = None,
        bn = bn, bn_decay = bn_decay,
        is_training = is_training, use_bias = False)
    XT = FC(
        HT, units = D, activations = None,
        bn = bn, bn_decay = bn_decay,
        is_training = is_training, use_bias = True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return H

def STAttentionModule(X, STE, K, d, bn, bn_decay, is_training, mask = True):
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask = mask)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)

def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    D = K * d
    query = FC(
        STE_Q, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        STE_P, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    query = tf.transpose(query, perm = (0, 2, 1, 3))
    key = tf.transpose(key, perm = (0, 2, 3, 1))
    value = tf.transpose(value, perm = (0, 2, 1, 3))
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X
    
def make_ph(P, Q, N):
    X = tf.compat.v1.placeholder(
        shape = (None, P, N), dtype = tf.float32, name = 'X')
    TE = tf.compat.v1.placeholder(
        shape = (None, P + Q, 2), dtype = tf.int32, name = 'TE')
    label = tf.compat.v1.placeholder(
        shape = (None, Q, N), dtype = tf.float32, name = 'label')
    is_training = tf.compat.v1.placeholder(
        shape = (), dtype = tf.bool, name = 'is_training')
    return X, TE, label, is_training

def mae_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss
