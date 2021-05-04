import tensorflow as tf

def networks_model(x1):

    conv31 = tf.layers.conv2d(inputs=x1, filters=200, kernel_size=[8, 4], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31 = tf.layers.max_pooling2d(inputs=conv31, pool_size=[3, 1], strides=3)
    drop31 = tf.layers.dropout(pool31, 0.3)

    conv32 = tf.layers.conv2d(inputs=drop31, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 1], strides=2)
    drop32 = tf.layers.dropout(pool32, 0.3)

    conv33 = tf.layers.conv2d(inputs=drop32, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool33 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2, 1], strides=2)
    drop33 = tf.layers.dropout(pool33, 0.3)

    re31 = tf.reshape(drop33, [-1, 31*200])
    # re31 = tf.reshape(drop33, [-1, 20*200])

    dense31 = tf.layers.dense(inputs=re31, units=100, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    drop34 = tf.layers.dropout(dense31, 0.3)
    logits = tf.layers.dense(inputs=drop34, units=2, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    _y = tf.nn.softmax(logits)

    return _y, logits


def networks_model_frame(x1):

    x2 = tf.expand_dims(x1,3)

    conv31 = tf.layers.conv2d(inputs=x2, filters=200, kernel_size=[8, 1], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31 = tf.layers.max_pooling2d(inputs=conv31, pool_size=[3, 1], strides=3)
    drop31 = tf.layers.dropout(pool31, 0.3)

    conv32 = tf.layers.conv2d(inputs=drop31, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 1], strides=2)
    drop32 = tf.layers.dropout(pool32, 0.3)

    conv33 = tf.layers.conv2d(inputs=drop32, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool33 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2, 1], strides=2)
    drop33 = tf.layers.dropout(pool33, 0.3)

    re31 = tf.reshape(drop33, [-1, 31*200])
    # re31 = tf.reshape(drop33, [-1, 20*200])

    dense31 = tf.layers.dense(inputs=re31, units=100, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    drop34 = tf.layers.dropout(dense31, 0.3)
    logits = tf.layers.dense(inputs=drop34, units=2, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    _y = tf.nn.softmax(logits)

    return _y, logits


def networks_model_deeptis(x1,x3):

    x2 = tf.expand_dims(x1,3)

    conv31 = tf.layers.conv2d(inputs=x2, filters=200, kernel_size=[8, 1], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31 = tf.layers.max_pooling2d(inputs=conv31, pool_size=[3, 1], strides=3)
    drop31 = tf.layers.dropout(pool31, 0.3)

    conv32 = tf.layers.conv2d(inputs=drop31, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 1], strides=2)
    drop32 = tf.layers.dropout(pool32, 0.3)

    conv33 = tf.layers.conv2d(inputs=drop32, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool33 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2, 1], strides=2)
    drop33 = tf.layers.dropout(pool33, 0.3)

    re31 = tf.reshape(drop33, [-1, 31*200])
    # re31 = tf.reshape(drop33, [-1, 20*200])



    conv31_ = tf.layers.conv2d(inputs=x3, filters=200, kernel_size=[8, 4], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31_ = tf.layers.max_pooling2d(inputs=conv31_, pool_size=[3, 1], strides=3)
    drop31_ = tf.layers.dropout(pool31_, 0.3)

    conv32_ = tf.layers.conv2d(inputs=drop31_, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32_ = tf.layers.max_pooling2d(inputs=conv32_, pool_size=[2, 1], strides=2)
    drop32_ = tf.layers.dropout(pool32_, 0.3)

    conv33_ = tf.layers.conv2d(inputs=drop32_, filters=200, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool33_ = tf.layers.max_pooling2d(inputs=conv33_, pool_size=[2, 1], strides=2)
    drop33_ = tf.layers.dropout(pool33_, 0.3)
    re31_ = tf.reshape(drop33_, [-1, 31*200])
    # re31 = tf.reshape(drop33, [-1, 20*200])

    x1_re = tf.concat([re31,re31_], axis=1)

    dense31 = tf.layers.dense(inputs=x1_re, units=100, activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    drop34 = tf.layers.dropout(dense31, 0.3)
    logits = tf.layers.dense(inputs=drop34, units=2, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    _y = tf.nn.softmax(logits)


    return _y, logits


