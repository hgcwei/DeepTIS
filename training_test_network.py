import tensorflow as tf
import numpy as np
import time
import os
import uuid
import tfrecords_gen_parser
import network_model
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(train_tfrecords_ls,test_tfrecords_ls,model_folder,type,n_epoch=3,learning_rate=1e-3,batch_size=500):

    h1 = 400
    w1 = 4

    h2 = 400
    c = 1

    pw = 1.0

    x1 = tf.placeholder(tf.float32, shape=[None, h1, w1, c], name='x1')
    x2 = tf.placeholder(tf.float32, shape=[None, h2, c], name='x2')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    y2,logits = network_model.networks_model_frame(x2)
    # y2,logits = network_model.networks_model(x1)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=pw))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # c2_, kmer_, label_ = parse_tfrecord('C:/Users/Wei/PycharmProjects/tfrecord_test/TestFinal/train_mh.tfrecords')
    c4_, css_, label_ = tfrecords_gen_parser.parse_tfrecord2(train_tfrecords_ls)
    c4_batch, css_batch, label_batch = tf.train.shuffle_batch([c4_, css_, label_],batch_size=batch_size, num_threads=64, capacity=10000, min_after_dequeue=1000)

    # c2_test, kmer_test, label_test = parse_tfrecord('C:/Users/Wei/PycharmProjects/tfrecord_test/TestFinal/test.tfrecords')
    c4_test, css_test, label_test = tfrecords_gen_parser.parse_tfrecord2(test_tfrecords_ls)
    c4_batch_test, css_batch_test, label_batch_test = tf.train.shuffle_batch([c4_test, css_test, label_test],batch_size=batch_size, capacity=10000, min_after_dequeue=1000)
    # c2_batch_test2, kmer_batch_test2, label_batch_test2 = tf.train.batch([c2_test, kmer_test, label_test],
    #                                                                           batch_size=1, capacity=100,name='test_batch2')
    # 训练和测试数据，可将n_epoch设置更大一些
    # sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    start = time.clock()

    total_num = tfrecords_gen_parser.count_tfrecord_number(train_tfrecords_ls)
    print(total_num)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        batch_idxs = int(total_num / batch_size)
        # batch_idxs = int(2533951 / batch_size)

        for epoch in range(n_epoch):
            train_loss, train_acc, train_batch = 0, 0, 0
            val_loss, val_acc, val_batch = 0, 0, 0
            for j in range(batch_idxs):
                c4_batchs, css_batchs, label_batchs = sess.run([c4_batch,css_batch,label_batch])
                _, err, ac = sess.run([train_op, loss, acc], feed_dict={x1: c4_batchs, x2: css_batchs, y_: label_batchs})
                train_loss += err
                train_acc += ac
                train_batch += 1

                c2_batchs_test, css_batchs_test, label_batchs_test = sess.run([c4_batch_test, css_batch_test, label_batch_test])
                err, ac = sess.run([loss, acc], feed_dict={x1: c2_batchs_test, x2: css_batchs_test, y_: label_batchs_test})
                val_loss += err
                val_acc += ac
                val_batch += 1

                if np.mod(j, 50) == 0:
                    print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f" % (n_epoch, epoch + 1, train_loss / train_batch, train_acc / train_batch, val_loss / val_batch, val_acc / val_batch))

        end = time.clock()
        time_elaspe = end - start
        print("time elaspe: %s" % time_elaspe)

        uuid_str = uuid.uuid4().hex
        model_file = model_folder + type + '/model_%s.ckpt' % uuid_str

        saver.save(sess, model_file)
        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()
    return time_elaspe, model_file, uuid_str

def test_model(model_file, test_tfrecord_ls, batch_size):
    total_num = tfrecords_gen_parser.count_tfrecord_number(test_tfrecord_ls)
    h1 = 400
    w1 = 4

    h2 = 400
    c = 1
    x1 = tf.placeholder(tf.float32, shape=[None, h1, w1, c], name='x1')
    x2 = tf.placeholder(tf.float32, shape=[None, h2, c], name='x2')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    y2,logits = network_model.networks_model_frame(x2)

    c4_test, css_test, label_test = tfrecords_gen_parser.parse_tfrecord2(test_tfrecord_ls)
    c4_batch_test, css_batch_test, label_batch_test = tf.train.shuffle_batch([c4_test, css_test, label_test],
                                                                              batch_size=batch_size, capacity=10000,
                                                                              min_after_dequeue=1000, name='test_batch')
    # c2_batch_test2, kmer_batch_test2, label_batch_test2 = tf.train.batch([c2_test, kmer_test, label_test],
    #                                                                           batch_size=1, capacity=100,name='test_batch2')


    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        batch_idxs = int(total_num / batch_size)
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        scores = []
        labels = []
        for i in range(batch_idxs):
            # c2_batchs_test, kmer_batchs_test, label_batchs_test = sess.run([c2_batch_test, kmer_batch_test, label_batch_test])
            c4_batchs_test, css_batchs_test, label_batchs_test = sess.run(
                [c4_batch_test, css_batch_test, label_batch_test])
            sco = sess.run(y2, feed_dict={x1: c4_batchs_test, x2: css_batchs_test, y_: label_batchs_test})
            scores.append(sco[:, 1])
            labels.append(label_batchs_test)

        scores, labels = np.asarray(scores, np.float32), np.asarray(labels, np.int32)
        scores = np.reshape(scores, (-1, 1))
        labels = np.reshape(labels, (-1, 1))
        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()
    return scores, labels