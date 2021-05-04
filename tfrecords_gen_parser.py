import tensorflow as tf
from discrete_model import kmer_featurization
import sequential_model
import numpy as np
def count_tfrecord_number(tf_records_ls):
    c = 0
    for fn in tf_records_ls:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c

def parse_tfrecord(filename_ls):
    filename_queue = tf.train.string_input_producer(filename_ls, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'c2_': tf.FixedLenFeature([], tf.string),
          'gkm_': tf.FixedLenFeature([], tf.string),
                    }
            )
    return tf.reshape(tf.decode_raw(features['c2_'],tf.uint8),[90,6,1]),tf.reshape(tf.decode_raw(features['gkm_'],tf.uint8),[640,3,1])


def parse_tfrecord2(filename_ls):
    filename_queue = tf.train.string_input_producer(filename_ls, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'c4_': tf.FixedLenFeature([], tf.string),
          'css_': tf.FixedLenFeature([], tf.string),
          # 'css_': tf.FixedLenFeature([], tf.float32),
          'label': tf.FixedLenFeature([], tf.int64)
                    }
            )
    return tf.reshape(tf.decode_raw(features['c4_'],tf.uint8),[400,4,1]),tf.reshape(tf.decode_raw(features['css_'],tf.float32),[400,1]),features['label']
    # return features['c4_'],features['css_'],features['label']


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_one_sample(line,ws):
     l = len(line)
     seq_list = []
     for i in range(int(l/ws)):
         seq_list.append(line[i*ws:i*ws+ws])
     obj = kmer_featurization(5,3)
     kmer_features = obj.obtain_frame_sensitive_gapped_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)
     c2_features = sequential_model.obtain_c2_feature_for_a_list_of_sequences(seq_list)
     return c2_features,kmer_features.T


def encode_one_sample2(line, ws):
    l = len(line)
    c4_features = sequential_model.obtain_c4_feature_for_one_sequence(line[0:ws])
    return c4_features,int(line[l-1])

def samples2tfRecord(filename,recordname,ws):
    f = open(filename)
    writer = tf.python_io.TFRecordWriter(recordname)
    for line in f.readlines():
        line = line.strip('\n')
        c2,gkm = encode_one_sample(line,ws)
        c2_ = c2.tostring()
        gkm_ = gkm.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'c2_':_bytes_feature(c2_),
                'gkm_': _bytes_feature(gkm_),
                }))
        writer.write(example.SerializeToString())
    writer.close()
    return recordname

def samples2tfRecord2(filename1,filename2, recordname,ws):
    f = open(filename1)
    writer = tf.python_io.TFRecordWriter(recordname)
    # css_mat = np.loadtxt(filename2,dtype=np.float32)
    css_mat = np.loadtxt(open(filename2, "rb"), delimiter=",", skiprows=0, dtype=np.float32)
    i = 0
    for line in f.readlines():
        line = line.strip('\n')
        c4,label = encode_one_sample2(line,ws)
        css = css_mat[i,:]
        css = np.reshape(css,[400,1])
        # css = np.array(css, dtype=np.float32)
        i += 1
        c4_ = c4.tostring()
        css_ = css.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'c4_':_bytes_feature(c4_),
                'css_': _bytes_feature(css_),
                # 'css_': _floats_feature(css),
                'label': _int64_feature(label)
                }))
        writer.write(example.SerializeToString())
    writer.close()
    return recordname

#
# samples2tfRecord2('D:/matlab_projs/DeepTIS2/tis_seqs1.txt','scores1.csv','tis_data1.tfrecords',400)
# samples2tfRecord2('D:/matlab_projs/DeepTIS2/tis_seqs2.txt','scores2.csv','tis_data2.tfrecords',400)
# samples2tfRecord2('D:/matlab_projs/DeepTIS2/tis_seqs3.txt','scores3.csv','tis_data3.tfrecords',400)
# samples2tfRecord2('D:/matlab_projs/DeepTIS2/tis_seqs4.txt','scores4.csv','tis_data4.tfrecords',400)
# print(count_tfrecord_number(['tis_train.tfrecords']))
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_tm2.txt','data_tm2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_tm3.txt','data_tm3.tfrecords',90)

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/test_tm2.txt','test_tm2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_th2.txt','data_th2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_th3.txt','data_th3.tfrecords',90)

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gh1.txt','data_gh1.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gh2.txt','data_gh2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gh3.txt','data_gh3.tfrecords',90)

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gm1.txt','data_gm1.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gm2.txt','data_gm2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gm3.txt','data_gm3.tfrecords',90)
#
# x = np.loadtxt('scores.csv',dtype=np.float32)
# x_split = np.vsplit(x,4)
# np.savetxt('scores1.csv',x_split[0],delimiter=',')
# np.savetxt('scores2.csv',x_split[1],delimiter=',')
# np.savetxt('scores3.csv',x_split[2],delimiter=',')
# np.savetxt('scores4.csv',x_split[3],delimiter=',')

