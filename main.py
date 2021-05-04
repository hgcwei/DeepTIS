import training_test_network
from common import utils
import time
import numpy as np



def cross_validation(type,method):

    model_folder = 'model_tis_' + method + '/'
    result_folder = 'results_' + method + '/'

    batch_sz = 100
    epoch_num = 10
    learning_rt = 1e-3

    tfrecords_ls = ['data2/'+ type + '/tis_data1_shift1.tfrecords', 'data2/'+ type + '/tis_data1_shift2.tfrecords', 'data2/'+ type + '/tis_data1_shift3.tfrecords','data2/'+ type + '/tis_data1_shift4.tfrecords']

    l = len(tfrecords_ls)
    print(l)
    fprs = np.zeros(l,dtype=np.float32)
    tprs = np.zeros(l,dtype=np.float32)
    aurocs = np.zeros(l,dtype=np.float32)
    auprcs = np.zeros(l,dtype=np.float32)
    time_elaspes = np.zeros(l,dtype=np.float32)

    # for i in range(1):
    for i in range(l):
        test_tfrecords_ls = [tfrecords_ls[i]]
        train_tfrecords_ls = tfrecords_ls[0:i] + tfrecords_ls[i+1:4]

        time_elaspe, model_file, uuid_str = training_test_network.train_model(train_tfrecords_ls, test_tfrecords_ls, model_folder, type, n_epoch=epoch_num,
                                                               learning_rate=learning_rt, batch_size= batch_sz)
        time.sleep(10)
        scores, labels = training_test_network.test_model(model_file, test_tfrecords_ls, batch_sz)
        print(scores)
        print(labels)
        fpr, tpr, auroc, auprc = utils.eval_perf(labels, scores)

        fprs[i] = fpr
        tprs[i] = tpr
        aurocs[i] = auroc
        auprcs[i] = auprc
        time_elaspes[i] = time_elaspe
        utils.save_result(result_folder + type + '/labels' + uuid_str + '.csv', labels)
        utils.save_result(result_folder + type + '/scores' + uuid_str + '.csv', scores)
        utils.save_result(result_folder + type + '/results' + uuid_str + '.csv', [tpr, 1 - fpr, auroc, auprc, time_elaspe])

    utils.save_result(result_folder + type + '/results_avg.csv', [np.mean(tprs), 1 - np.mean(fprs), np.mean(aurocs), np.mean(auprcs), np.mean(time_elaspes)])



cross_validation('gm_imb','frame_shift')
cross_validation('gh_imb','frame_shift')
# cross_validation('gm')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('th')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('tm')

# tfrecords_gen_parser.samples2tfRecord('D:/matlab_projs/DeepTIS2/coding_seqs.txt','test.tfrecords',90)

# type = 'gm'
# # tfrecords_ls = ['../data/' + type + '/data_' + type + '1.tfrecords',
# #                 '../data/' + type + '/data_' + type + '2.tfrecords',
# #                 '../data/' + type + '/data_' + type + '3.tfrecords',
# #                 '../data/' + type + '/data_' + type + '4.tfrecords',
# #                 '../data/' + type + '/data_' + type + '5.tfrecords']
# model_file = 'model/gm/model_402c784bdce9494fabfaf4e217a014c1.ckpt'
# uuid_str = '402c784bdce9494fabfaf4e217a014c1'
# scores = training_test_network.test_model(model_file, ['test.tfrecords'], 400)
# print(scores)
# utils.save_result('scores.csv', scores)

# fpr, tpr, auc = train_test_model.test_model('model/th/model_68015182f83e483883b69401e393bf19.ckpt', [tfrecords_ls[0]], type,'68015182f83e483883b69401e393bf19')
#
#
#














