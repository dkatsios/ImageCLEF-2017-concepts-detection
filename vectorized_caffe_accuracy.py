import os
from sklearn.metrics import f1_score
from numpy import linspace
import numpy as np
from numba import vectorize
from tempfile import mkdtemp
import os.path as path
import gzip
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing



os.environ["GLOG_minloglevel"] = "1"
import caffe
import deepdish as dd
import numpy as np
from time import sleep, time

caffe.set_device(0)
caffe.set_mode_gpu()

val_hdf5_file = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\val_h5.txt'
val_h5_vectors_train = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\val_h5_vectors_train.txt'
train_h5_file = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\whole_train_folder\\'
folder_path = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\resnet\\'
deploy_prototxt_filename = folder_path + r'ResNet-50-deploy.prototxt'
caffemodel_path = folder_path + r'ResNet_50__iter_32000.caffemodel'
tuples_matrices_gz = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\tuples_matrices_gz.txt'

num_of_labels = 20464
matrix_constructed = False
thresholds_computed = False
infinitesimal = 0.1

f = open(val_h5_vectors_train)
counter = 0
for line in f:
    counter += 1
num_of_images = 500 * counter


def construct_tuples_matrix():
    net = caffe.Net(deploy_prototxt_filename, caffemodel_path, caffe.TEST)
    hdf5_files = open(val_h5_vectors_train)
    tuples_matrix = []
    for line in hdf5_files:
        line = line.strip()
        data = dd.io.load(line)
        inputs = data['data']
        labels = data['label']

        for index, actual_labels in enumerate(labels):
            data = np.zeros((1, 3, 227, 227))
            data[0] = inputs[index]
            outputs = net.forward(data=data)
            predicted_labels = np.asarray(outputs[net.outputs[0]])[0, :]
            new_tuple_list = list(zip(actual_labels, predicted_labels))
            tuples_matrix.append(new_tuple_list)
    tuples_matrix = np.array(tuples_matrix)
    return tuples_matrix


def construct_tuples_matrix_2(tuples_matrix):
    net = caffe.Net(deploy_prototxt_filename, caffemodel_path, caffe.TEST)
    hdf5_files = open(val_h5_vectors_train)
    for hdf5_num, line in enumerate(hdf5_files):
        line = line.strip()
        data = dd.io.load(line)
        inputs = data['data']
        labels = data['label']

        for index, actual_labels in enumerate(labels):
            data = np.zeros((1, 3, 227, 227))
            data[0] = inputs[index]
            outputs = net.forward(data=data)
            predicted_labels = np.asarray(outputs[net.outputs[0]])[0, :]
            new_tuple_list = list(zip(actual_labels, predicted_labels))
            tuples_matrix[hdf5_num * 500 + index, :] = new_tuple_list
            # tuples_matrix.append(new_tuple_list)
    return tuples_matrix


def return_tuples_list(tuples_matrix, threshold_num):
    # tuples_list = []
    for example_num in range(len(tuples_matrix)):
        # tuples_list.append(tuples_matrix[example_num][threshold_num])
        yield tuples_matrix[example_num][threshold_num]
    # return tuples_list


def compute_threshold_2(tuples_matrix, threshold_num, num_range):
    ###############
    print(threshold_num)
    num_of_thresholds = 10
    infinitesimal = 0.5
    best_thresholds = []
    for i in range(num_range):
        if i + threshold_num >= tuples_matrix.shape[1]:
            break
        actual_label = np.array([1 if label[0] > 0 else 0 for label in tuples_matrix[:, threshold_num + i]])
        predicted_label = np.array([label[1] for label in tuples_matrix[:, threshold_num + i]])
        thresholds_list = linspace(min(predicted_label) - infinitesimal, max(predicted_label) + infinitesimal,
                                   num_of_thresholds)
        ###################
        # num_cores = multiprocessing.cpu_count()
        # F1_scores = Parallel(n_jobs=1, verbose=50)(delayed(return_similarity)
        #                                        (actual_label, predicted_label, thres) for thres in thresholds_list)
        # best_threshold = thresholds_list[F1_scores.index(max(F1_scores))]
        # return best_threshold
        ###################
        F1_scores = [return_similarity(actual_label, predicted_label, thres) for thres in thresholds_list]
        best_threshold = thresholds_list[F1_scores.index(max(F1_scores))]
        best_thresholds.append(best_threshold)
    return best_thresholds


def compute_threshold(tuples_matrix, threshold_num, num_range):
    ###############
    if threshold_num % 500 == 0:
        print(threshold_num)
    num_of_thresholds = 10
    infinitesimal = 1
    actual_label = np.array([1 if label[0] > 0 else 0 for label in tuples_matrix[:, threshold_num]])
    predicted_label = np.array([label[1] for label in tuples_matrix[:, threshold_num]])
    thresholds_list = linspace(min(predicted_label) - infinitesimal, max(predicted_label) + infinitesimal, num_of_thresholds)
    ###################
    # num_cores = multiprocessing.cpu_count()
    # F1_scores = Parallel(n_jobs=1, verbose=50)(delayed(return_similarity)
    #                                        (actual_label, predicted_label, thres) for thres in thresholds_list)
    # best_threshold = thresholds_list[F1_scores.index(max(F1_scores))]
    # return best_threshold
    ###################
    F1_scores = [return_similarity(actual_label, predicted_label, thres) for thres in thresholds_list]
    best_threshold = thresholds_list[F1_scores.index(max(F1_scores))]
    return best_threshold
    ###################

    # while rep_num < 200:
    #     rep_num += 1
    #     mean_threshold = (min_threshold + max_threshold) / 2
    #     min_F1_score = return_score(actual_label, predicted_label, min_threshold)
    #     mean_F1_score = return_score(actual_label, predicted_label, mean_threshold)
    #     max_F1_score = return_score(actual_label, predicted_label, max_threshold)
    #     if max(abs(min_F1_score - mean_F1_score), abs(max_F1_score - mean_F1_score)) <= acceptable_f1_error:
    #         return mean_threshold
    #     if min_F1_score <= mean_F1_score <= max_F1_score:
    #         min_threshold = mean_threshold
    #     elif min_F1_score >= mean_F1_score >= max_F1_score:
    #         max_threshold = mean_threshold
    #     else:
    #         left_mean_threshold = (min_threshold + mean_threshold) / 2
    #         left_mean_F1_score  = return_score(actual_label, predicted_label, left_mean_threshold)
    #         right_mean_threshold = (max_threshold + mean_threshold) / 2
    #         right_mean_F1_score = return_score(actual_label, predicted_label, right_mean_threshold)
    #         if min_F1_score <= left_mean_F1_score <= mean_F1_score:
    #             min_threshold = left_mean_threshold
    #         elif max_F1_score <= right_mean_F1_score <= mean_F1_score:
    #             max_threshold = right_mean_threshold
    # print(threshold_num, 'repetitions termination')
    # return mean_threshold

    ###############

    # zeros_list = []
    # ones_list = []
    # for label in tuples_list:
    #     if label[0] == 0:
    #         zeros_list.append(label[1])
    #     else:
    #         ones_list.append(label[1])
    # if len(zeros_list) == 0:
    #     return min(ones_list) - infinitesimal
    # if len(ones_list) == 0:
    #     return max(zeros_list) + infinitesimal
    # max_zero = max(zeros_list)
    # min_one = min(ones_list)
    # if max_zero <= min_one:
    #     return (min_one + max_zero) / 2
    # short_zeros_list = [x for x in zeros_list if x >= min_one]
    # short_ones_list = [x for x in ones_list if x <= max_zero]
    # upper_limit = max_zero
    # lower_limit = min_one
    # # return lower_limit
    # while True:
    #     threshold = (upper_limit + lower_limit) / 2
    #     wrong_zeros_num = sum(num >= threshold for num in short_zeros_list)
    #     wrong_ones_num = sum(num <= threshold for num in short_ones_list)
    #     if abs(wrong_ones_num - wrong_zeros_num) <= 1:
    #         return threshold
    #     if wrong_zeros_num > wrong_ones_num:
    #         lower_limit = threshold
    #     else:
    #         upper_limit = threshold


@vectorize(['int32(int32, int32)'], target='cuda')
def find_result_vector(a, p):
    return (10 * a * p) + ((a - 1) * (p - 1)) + (a * (p - 1))
    # return a * p


def return_similarity(actual_label, predicted_label, threshold):
    ones_weight = 10
    zeros_weight = 1
    one_zero_weight = 0
    try:
        predicted_label = np.array([1 if x[0] > x[1] else 0 for x in zip(predicted_label, threshold)])
    except TypeError:
        predicted_label = np.array([1 if n > threshold else 0 for n in predicted_label])
    # predicted_label = np.asarray(predicted_label)
    # actual_label = np.array([1 if n > 0 else 0 for n in actual_label])
    # actual_label = np.asarray(actual_label)
    # lenght = len(actual_label)
    counter = 0

    #########
    # similarity = sum(find_result_vector(actual_label, predicted_label))
    #########
    # ones = sum(find_result_vector(actual_label, predicted_label))
    # zeros = sum(find_result_vector(actual_label - 1, predicted_label - 1))
    # one_zero = -sum(find_result_vector(actual_label, predicted_label - 1))
    # similarity = (ones * ones_weight) + (zeros * zeros_weight) + (one_zero * one_zero_weight)
    #########
    ones = sum(predicted_label * actual_label)
    zeros = sum((predicted_label - 1) * (actual_label - 1))
    one_zero = -sum(actual_label * (predicted_label - 1))
    similarity = (ones * ones_weight) + (zeros * zeros_weight) + (one_zero * one_zero_weight)
    #########
    return similarity
    # for n in zip(actual_label, predicted_label):
    #     if n[0] * n[1] == 1:
    #         counter += 8
    #     elif n[0] == n[1]:
    #         counter += 1
    #     elif n[0] == 1 and n[1] == 0:
    #         counter -= 2
    # # similarity = counter #/ lenght
    # return counter


def return_score(actual_label, predicted_label, threshold):
    # predicted_label = np.nan_to_num(predicted_label)
    try:
        predicted_label = [x[0] > x[1] for x in zip(predicted_label, threshold)]
    except TypeError:
        predicted_label = [n > threshold for n in predicted_label]
    actual_label = [n > 0 for n in actual_label]
    F1_score = f1_score(actual_label, predicted_label)
    return F1_score


def find_f1_score(data, net, threshold):
    inputs = data['data']
    labels = data['label']

    num_of_data = inputs.shape[0]
    total_precision = 0
    total_recall = 0
    F1_score = 0
    for index, actual_label in enumerate(labels):
        data = np.zeros((1, 3, 227, 227))
        data[0] = inputs[index]
        outputs = net.forward(data=data)
        predicted_label = np.asarray(outputs[net.outputs[0]])[0, :]
        # F1_score += f1_score(actual_label, predicted_label)
        F1_score += return_score(actual_label, predicted_label, threshold)
        # total_precision += precision
        # total_recall += recall
    F1_score /= num_of_data
    # total_precision /= num_of_data
    # total_recall /= num_of_data
    # f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall)
    # print('total accuracy: ', end='')
    # print(total_accuracy * 100, end='')
    # print('%\n')
    return F1_score


def compute_thresholds(tuples_matrix):
    num_range = 1000
    ###############
    tmp_thresholds = Parallel(n_jobs=-1)\
        (delayed(compute_threshold_2)(tuples_matrix, threshold_num, num_range)
         for threshold_num in range(0, num_of_labels, num_range))
    thresholds = [item for sublist in tmp_thresholds for item in sublist]
    ###############

    # thresholds = []
    # for threshold_num in range(num_of_labels):
    #     # tuples_list = return_tuples_list(tuples_matrix, threshold_num)
    #     threshold = compute_threshold(tuples_matrix, threshold_num, num_range)
    #     thresholds.append(threshold)
    #     # print(threshold_num)
    return thresholds


def compute_overall_accuracy(caffemodel_path, threshold):
    hdf5_files = open(val_hdf5_file)
    net = caffe.Net(deploy_prototxt_filename, caffemodel_path, caffe.TEST)
    total_f1_score = 0
    num_of_hdf5s = 0
    for line in hdf5_files:
        line = line.replace('\n', '')
        data = dd.io.load(line)
        total_f1_score += find_f1_score(data, net, threshold)
        num_of_hdf5s += 1
    return_str = 'threshold: ' + str(threshold) + '\noverall accuracy:' + \
                 str(total_f1_score / num_of_hdf5s * 100) + '%\n'
    print(return_str)
    return return_str


def construct_matrix():
    start_time = time()
    # tuples_matrix = construct_tuples_matrix()

    # tuples_matrix = np.empty((num_of_images, num_of_labels, 2))
    # tuples_matrix = construct_tuples_matrix_2(tuples_matrix)

    if matrix_constructed:
        # tuples_matrix = joblib.load('doc_tuples_matrix.pkl')
        f = gzip.GzipFile('tuples_matrix.npy.gz', 'r')
        tuples_matrix = np.load(file=f)
        f.close()
    else:
        tuples_matrix = np.empty((num_of_images, num_of_labels, 2), dtype='float32')
        tuples_matrix = construct_tuples_matrix_2(tuples_matrix)
        # joblib.dump(tuples_matrix, 'doc_tuples_matrix.pkl')
        gzip_time = time()
        f = gzip.GzipFile('tuples_matrix_52_55.npy.gz', 'w')
        np.save(file=f, arr=tuples_matrix)
        f.close()
        print("--- %.2f seconds ---" % (time() - gzip_time))
        print('tuples matrix gziped')
        return tuples_matrix

    print("--- %.2f seconds ---" % (time() - start_time))
    print('tuples matrix computed')


def return_thresholds(tuples_matrix):
    start_time = time()

    if thresholds_computed:
        thresholds = joblib.load('doc_thresholds.pkl')
    else:
        thresholds = compute_thresholds(tuples_matrix)
        # joblib.dump(thresholds, 'doc_thresholds.pkl')

    print("--- %.2f seconds ---" % (time() - start_time))
    print('thresholds computed')

    print("--- %.2f seconds ---" % (time() - training_time))
    print('training time')
    return thresholds


def compute_combined_matrix(start, stop):
    f = open(tuples_matrices_gz)
    is_first = True
    for line in f:
        line = line.strip()
        f_tmp = gzip.GzipFile(line, 'r')
        tuples_matrix_tmp = np.load(file=f_tmp)
        f_tmp.close()
        tuples_matrix_tmp = tuples_matrix_tmp[:, start:stop]
        if is_first:
            tuples_matrix = tuples_matrix_tmp
            is_first = False
        else:
            tuples_matrix = np.concatenate((tuples_matrix, tuples_matrix_tmp), axis=0)
    f.close()
    # f0 = gzip.GzipFile('tuples_matrix_0_3.npy.gz', 'r')
    # tuples_matrix_0 = np.load(file=f0)
    # f0.close()
    # tuples_matrix_0 = tuples_matrix_0[:, start:stop]
    # f1 = gzip.GzipFile('tuples_matrix_4_7.npy.gz', 'r')
    # tuples_matrix_1 = np.load(file=f1)
    # f1.close()
    # tuples_matrix_1 = tuples_matrix_1[:, start:stop]
    # tuples_matrix = np.concatenate((tuples_matrix_0, tuples_matrix_1), axis=0)
    return tuples_matrix


if __name__ == '__main__':
    # tuples_matrix = construct_matrix()

    f = open(tuples_matrices_gz)
    for i, l in enumerate(f):
        pass
    num_of_gzs = i + 1

    training_time = time()
    milestones = list(range(0, num_of_labels, 7000))
    if num_of_labels not in milestones:
        milestones.append(num_of_labels)
    thresholds = []
    for index in range(len(milestones) - 1):
        milestone_0 = milestones[index]
        milestone_1 = milestones[index + 1]
        tmp_tuples_matrix = compute_combined_matrix(milestone_0, milestone_1)
        tmp_thresholds = return_thresholds(tmp_tuples_matrix)
        thresholds = thresholds + tmp_thresholds
    joblib.dump(thresholds, 'doc_thresholds_{}.pkl'.format(num_of_gzs))

    # thresholds_1_2 = return_thresholds(tuples_matrix)
    # tuples_matrix = compute_combined_matrix(milestone_1, num_of_labels)
    # thresholds_2_2 = return_thresholds(tuples_matrix)
    # thresholds = thresholds_1_2 + thresholds_2_2

    # start_time = time()
    # thresholds = np.ones(num_of_labels) * -2
    # compute_overall_accuracy(caffemodel_path, threshold=thresholds)
    # print("--- %.2f seconds ---" % (time() - start_time))
    # print('-2 thresholds accuracy computed')

    start_time = time()
    compute_overall_accuracy(caffemodel_path, threshold=thresholds)
    print("--- %.2f seconds ---" % (time() - start_time))
    print('vectorized thresholds accuracy computed')
