import os
os.environ["GLOG_minloglevel"] = "1"
import caffe
import caffe.draw
import deepdish as dd
import numpy as np
from time import sleep

caffe.set_device(0)
caffe.set_mode_gpu()

# log_file_path = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\ResNet50_threshold_log.txt'
val_hdf5_file = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\val_h5.txt'

#######################################################
# Resnet 10
# folder_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\ResNet10_cvgj\\'
#
# deploy_prototxt_filename = folder_path + 'deploy.prototxt'
#
# caffemodel_path_38700 = folder_path + 'resnet10_cvgj_iter_38700.caffemodel'

#######################################################
# Residual 50
folder_path = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\resnet\\'
deploy_prototxt_filename = folder_path + 'ResNet-50-deploy.prototxt'
# ResNet_50__iter_13500 = folder_path + 'ResNet_50__iter_13500.caffemodel'
# ResNet_50__iter_15000 = folder_path + 'ResNet_50__iter_15000.caffemodel'
ResNet_50__iter_32000 = folder_path + 'ResNet_50__iter_32000.caffemodel'
#######################################################
# threshold_list = [-1, -2, -3]
# caffemodels_list = [3000]


def return_accuracy(actual_label, predicted_label, threshold):
    # predicted_label = np.nan_to_num(predicted_label)
    predicted_label = predicted_label > threshold
    actual_label = actual_label > 0
    correct = sum(actual_label * predicted_label)
    total = sum(actual_label + predicted_label > 0)
    if total > 0:
        return correct / total
    else:
        return 1


def find_accuracy(data, net, threshold):
    inputs = data['data']
    labels = data['label']

    num_of_data = inputs.shape[0]
    total_accuracy = 0
    for index, actual_label in enumerate(labels):
        data = np.zeros((1, 3, 227, 227))
        data[0] = inputs[index]
        outputs = net.forward(data=data)
        predicted_label = np.asarray(outputs[net.outputs[0]])[0, :]
        single_accuracy = return_accuracy(actual_label, predicted_label, threshold)
        total_accuracy += single_accuracy

    total_accuracy /= num_of_data
    # print('total accuracy: ', end='')
    # print(total_accuracy * 100, end='')
    # print('%\n')
    return total_accuracy


def compute_overall_accuracy(caffemodel_path, threshold):
    hdf5_files = open(val_hdf5_file)
    net = caffe.Net(deploy_prototxt_filename, caffemodel_path, caffe.TEST)
    total_accuracy = 0
    num_of_hdf5s = 0
    for line in hdf5_files:
        line = line.replace('\n', '')
        # print(line.replace('E:\Data_Files\Workspaces\PyCharm\concepts_prediction\whole_val_folder\\', ''))
        data = dd.io.load(line)
        total_accuracy += find_accuracy(data, net, threshold)
        num_of_hdf5s += 1
    return_str = 'threshold: ' + str(threshold) + '\noverall accuracy:' +\
                 str(total_accuracy / num_of_hdf5s * 100) + '%\n'
    print(return_str)
    return return_str

if __name__ == '__main__':
    threshold = np.ones(20464) * -2
    compute_overall_accuracy(ResNet_50__iter_32000, threshold=threshold)


# compute_overall_accuracy(ResNet_50__iter_9500, threshold=-2)

# for threshold in threshold_list:
#     compute_overall_accuracy(ResNet_50, threshold=threshold)

# log_file = open(log_file_path, 'w')
# log_file.write('Start\n')
# for caffemodel in caffemodels_list:
#     caffemodel_path = folder_path + 'ResNet_50__iter_' + str(caffemodel) + '.caffemodel'
#     log_file.write('ResNet_50__iter_ ' + str(caffemodel) + '\n')
#     print('ResNet_50__iter_ ' + str(caffemodel))
#     for threshold in threshold_list:
#         log_file.write(str(threshold) + '\n')
#         # print('threshold: ' + str(threshold))
#         log_file.write(compute_overall_accuracy(caffemodel_path, threshold))
#         # print('overall accuracy: ' + str(compute_overall_accuracy(caffemodel_path, threshold)))
#         log_file.write('\n')
#     log_file.write('===========================\n')
#     print('===========================\n')
# log_file.close()
