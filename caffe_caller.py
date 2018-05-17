import caffe
import copy
import google.protobuf
import matplotlib.pyplot as plt
import sklearn.metrics
import caffe.draw
import deepdish as dd
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()

# solver_prototxt_filename = 'E:/Data_Files/Workspaces/PyCharm/concepts_prediction/solver.prototxt'

# caffemodel_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\mynet_model__iter_14000.caffemodel'
# train_prototxt_filename = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\mynet.prototxt'
# deploy_prototxt_filename = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\mynet_deploy.prototxt'

# caffemodel_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\pascalvoc2012_train_simple2_iter_30000.caffemodel'
# train_prototxt_filename = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\train_x30.prototxt'
# deploy_prototxt_filename = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\deploy_x30.prototxt'

# caffemodel_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\resnet\ResNet-50-model.caffemodel'
# train_prototxt_filename = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\resnet\ResNet-50-train.prototxt'
# deploy_prototxt_filename = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\resnet\ResNet-50-deploy.prototxt'
# caffe_solverstate_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\resnet\ResNet-50-model.solverstate'

#######################################################
# Resnet 10
# folder_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\ResNet10_cvgj\\'
# solver_prototxt_filename = folder_path + 'train_solver.prototxt'
# train_prototxt_filename = folder_path + 'train.prototxt'
# deploy_prototxt_filename = folder_path + 'deploy.prototxt'
# caffemodel_path = folder_path + 'resnet10_cvgj_iter_22400.caffemodel'
# caffe_solverstate_path = folder_path + 'resnet10_cvgj_iter_22400.solverstate'

#######################################################
# Residual 50
folder_path = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\resnet\\'
deploy_prototxt_filename = folder_path + 'ResNet-50-deploy.prototxt'
solver_prototxt_filename = folder_path + 'solver.prototxt'
train_prototxt_filename = folder_path + 'train.prototxt'
caffe_solverstate_path = folder_path + 'ResNet-50-model_iter_158000.solverstate'
caffemodel_path = folder_path + 'ResNet-50-model_iter_158000.caffemodel'
#######################################################

def train(solver_prototxt_filename):
    """
    Train the ANN
    """
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.net.copy_from(caffemodel_path)
    solver.restore(caffe_solverstate_path)
    solver.solve()


def print_network_parameters(net):
    """
    Print the parameters of the network
    """
    print(net)
    print('net.inputs: {0}'.format(net.inputs))
    print('net.outputs: {0}'.format(net.outputs))
    print('net.blobs: {0}'.format(net.blobs))
    print('net.params: {0}'.format(net.params))    


def get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net=None):
    """
    Get the predicted output, i.e. perform a forward pass
    """
    if net is None:
        net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)
    out = net.forward(data=input)
    return out[net.outputs[0]]


def print_network(prototxt_filename):
    """
    Draw the ANN architecture
    """
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png')
    print('Draw ANN done!')


def print_network_weights(prototxt_filename, caffemodel_filename):
    """
    For each ANN layer, print weight heatmap and weight histogram 
    """
    net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST)
    for layer_name in net.params: 
        # weights heatmap 
        arr = net.params[layer_name][0].data
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(arr, interpolation='none')
        fig.colorbar(cax, orientation="horizontal")
        plt.savefig('{0}_weights_{1}.png'.format(caffemodel_filename, layer_name), dpi=100, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()

        # weights histogram  
        plt.clf()
        plt.hist(arr.tolist(), bins=20)
        plt.savefig('{0}_weights_hist_{1}.png'.format(caffemodel_filename, layer_name), dpi=100, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()


def get_predicted_outputs(deploy_prototxt_filename, caffemodel_filename, inputs):
    """
    Get several predicted outputs
    """
    outputs = []
    net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)
    for input in inputs:
        outputs.append(copy.deepcopy(get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net)))
    return outputs    


def get_accuracy(true_outputs, predicted_outputs):
    """

    """
    number_of_samples = true_outputs.shape[0]
    number_of_outputs = true_outputs.shape[1]
    threshold = 0  # 0 if SigmoidCrossEntropyLoss ; 0.5 if EuclideanLoss
    for output_number in range(number_of_outputs):
        predicted_output_binary = []
        for sample_number in range(number_of_samples):
            if predicted_outputs[sample_number][0][output_number] < threshold:
                predicted_output = 0
            else:
                predicted_output = 1
            predicted_output_binary.append(predicted_output)

        print('accuracy: {0}'.format(sklearn.metrics.accuracy_score(true_outputs[:, output_number], predicted_output_binary)))
        print(sklearn.metrics.confusion_matrix(true_outputs[:, output_number], predicted_output_binary))


def main():
    """
    This is the main function
    """

    # Set parameters

    # Train network
    train(solver_prototxt_filename)

    # Print network
    print_network(train_prototxt_filename)
    # print_network_weights(train_prototxt_filename, caffemodel_path)
    data = dd.io.load('E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\whole_val_folder\\val_h5_file_0.h5')
    inputs = data['data']
    # for input in inputs:
    #     print(input.shape)
    #     break
    input = np.zeros((1, 3, 227, 227))
    input[0] = inputs[0]
    print(input.shape)
    # Compute performance metrics
    # outputs = get_predicted_outputs(deploy_prototxt_filename, caffemodel_path, inputs)
    net = caffe.Net(deploy_prototxt_filename, caffemodel_path, caffe.TEST)
    outputs = net.forward(data=input)
    outputs = outputs[net.outputs[0]]
    print(outputs)
    # get_accuracy(data['output'], outputs)

if __name__ == "__main__":
    train(solver_prototxt_filename)
