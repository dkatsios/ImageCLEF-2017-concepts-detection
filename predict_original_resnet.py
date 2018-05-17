#!/usr/bin/python
import cv2
import numpy as np
import os
import re
os.environ["GLOG_minloglevel"] = "1"
import caffe
import sys

caffe.set_device(0)
caffe.set_mode_gpu()

default_image_path = r'C:\Users\dkats\Desktop\bananas.jpg'
deploy_prototxt_filename = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\resnet_50_original\ResNet-50-deploy.prototxt'
caffemodel_path = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\resnet_50_original\ResNet-50-model.caffemodel'
imagenet1000_clsid_path = r'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\resnet_50_original\imagenet1000_clsid_to_human.txt'
label_path = r'C:\Users\dkats\Desktop\predicted_label.txt'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def return_label(predicted_number):
    with open(imagenet1000_clsid_path, 'r') as imagenet1000_clsid:
        for line in imagenet1000_clsid:
            line_parts = line.split(':')
            number = int(re.search(r'\d+', line_parts[0]).group())
            max_label = line_parts[1].split(',')[0].strip().replace("'", '')
            if number == predicted_number:
                return max_label


def predict_label(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net = caffe.Net(deploy_prototxt_filename, caffemodel_path, caffe.TEST)
    outputs = net.forward(data=img)
    predicted_number = np.argmax(outputs['prob'])

    label = return_label(predicted_number)
    print(label)
    return label


if __name__ == '__main__':
    images_path_list = [default_image_path]
    if len(sys.argv) > 1:
        images_path_list = sys.argv[1:]
    for image_path in images_path_list:
        label = predict_label(image_path)
        with open(label_path, 'w') as fout:
            fout.write(label)

