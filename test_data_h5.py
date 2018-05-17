import time
import os
import shutil
import cv2
import h5py
import numpy as np
from sklearn.externals import joblib
batch_size = 500

## validation phase
# images_path = r'E:\Downloads\ConceptDetectionValidation2017\ConceptDetectionValidation2017\\'
# concepts_path = r'E:\Downloads\ConceptDetectionValidation2017-Concepts.txt'
# val_folder = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\whole_val_folder\\'

# test phase
images_path = r'E:\Downloads\ConceptDetectionTesting2017\ConceptDetectionTesting2017\\'
concepts_path = r'E:\Downloads\ConceptDetectionTesting2017-List.txt'
test_folder = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\whole_test_folder\\'


################################################################
# image processing
print('\nprocess images')
start_time = time.time()

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
val_h5_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\val_h5.txt'

if os.path.exists(test_folder):
    shutil.rmtree(test_folder)
os.mkdir(test_folder)


def return_test_ids(test_text):
    fin = open(test_text)
    test_ids = []
    for line in fin:
        if len(line) < 3:
            continue
        line = line.strip()
        print(line)
        test_ids.append(line)
    return test_ids

test_ids = return_test_ids(concepts_path)


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1))
    return img

batch_img = np.zeros((batch_size, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
batch_ind = 0
batch_counter = 0
test_h5_text = ''

for ind, id_ in enumerate(test_ids):
    h5_file_path = test_folder + 'test_h5_file_' + str(batch_ind) + '.h5'
    img_path = images_path + id_ + '.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    batch_img[batch_counter, :, :, :] = img
    batch_counter += 1
    if batch_counter >= batch_size or ind == len(test_ids) - 1:
        if ind == len(test_ids) - 1:
            batch_img = batch_img[:batch_counter+1]
        with h5py.File(h5_file_path, 'w') as H:
            H.create_dataset('data', data=batch_img)
        print(h5_file_path)
        test_h5_text = test_h5_text + h5_file_path + '\n'
        batch_counter = 0
        batch_ind += 1
        batch_img = np.zeros((batch_size, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
test_h5 = open(test_h5_path, 'w')
test_h5.write(test_h5_text)
test_h5.close()
