from sklearn.externals import joblib
import time
import cv2
import h5py, os, caffe
import sys
import numpy as np
import deepdish as dd
import shutil

################################################################
# parameters etc.

use_whole_train_data = True
use_whole_val_data = True

train_concepts_proceeded = False
val_concepts_proceeded = False
train_labels_proceeded = False
val_labels_proceeded = False
################################################################
# paths

if use_whole_train_data:
    # actual sets
    concepts_path = r'E:\Downloads\ConceptDetectionTraining2017-Concepts.txt'
    images_path = r'E:\Downloads\ConceptDetectionTraining2017\ConceptDetectionTraining2017\\'
    train_folder = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\whole_train_folder\\'
    batch_size = 500
else:
    # toy sets
    concepts_path = r'E:\Downloads\\tmp_concepts.txt'
    images_path = 'E:\Downloads\\tmp_concepts_images_path\\'
    train_folder = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\train_folder\\'
    batch_size = 150

if use_whole_val_data:
    # actual sets
    val_concepts_path = 'E:\Downloads\ConceptDetectionValidation2017-Concepts.txt'
    val_images_path = r'E:\Downloads\ConceptDetectionValidation2017\ConceptDetectionValidation2017\\'
    val_folder = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\whole_val_folder\\'
else:
    # toy sets
    val_concepts_path = 'E:\Downloads\\tmp_val_concepts.txt'
    val_images_path = 'E:\Downloads\\tmp_val_concepts_images_path\\'
    val_folder = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\val_folder\\'


################################################################
# make concepts and ids lists
print('make concepts and ids lists')
start_time = time.time()


def find_ids_concepts_dict(text):
    tmp = []
    for line in text:
        if len(line) < 3:
            continue
        tmp.append(line.replace('\n', '').split("\t"))
    id_concepts = np.array(tmp)
    ids = id_concepts[:,0]
    concepts = [concept.split(',') for concept in id_concepts[:, 1]]
    return dict(zip(ids, concepts))

train_text = open(concepts_path, encoding="utf8")
if not train_concepts_proceeded:
    training_ids_concepts_dict = find_ids_concepts_dict(train_text)
    # joblib.dump(training_ids_concepts_dict, 'doc_ids_concepts_dict.pkl')
else:
    training_ids_concepts_dict = joblib.load('doc_ids_concepts_dict.pkl')

num_to_process = len(training_ids_concepts_dict)
[training_ids, training_concepts] = [list(training_ids_concepts_dict.keys()), list(training_ids_concepts_dict.values())]

val_text = open(val_concepts_path, encoding="utf8")
if not val_concepts_proceeded:
    val_ids_concepts_dict = find_ids_concepts_dict(val_text)
    # joblib.dump(val_ids_concepts_dict, 'doc_val_ids_concepts_dict.pkl')
else:
    val_ids_concepts_dict = joblib.load('doc_val_ids_concepts_dict.pkl')

[val_ids, val_concepts] = [list(val_ids_concepts_dict.keys()), list(val_ids_concepts_dict.values())]

unique_concepts_set = set()
for concept in training_concepts:
    unique_concepts_set.update(concept)
num_of_unique_concepts = len(unique_concepts_set)
print(str(num_of_unique_concepts) + ' unique concepts\n')

# unique_concepts_list = list(unique_concepts_set)
unique_concepts_list = joblib.load('doc_unique_concepts_list.pkl')

# joblib.dump(unique_concepts_list, 'doc_unique_concepts_list.pkl')


def make_output_labels(concepts_list):
    output_labels = np.zeros((len(concepts_list), len(unique_concepts_list)), dtype=np.int8)
    for i in range(output_labels.shape[0]):
        for concept in concepts_list[i]:
            if len(concept) > 3:
                try:
                    concept_index = unique_concepts_list.index(concept)
                    output_labels[i, concept_index] = 1
                except ValueError:
                    pass
    return output_labels


if not train_labels_proceeded:
    training_labels = make_output_labels(training_concepts)
    # joblib.dump(training_labels, 'doc_training_labels.pkl')  # doc_whole_training_labels
else:
    training_labels = joblib.load('doc_training_labels.pkl')  # doc_whole_training_labels

if not val_labels_proceeded:
    val_labels = make_output_labels(val_concepts)
    # joblib.dump(val_labels, 'doc_val_labels.pkl')  # doc_whole_val_labels
else:
    val_labels = joblib.load('doc_val_labels.pkl')  # doc_whole_val_labels

print('concepts and ids lists completed')
print("--- %.2f seconds ---" % (time.time() - start_time))
print()

################################################################
# image processing
print('\nprocess images')
start_time = time.time()

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
train_h5_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\train_h5.txt'
val_h5_path = 'E:\Data_Files\Workspaces\PyCharm\concepts_prediction\\val_h5.txt'

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
os.mkdir(train_folder)

if os.path.exists(val_folder):
    shutil.rmtree(val_folder)
os.mkdir(val_folder)


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
batch_label = np.zeros((batch_size, num_of_unique_concepts))
batch_ind = 0
batch_counter = 0
train_h5_text = ''

for ind, id_ in enumerate(training_ids):
    h5_file_path = train_folder + 'train_h5_file_' + str(batch_ind) + '.h5'
    img_path = images_path + id_ + '.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    label = training_labels[ind]
    batch_img[batch_counter, :, :, :] = img
    batch_label[batch_counter, :] = label
    batch_counter += 1
    if batch_counter >= batch_size or ind == len(training_ids) - 1:
        if ind == len(training_ids) - 1:
            batch_img = batch_img[:batch_counter+1]
            batch_label = batch_label[:batch_counter+1]
        with h5py.File(h5_file_path, 'w') as H:
            H.create_dataset('data', data=batch_img)
            H.create_dataset('label', data=batch_label)
        # dd.io.save(h5_file_path, {'data': batch_img, 'label': batch_label}, compression=None)
        print(h5_file_path)
        train_h5_text = train_h5_text + h5_file_path + '\n'
        batch_counter = 0
        batch_ind += 1
        batch_img = np.zeros((batch_size, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
        batch_label = np.zeros((batch_size, num_of_unique_concepts))
train_h5 = open(train_h5_path, 'w')
train_h5.write(train_h5_text)
train_h5.close()


batch_img = np.zeros((batch_size, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
batch_label = np.zeros((batch_size, num_of_unique_concepts))
batch_ind = 0
batch_counter = 0
val_h5_text = ''

for ind, id_ in enumerate(val_ids):
    h5_file_path = val_folder + 'val_h5_file_' + str(batch_ind) + '.h5'
    img_path = val_images_path + id_ + '.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    label = val_labels[ind]
    batch_img[batch_counter, :, :, :] = img
    batch_label[batch_counter, :] = label
    batch_counter += 1
    if batch_counter >= batch_size or ind == len(val_ids) - 1:
        if ind == len(val_ids) - 1:
            batch_img = batch_img[:batch_counter+1]
            batch_label = batch_label[:batch_counter+1]
        with h5py.File(h5_file_path, 'w') as H:
            H.create_dataset('data', data=batch_img)
            H.create_dataset('label', data=batch_label)
        # dd.io.save(h5_file_path, {'data': batch_img, 'label': batch_label}, compression=None)
        print(h5_file_path)
        val_h5_text = val_h5_text + h5_file_path + '\n'
        batch_counter = 0
        batch_ind += 1
        batch_img = np.zeros((batch_size, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
        batch_label = np.zeros((batch_size, num_of_unique_concepts))
val_h5 = open(val_h5_path, 'w')
val_h5.write(val_h5_text)
val_h5.close()

# print(training_labels.shape)
print(val_labels.shape)
print('image process completed')
print("--- %.2f seconds ---" % (time.time() - start_time))
print()
################################################################
