import numpy as np
import pandas as pd
import nltk
import re
from PIL import Image
import os
import ssim
import heapq
import mahotas
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time

################################################################
# parameters etc.

use_whole_train_data = False
use_whole_val_data = False

tfidf_lsi_computed = False
train_concepts_proceeded = False
val_concepts_proceeded = False
cluster_created = False
forest_found = False

num_clusters = 80   # number of clusters at k-means algorithm
num_to_compare = 20  # number of images to be considered similar

################################################################
# paths

if use_whole_train_data:
    # actual sets
    concepts_path = r'E:\Downloads\ConceptDetectionTraining2017-Concepts.txt'
    images_path = r'E:\Downloads\ConceptDetectionTraining2017\CaptionPredictionTraining2017\\'
    num_topics = 300
    min_df = 5
else:
    # toy sets
    concepts_path = r'E:\Downloads\\tmp_concepts.txt'
    images_path = 'E:\Downloads\\tmp_concepts_images_path\\'
    num_topics = 10
    min_df = 1

if use_whole_val_data:
    # actual sets
    val_concepts_path = 'E:\Downloads\ConceptDetectionValidation2017-Concepts.txt'
    val_images_path = r'E:\Downloads\ConceptDetectionValidation2017\CaptionPredictionValidation2017\\'
else:
    # toy sets
    val_concepts_path = 'E:\Downloads\\tmp_val_concepts.txt'
    val_images_path = 'E:\Downloads\\tmp_val_concepts_images_path\\'


################################################################
# make concepts and ids lists
print('make concepts and ids lists')
start_time = time.time()


def find_ids_concepts_dict(text):
    tmp = []
    for line in text:
        if len(line) < 3:
            continue
        tmp.append(line.split("\t"))
    id_concepts = np.array(tmp)
    ids = id_concepts[:,0]
    concepts = id_concepts[:,1]
    return dict(zip(ids, concepts))

train_text = open(concepts_path, encoding="utf8")
if not train_concepts_proceeded:
    ids_concepts_dict = find_ids_concepts_dict(train_text)
    joblib.dump(ids_concepts_dict, 'doc_ids_concepts_dict.pkl')
else:
    ids_concepts_dict = joblib.load('doc_ids_concepts_dict.pkl')

num_to_process = len(ids_concepts_dict)
[ids, concepts] = [list(ids_concepts_dict.keys()), list(ids_concepts_dict.values())]

val_text = open(val_concepts_path, encoding="utf8")
if not val_concepts_proceeded:
    val_ids_concepts_dict = find_ids_concepts_dict(val_text)
    joblib.dump(val_ids_concepts_dict, 'doc_val_ids_concepts_dict.pkl')
else:
    val_ids_concepts_dict = joblib.load('doc_val_ids_concepts_dict.pkl')

[val_ids, val_concepts] = [list(val_ids_concepts_dict.keys()), list(val_ids_concepts_dict.values())]

print('concepts and ids lists completed')
print("--- %.2f seconds ---" % (time.time() - start_time))
print()

################################################################
# make concept matrices
print('make concept matrices')
print()
start_time = time.time()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def tokenize_concepts(concepts_string):
    concepts_string = concepts_string.replace('\n', '')
    concepts_list = concepts_string.split(',')
    return concepts_list

unique_concepts_set = set()
for concept in concepts:
    unique_concepts_set.update(concept.replace('\n', '').split(','))

print(str(len(unique_concepts_set)) + ' unique concepts')

# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=300000,
                                   min_df=min_df, use_idf=True, tokenizer=tokenize_concepts)

if not tfidf_lsi_computed:
    print('find tf-idf matrix')
    tfidf_vectorizer = tfidf_vectorizer.fit(concepts)
    tfidf_matrix = tfidf_vectorizer.transform(concepts)  # fit the vectorizer to concepts
    print('find lsi matrix')
    print()
    svd = TruncatedSVD(n_components=num_topics)
    svd = svd.fit(tfidf_matrix)
    lsi_matrix = svd.transform(tfidf_matrix)
    joblib.dump(svd,  'doc_svd.pkl')
    joblib.dump(tfidf_vectorizer, 'doc_tfidf_vectorizer.pkl')
    joblib.dump(tfidf_matrix,  'doc_tfidf_matrix.pkl')
    joblib.dump(lsi_matrix,  'doc_lsi_matrix.pkl')
else:
    svd = joblib.load('doc_svd.pkl')
    tfidf_vectorizer = joblib.load('doc_tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('doc_tfidf_matrix.pkl')
    lsi_matrix = joblib.load('doc_lsi_matrix.pkl')

print('shape of tf-idf matrix:')
print(tfidf_matrix.shape)
print()
terms = tfidf_vectorizer.get_feature_names()

print('shape of lsi matrix:')
print(lsi_matrix.shape)
print()

words_matrix = lsi_matrix

print('concept matrices completed')
print("--- %.2f seconds ---" % (time.time() - start_time))
print()

#########################################################
# takes the path of an image and returns an 1D np.array of its lbp values (for 12 points it is 352 length)
def image_to_lbp(image_path):
    pic = Image.open(image_path).convert('L')
    pic.load()
    data = np.asarray(pic)
    return mahotas.features.lbp(data, 16, 12)

import cv2
# takes the path of an image and returns an 1D np.array of its SIFT values (128 length)
def image_to_SIFT(image_path):
    pic = cv2.imread(image_path)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    sift =cv2.xfeatures2d.
    pic = Image.open(image_path).convert('L')
    pic.load()
    data = np.asarray(pic)
    return mahotas.features.lbp(data, 16, 12)


def find_forest(images_path):
    # forest = GradientBoostingClassifier(n_estimators=100)
    forest = RandomForestClassifier(n_estimators=150)
    # images = np.empty((num_to_process, im_width*im_height))  # for image pixels (image_to_array)
    images = np.empty((num_to_process, 352))   # for image lbp (image_to_array2)
    i = 0
    for image in os.listdir(images_path):
        clusters_list.append(ids_clusters_dict[image[:-4]])
        images[i] = image_to_array2(os.path.join(images_path, image))
        i += 1
    return forest.fit(images,clusters_list)


def find_cluster_ids():
    cluster_ids = []  # list of sets: each set has the names of the images of the corresponding cluster
    for i in range(num_clusters):
        cluster_ids.append(list())
    for image in os.listdir(images_path):
        clusters_list.append(ids_clusters_dict[image[:-4]])
        pix = image_to_array2(os.path.join(images_path, image))
        pix = pix.reshape(1, -1)
        this_cluster = forest.predict(pix)[0]
        cluster_ids[this_cluster].append(image[:-4])
    return cluster_ids

if not forest_found:
    forest = find_forest(images_path)  # fits the forest to the clusters
    joblib.dump(forest, 'doc_forest.pkl')
    cluster_ids = find_cluster_ids()
    joblib.dump(cluster_ids, 'doc_cluster_ids.pkl')
else:
    forest = joblib.load('doc_forest.pkl')
    cluster_ids = joblib.load('doc_cluster_ids.pkl')

print('random forest completed')
print("--- %.2f seconds ---" % (time.time() - start_time))
print()
#########################################################
# data fit
print('data fit started...')
start_time = time.time()

from sklearn.metrics.pairwise import linear_kernel

num_of_related_words = 6


def find_similar_image(test_image):
    pix = image_to_array2(test_image)
    pix = pix.reshape(1, -1)
    this_cluster = forest.predict(pix)[0]
    tmp_ssim = []
    for image in cluster_ids[this_cluster]:
        compare_image = os.path.join(images_path, image + '.jpg')
        tmp_ssim.append(ssim.compute_ssim(test_image, compare_image))
    index = tmp_ssim.index(max(tmp_ssim))
    return cluster_ids[this_cluster].__getitem__(index)


def find_similar_images(test_image):
    pix = image_to_array2(test_image)
    pix = pix.reshape(1, -1)
    this_cluster = forest.predict(pix)[0]
    tmp_ssim = []
    for image in cluster_ids[this_cluster]:
        val_image_path = os.path.join(val_images_path, val_image)
        compare_image_path = os.path.join(images_path, image + '.jpg')
        tmp_ssim.append(ssim.compute_ssim(val_image_path, compare_image_path))
    indices = heapq.nlargest(num_to_compare, range(len(tmp_ssim)), tmp_ssim.__getitem__)
    tmp_image_ids = []
    weights = []
    for ind in indices:
        tmp_image_ids.append(cluster_ids[this_cluster].__getitem__(ind))
        weights.append(tmp_ssim.__getitem__(ind))
    return tmp_image_ids, weights


def find_related_words(related_ids, weights, num_of_related_words):
    weights = np.asarray(weights)
    num_of_related = len(related_ids)
    tfidf_rows = np.empty((num_of_related, tfidf_matrix.shape[1]))
    related_words = []
    for i in range(num_of_related):
        ind = ids.index(related_ids[i])
        tfidf_rows[i] = tfidf_matrix.getrow(ind).todense()
    centroid_coords = np.dot(weights, tfidf_rows) / sum(weights)
    order_centroid = centroid_coords.argsort()[::-1]
    for ind in order_centroid[:num_of_related_words]:
        related_words.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
    return related_words


def find_related_words_lsi(related_ids, weights, num_of_related_words):
    weights = np.asarray(weights)
    num_of_related = len(related_ids)
    lsi_rows = np.empty((num_of_related, lsi_matrix.shape[1]))
    related_words = []
    for i in range(num_of_related):
        ind = ids.index(related_ids[i])
        lsi_rows[i] = lsi_matrix[ind]
    centroid_coords_lsi = np.dot(weights, lsi_rows) / sum(weights)
    centroid_coords_tfidf  = svd.inverse_transform(centroid_coords_lsi)
    order_centroid = centroid_coords_tfidf.argsort()[::-1]
    # find related words
    for ind in order_centroid[0,:num_of_related_words]:
        related_words.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
    # find related description
    related_tfidf_matrix = np.empty([num_of_related, tfidf_matrix.shape[1]])
    for i in range(num_of_related):
        ind = ids.index(related_ids[i])
        related_tfidf_matrix[i] = tfidf_matrix.getrow(ind).todense()
    cosine_similarities = linear_kernel(order_centroid, related_tfidf_matrix).flatten()
    index = np.argmax(cosine_similarities)
    related_concept = concepts[index]
    return related_words, related_concept

val_images_concepts_predictions = []
total_bleu = 0
val_num = 0
for val_image in os.listdir(val_images_path):
    image_path = os.path.join(val_images_path, val_image)
    similar_images, weights = find_similar_images(image_path)
    # related_words = find_related_words(similar_images, weights, num_of_related_words)
    related_words, predicted_concept = find_related_words_lsi(similar_images, weights, num_of_related_words)
    similar_image = similar_images[0]
    # similar_image = find_similar_image(image_path)
    # predicted_concept = ids_concepts_dict[similar_image]
    predicted_concept_list = tokenize_only(predicted_concept)
    actual_concept = val_ids_concepts_dict[val_image[:-4]]
    actual_concept_list = tokenize_only(actual_concept)
    val_images_concepts_predictions.append(predicted_concept)
    print(similar_image + ": " + predicted_concept)
    print('actual concept: ' + actual_concept)
    print('predicted key concepts: ' + str(related_words))
    bleu_score = nltk.translate.bleu_score.sentence_bleu(actual_concept_list, predicted_concept_list, weights=[1])
    total_bleu += bleu_score
    val_num += 1
    print('BLEU score: ' + str(bleu_score))
print('Mean BLEU: ' + str(total_bleu / val_num))

print('data fit completed')
print("--- %.2f seconds ---" % (time.time() - start_time))
print()