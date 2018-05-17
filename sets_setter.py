import numpy as np
import os
import shutil
from shutil import copyfile

captions_path = 'E:\Downloads\ConceptDetectionTraining2017-Concepts.txt'
images_path = 'E:\Downloads\ConceptDetectionTraining2017\CaptionPredictionTraining2017\\'
val_captions_path = 'E:\Downloads\ConceptDetectionValidation2017-Concepts.txt'
val_images_path = 'E:\Downloads\ConceptDetectionValidation2017\CaptionPredictionValidation2017\\'
tmp_captions_path = 'E:\Downloads\\tmp_concepts.txt'
tmp_val_captions_path = 'E:\Downloads\\tmp_val_concepts.txt'
tmp_images_path = 'E:\Downloads\\tmp_concepts_images_path\\'
tmp_val_images_path = 'E:\Downloads\\tmp_val_concepts_images_path\\'
num_to_process = 753
num_to_pass = 0
num_to_validate = 218

text = open(captions_path, encoding="utf8")
tmp = []
line_num = 1
f = open(tmp_captions_path, 'wb')

for line in text:
    f.write(line.encode('utf8'))
    # f.write('\r\n'.encode('utf8'))
    tmp.append(line.split("\t"))
    if line_num >= num_to_process:
        break
    line_num += 1
f.close()

id_captions = np.array(tmp)
ids = id_captions[:,0]
# captions = id_captions[:,1]


if os.path.exists(tmp_images_path):
    shutil.rmtree(tmp_images_path)

os.makedirs(tmp_images_path)

for ide in ids:
    src = images_path + ide + '.jpg'
    dst = tmp_images_path + ide + '.jpg'
    copyfile(src, dst)

###############################
# validation


text = open(val_captions_path, encoding="utf8")
tmp = []
line_num = 1
f = open(tmp_val_captions_path, 'wb')

tmp_to_pass = 0
for line in text:
    if tmp_to_pass <= num_to_pass:
        tmp_to_pass += 1
        continue
    f.write(line.encode('utf8'))
    # f.write('\r\n'.encode('utf8'))
    tmp.append(line.split("\t"))
    if line_num >= num_to_validate:
        break
    line_num += 1
f.close()

id_captions = np.array(tmp)
ids = id_captions[:,0]
# captions = id_captions[:,1]


if os.path.exists(tmp_val_images_path):
    shutil.rmtree(tmp_val_images_path)

os.makedirs(tmp_val_images_path)

for ide in ids:
    src = val_images_path + ide + '.jpg'
    dst = tmp_val_images_path + ide + '.jpg'
    copyfile(src, dst)