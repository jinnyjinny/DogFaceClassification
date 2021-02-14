import os
import warnings

warnings.filterwarnings("ignore")
import plaidml.keras

plaidml.keras.install_backend()

import tensorflow as tf
import keras
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import csv
import dlib, cv2
import math
from imutils import face_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import xml.etree.ElementTree as ET

import numpy as np

np.seterr(over='ignore')
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import keras
from keras.models import *
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.utils import np_utils
from keras.callbacks import *

from keras.applications.densenet import DenseNet121, preprocess_input


def csv_sigor(input, label, model, detector, predictor):
    data = []
    ratio = []

    ##### Breed Predict #####
    breed_list = os.listdir(label)
    num_classes = len(breed_list)
    # print("{} breeds".format(num_classes))

    n_total_images = 0
    for breed in breed_list:
        n_total_images += len(os.listdir(label + "/{}".format(breed)))
    # print("{} images".format(n_total_images))

    label_maps = {}
    label_maps_rev = {}
    for i, v in enumerate(breed_list):
        label_maps.update({v: i})
        label_maps_rev.update({i: v})

        ##### Load Image #####
    img = cv2.imread(input)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print("img: ", img.shape)

    ##### Show image #####
    # plt.figure(figsize=(4, 4))
    # plt.imshow(breed_img)
    # plt.axis('off')

    ##### Predict image #####
    breed_img = imread(input)
    # breed_img = resize(breed_img, (224, 224))

    # print("breed_img: ", breed_img.shape)
    filename = os.path.splitext(os.path.basename(input))[0]
    filename = filename[1:-1]  # 양쪽 큰따옴표 제거
    filename = filename.replace(".jpg", ".null")  # 주소변경
    url_http = filename[0:5]
    url_colo = filename[6]
    url_name = '//' + filename[8:]
    url_name = url_name.replace(":", "/")
    filename = url_http + url_colo + url_name
    breed_img = preprocess_input(breed_img)
    probs = model.predict(np.expand_dims(breed_img, axis=0))

    for idx in probs.argsort()[0][::-1][:1]:
        breed = label_maps_rev[idx].split("-")[-1]

    ##### Load Model #####
    detector = dlib.cnn_face_detection_model_v1(detector)
    predictor = dlib.shape_predictor(predictor)

    ##### Detect Face #####
    try:
        dets = detector(img, upsample_num_times=2)
    except RuntimeWarning as e:
        pass
    img_result = img.copy()

    for i, d in enumerate(dets):
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255, 0, 0), lineType=cv2.LINE_AA)

    ##### Detect Landmarks #####
    for i, d in enumerate(dets):
        shape = predictor(img, d.rect)  # detect in range of d.rect
        shape = face_utils.shape_to_np(shape)
        shape = shape.reshape(-1)  # x0 y0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        ##### Landmarks Ratio #####
        x3 = int(shape[6])
        y3 = int(shape[7])
        x5 = int(shape[10])
        y5 = int(shape[11])
        x2 = int(shape[4])
        y2 = int(shape[5])
        area = abs((x5 - x3) * (y2 - y3) - (y5 - y3) * (x2 - x3))
        AB = ((x5 - x2) ** 2 + (y5 - y2) ** 2) ** 0.5
        ratio = (area / AB) * 0.01
        ratio = round(ratio, 3)

    ##### RGB Pixel #####
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    k = 5
    clt = KMeans(n_clusters=k)
    clt.fit(img)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    dictionary = {}
    for (percent, color) in zip(hist, clt.cluster_centers_):
        startX = 0
        endX = startX + (percent * 300)
        block = endX - startX
        dictionary[block] = color.astype("uint8")
        startX = endX

    rgb_max = max(dictionary.keys())
    rgb_max = dictionary[rgb_max]
    r = rgb_max[0]
    g = rgb_max[1]
    b = rgb_max[2]

    R = round(r / (r + g + b), 3)
    G = round(g / (r + g + b), 3)
    B = round(b / (r + g + b), 3)

    data = [filename, ratio, R, G, B, breed]
    print(data)
    return data


def sigor(input, label, model, detector, predictor):
    data = []
    ratio = []

    ##### Breed Predict #####
    breed_list = os.listdir(label)
    num_classes = len(breed_list)
    # print("{} breeds".format(num_classes))

    n_total_images = 0
    for breed in breed_list:
        n_total_images += len(os.listdir(label + "/{}".format(breed)))
    # print("{} images".format(n_total_images))

    label_maps = {}
    label_maps_rev = {}
    for i, v in enumerate(breed_list):
        label_maps.update({v: i})
        label_maps_rev.update({i: v})

        ##### Load Image #####
    img = cv2.imread(input)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print("img: ", img.shape)

    ##### Show image #####
    # plt.figure(figsize=(4, 4))
    # plt.imshow(breed_img)
    # plt.axis('off')

    ##### Predict image #####
    breed_img = imread(input)
    # breed_img = resize(breed_img, (224, 224))

    # print("breed_img: ", breed_img.shape)
    filename = os.path.splitext(os.path.basename(input))[0]
    breed_img = preprocess_input(breed_img)
    probs = model.predict(np.expand_dims(breed_img, axis=0))

    for idx in probs.argsort()[0][::-1][:1]:
        breed = label_maps_rev[idx].split("-")[-1]

    ##### Load Model #####
    detector = dlib.cnn_face_detection_model_v1(detector)
    predictor = dlib.shape_predictor(predictor)

    ##### Detect Face #####
    try:
        dets = detector(img, upsample_num_times=2)
    except RuntimeWarning as e:
        pass
    img_result = img.copy()

    for i, d in enumerate(dets):
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255, 0, 0), lineType=cv2.LINE_AA)

    ##### Detect Landmarks #####
    for i, d in enumerate(dets):
        shape = predictor(img, d.rect)  # detect in range of d.rect
        shape = face_utils.shape_to_np(shape)
        shape = shape.reshape(-1)  # x0 y0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        ##### Landmarks Ratio #####
        x3 = int(shape[6])
        y3 = int(shape[7])
        x5 = int(shape[10])
        y5 = int(shape[11])
        x2 = int(shape[4])
        y2 = int(shape[5])
        area = abs((x5 - x3) * (y2 - y3) - (y5 - y3) * (x2 - x3))
        AB = ((x5 - x2) ** 2 + (y5 - y2) ** 2) ** 0.5
        ratio = (area / AB) * 0.01
        ratio = round(ratio, 3)

    ##### RGB Pixel #####
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    k = 5
    clt = KMeans(n_clusters=k)
    clt.fit(img)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    dictionary = {}
    for (percent, color) in zip(hist, clt.cluster_centers_):
        startX = 0
        endX = startX + (percent * 300)
        block = endX - startX
        dictionary[block] = color.astype("uint8")
        startX = endX

    rgb_max = max(dictionary.keys())
    rgb_max = dictionary[rgb_max]
    r = rgb_max[0]
    g = rgb_max[1]
    b = rgb_max[2]

    R = round(r / (r + g + b), 3)
    G = round(g / (r + g + b), 3)
    B = round(b / (r + g + b), 3)

    data = [filename, ratio, R, G, B, breed]
    print(data)
    return data


label = '/Users/hyojin/stanford-dogs-dataset/images/Images'
model = tf.keras.models.load_model('/Users/hyojin/saved_model/my_model')
detector = '/Users/hyojin/lib/dogHeadDetector.dat'
predictor = '/Users/hyojin/lib/landmarkDetector.dat'

'''
# make csv file
folder_path = '/Users/hyojin/dog'
folder_list = os.listdir(folder_path)
folder_list.sort()
print(len(folder_list))

data = []
df = pd.DataFrame(columns=['filename' ,'ratio', 'r', 'g', 'b', 'breed'])
for f in folder_list:
    file_path = folder_path + '/' + f
    input = file_path
    try:
      fs = open('/Users/hyojin/url_Sigor.csv', 'a', newline='')
      wr = csv.writer(fs)
      data = csv_sigor(input, label, model, detector, predictor) # 여기서 df 한 행이 만들어짐
      wr.writerow(data)
      # df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
    except ValueError as e:
      pass    
# df.to_csv('/Users/hyojin/url_Sigor.csv')
# print(df.head())
'''

# Test Input Image
input = '/Users/hyojin/dingo.jpeg'

print('-----------Test Image-------------')
search_data = sigor(input, label, model, detector, predictor)

img = cv2.imread(input)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.imshow(img)

print('-----------Predict Image-------------')
# df = pd.DataFrame(columns=['filename', 'ratio', 'r', 'g', 'b', 'breed'])
df = pd.read_csv('/Users/hyojin/url_Sigor.csv', sep=',', names=['filename', 'ratio', 'r', 'g', 'b', 'breed'],
                 header=None)
print(df.head)

ratio2 = search_data[1]
r2 = search_data[2]
g2 = search_data[3]
b2 = search_data[4]
breed2 = search_data[5]

# 종이 같은 애들 파일 이름
breedlist = []

for filename, breed in zip(df['filename'], df['breed']):
    if breed == breed2:
        breedlist.append(filename)

# 삼각존 차이 0.05일 애들 파일 이름
ratiolist = []

for filename, ratio in zip(df['filename'], df['ratio']):
    if ratio != "[]":
        if ratio2 + 0.05 >= float(ratio):
            ratiolist.append(filename)

d_list = []

for filename, r, g, b in zip(df['filename'], df['r'], df['g'], df['b']):
    r1 = round(r, 3)
    g1 = round(g, 3)
    b1 = round(b, 3)
    d = round(math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2), 3)
    d_list.append(d)

d_min = min(d_list)


def filtering(x):
    return x <= d_min + 5


rgblist = list(filter(filtering, d_list))
len_num = len(rgblist)

# 종 파일 리스트와 삼각존 파일 리스트 겹치는 파일이름 추출
breedandratio = list(set(breedlist).intersection(ratiolist))
br_num = len(breedandratio)

print(breedandratio)

# rgb 거리차 + 5인 애들 파일 이름
rgbrgb = []

for i in range(len_num):
    d_index = d_list.index(rgblist[i])
    rgb_index = df['filename'][d_index]
    rgbrgb.append(rgb_index)
print(rgbrgb)

# 위의 겹치는 파일이름과 모색 겹치는 파일 이름 중복 추출
breedratiorgb = list(set(breedandratio).intersection(rgbrgb))
brg_num = len(breedratiorgb)


# for i in range(brg_num) :
#     path = '/Users/hyojin/dog2'
#     posted_img_path = path + breedratiorgb[i] + '.jpg'
#     img = cv2.imread(posted_img_path)
#     if img is not None:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.figure(figsize=(5,5))
#         plt.imshow(img)

# 리스트를 str로 return 함수
def pic():
    out_img = ','.join(breedratiorgb)
    return out_img
