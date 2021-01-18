import os
import warnings
warnings.filterwarnings("ignore")

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
import sigol

##### Load Image #####
label = label_path
model = load_model(saved_model_path)
detector = detector_path
predictor = predictor_path

folder_path = folder_path
folder_list = os.listdir(folder_path)
folder_list.sort()
print(len(folder_list))

data = []
df = pd.DataFrame(columns=['filename' ,'ratio', 'r', 'g', 'b', 'breed'])
for f in folder_list:
    file_path = folder_path + '/' + f
    input = file_path
    try:
      data = sigol(input, label, model, detector, predictor) # 여기서 df 한 행이 만들어짐
      df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
    except ValueError as e:
      pass

# Test Image
input = input_image

print('-----------Test Image-------------')
search_data = sigol(input, label, model, detector, predictor)

img = cv2.imread(input)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5,5))
plt.imshow(img)

print('-----------Predict Image-------------')
df = pd.read_csv(path)

d_list = []
r2 = search_data[2]
g2 = search_data[3]
b2 = search_data[4]

for filename,r,g,b in zip(df['filename'], df['r'], df['g'], df['b']):
  r1 = round(r, 3)
  g1 = round(g, 3)
  b1 = round(b, 3)
  d = round(math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2), 3)
  d_list.append(d)

d_min = min(d_list)
# print(pred)
index = d_list.index(d_min)
pred = df['filename'][index]
posted_img_path = path + pred + '.jpg'
print(posted_img_path)
img = cv2.imread(posted_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5,5))
plt.imshow(img)

