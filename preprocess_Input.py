import cv2
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('test_images.csv')     # reading the csv file

resize=224
X = [ ]     # creating an empty array
for myFile in data.Image_ID:
    image = plt.imread('test_images/' + myFile) 
    image = cv2.resize(image,(resize, resize))
    X.append (image)
X = np.array(X)    # converting list to array



base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(resize, resize, 3))


batch_size=64
X = base_model.predict(X, batch_size=batch_size, verbose=0, steps=None)
np.save('preprocessed_test_images.npy', X)
