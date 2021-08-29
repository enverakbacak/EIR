#coding=utf-
from tensorflow.keras import Input, Model, Sequential
#from tensorflow.keras.models import Sequential,Input,Model,InputLayer
from tensorflow.keras.models import model_from_json, load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)

json_file = open('./models/eih_16_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("./models/eih_16_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)

X = np.load('preprocessed_test_images.npy')
X.shape

features = model.predict(X, batch_size=64, verbose=0, steps=None) #     TRY THIS ALSO XX = model.predict(X, batch_size)
features = features.astype(float)
np.savetxt('hashCodes/features_test_images_16.txt',features, fmt='%f')
hashCodes = features > 0
hashCodes = hashCodes.astype(int)
np.savetxt('hashCodes/hashCodes_test_images_16.txt',hashCodes, fmt='%d')


