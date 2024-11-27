import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import numpy as np
import keras
from tqdm.notebook import tqdm
from keras.applications import Xception
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout


import os
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception, preprocess_input

image_model = Xception(weights='imagenet', include_top=False, pooling='avg')

featuresx = {}

directory = '/content/Images'


for img_name in tqdm(os.listdir(directory)):
    # Load the image from file
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(299, 299))  # Xception requires (299, 299) input shape

    # Convert image pixels to numpy array
    image = img_to_array(image)


    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


    image = preprocess_input(image)


    feature = image_model.predict(image, verbose=0)


    image_id = img_name.split('.')[0]

    featuresx[image_id] = feature


captions_directory = image_directory
with open(os.path.join(captions_directory, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()




'''

IN PROGRESS

'''


# Use LSTM cells 

