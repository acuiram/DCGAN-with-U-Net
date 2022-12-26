import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import time

total_starttime = time.time()

generator = load_model(r'C:\Licenta\GAN_IMAGES\model_16_batch\generator_model_99.h5')

damaged_directory = r"C:\Licenta\GAN_IMAGES\damaged\test"
damaged_dataset = tf.keras.utils.image_dataset_from_directory(damaged_directory,
                                                                  label_mode=None,
                                                                  batch_size=1,
                                                                  image_size=(256, 256),
                                                                  shuffle=False)

### Define a normalization layer ###
normalization_layer = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)

### Normalize the datasets ###
damaged_dataset = damaged_dataset.map(lambda x: normalization_layer(x))

i = 0
for img in (damaged_dataset):
    generated_image = generator(img)
    generated_image = (generated_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)
    tf.keras.preprocessing.image.save_img(r"C:\Licenta\GAN_IMAGES\results_GAN_bs32\16_batch_size_100_epochs/" + str(i) + ".jpg", generated_image)
    i = i + 1

total_endtime = time.time()
print(f'Total execution time: {total_endtime - total_starttime} s')
