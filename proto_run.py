import keras.models
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import PIL.Image as pil_image
import Segmentation

loaded_model = keras.models.load_model('proto2')
#results = Segmentation.segmentation('Images/test_img1.png')


image = keras.preprocessing.image.load_img('Images/test_img_lambda.jpg', target_size=(45, 45))
image = keras.preprocessing.image.img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = keras.applications.vgg19.preprocess_input(image)

pred = loaded_model.predict(image)
print(pred)
