import keras.models
import matplotlib.pyplot as plt

import cv2
import numpy as np

def segmentation(image_file_location):
    image_file_location = 'Images/test_img4.png'
    image = cv2.imread(image_file_location)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        digit = thresh[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (35, 35))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        preprocessed_digits.append(padded_digit)

    plt.imshow(image, cmap="gray")
    plt.show()
    inp = np.array(preprocessed_digits)

    # 학습된 모델 불러 오기
    # loaded_model = keras.models.load_model('trained_ml')

    for digit in preprocessed_digits:
        plt.imshow(digit.reshape(45, 45), cmap="gray")
        plt.show()

    return preprocessed_digits

segmentation('')
