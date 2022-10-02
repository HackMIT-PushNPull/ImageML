import matplotlib.pyplot as plt
from PIL import Image
import imageio

import cv2
import numpy as np


def to_one_hot(index):
    results = np.zeros(82)
    results[index] = 1
    print(results)


def segmentation(img_location):
    image_file_location = img_location

    image = cv2.imread(image_file_location)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        digit = thresh[y:y + h, x:x + w]
        new_character = pad_by_size(digit)
        preprocessed_digits.append(new_character)

    plt.imshow(image, cmap="gray")
    plt.show()
    inp = np.array(preprocessed_digits)

    # 학습된 모델 불러 오기
    # loaded_model = keras.models.load_model('trained_ml')

    for digit in preprocessed_digits:
        plt.imshow(digit.reshape(45, 45))
        plt.show()

    return preprocessed_digits


def pad_by_size(character):
    height = len(character)
    width = len(character[0])

    new_character = character

    # Resize character if either width or height is larger than 45
    if height > 45 or width > 45:
        if height > 45 > width:
            new_character = cv2.resize(character, (width, 45))
            height = 45
        if height < 45 < width:
            new_character = cv2.resize(character, (45, height))
            width = 45
        else:
            new_character = cv2.resize(character, (45, 45))
            width = 45
            height = 45

    # Pad remaining area
    width_pad = 45 - width
    r_width_pad = width_pad
    height_pad = 45 - height
    b_height_pad = height_pad

    if width_pad % 2 == 1:
        width_pad = (width_pad - 1) / 2
        r_width_pad = width_pad + 1
    else:
        width_pad = width_pad / 2
        r_width_pad = r_width_pad / 2

    if height_pad % 2 == 1:
        height_pad = (height_pad - 1) / 2
        b_height_pad = height_pad + 1
    else:
        height_pad = height_pad / 2
        b_height_pad = b_height_pad / 2

    new_character = np.pad(new_character, (
        (int(height_pad), int(b_height_pad)),
        (int(width_pad), int(r_width_pad))),
                           "constant", constant_values=0)
    return new_character



