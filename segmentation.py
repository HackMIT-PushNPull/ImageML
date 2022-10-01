import keras.models
import cv2

import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/test_img4.png')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

preprocessed_digits = []

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x,y), (x+w, y+h), color = (0, 255, 0), thickness=2)
    digit = thresh[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit, (18,18))
    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    preprocessed_digits.append(padded_digit)

plt.imshow(image, cmap="gray")
plt.show()
inp = np.array(preprocessed_digits)

loaded_model = keras.models.load_model('머신 러닝 이름')

for digit in preprocessed_digits:
    prediction = loaded_model.predict(digit.reshape(1, 28, 28, 1))

    plt.imshow(digit.reshape(28, 28), cmap="gray")
    plt.show()
    print("Result: " + format(np.argmax(prediction)))
