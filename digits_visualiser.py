import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("./digit1.jpg")
plt.imshow(img*255, cmap="viridis")
plt.show()