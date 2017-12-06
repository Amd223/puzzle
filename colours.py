import cv2
import numpy as np
from matplotlib import pyplot as plt

image = input("Input an image: \n")
image = image + ".jpg"
img = cv2.imread("images/"+image)

hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.show()