import cv2
from matplotlib import pyplot as plt

from puzzle.tools.utils import input_image, img_read


img = img_read(input_image())

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()