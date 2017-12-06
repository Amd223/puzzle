import cv2
import numpy as np
from matplotlib import pyplot as plt

img = input("Input an image: \n")
template = img + "2"
img = img + ".jpg"
template = template + ".jpg"

# Load images
img = cv2.imread("images/" + img, 0)
template = cv2.imread("images/" + template, 0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# apply template matching by sliding template over original image
# performs well as long as the template is a direct crop from the image
res = cv2.matchTemplate(template, img, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# draw box on template
cv2.rectangle(img, top_left, bottom_right, 255, 2)

plt.subplot(122), plt.imshow(img, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.show()

image = cv2.imread("images/adam.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
image = cv2.drawKeypoints(gray, kp, image)
cv2.imwrite('sift_keypoints.jpg', image)