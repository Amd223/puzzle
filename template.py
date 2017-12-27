import os
import math
import multiprocessing
from collections import OrderedDict

import cv2
import numpy as np
from matplotlib import pyplot as plt


# All the 6 methods for comparison in a list
METHODS_MAX = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']
METHODS_MIN = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
METHODS = METHODS_MAX + METHODS_MIN

# Target location of template
TARGET_LOC = (430, 406)


def safe_imread(img_name):
    img_path = os.path.join('images', img_name)
    img = cv2.imread(img_path, 0)
    if img is None:
        raise ValueError("No image at '{}'".format(img_name))
    return img

def compute_mse(results, target_loc, threshold=0.95):
    """
    Compute the mean squared error
    :return: str, name of method minimising the MSE
    """
    target_y, target_x = target_loc

    def do_single_mse(m, r, mse):
        e = []
        it = np.nditer(r, flags=['multi_index'])
        while not it.finished:
            v = it[0]
            y, x = it.multi_index
            if v >= threshold:
                e.append(math.sqrt((y-target_y)**2 + (x-target_x)**2))
            it.iternext()
        mse[m] = sum(e) / len(e)

    # Start multithreaded jobs
    manager = multiprocessing.Manager()
    mse = manager.dict()
    jobs = []
    for m, r in results.items():
        p = multiprocessing.Process(target=do_single_mse, args=(m, r, mse))
        jobs.append(p)
        p.start()

    # Wait for completion
    for proc in jobs:
        proc.join()

    # Sort by ascending error
    mse = OrderedDict(sorted(mse.items(), key=lambda x: x[1]))
    for m, e in mse.items():
        print('Method {0: <20}: {1}'.format(m, e))

    return list(mse.items())[0][0]


def apply_methods(draw=True):
    results = {}

    for m in METHODS:
        # apply template matching by sliding template over original image
        # performs well as long as the template is a direct crop from the image
        res = cv2.matchTemplate(template, img, eval(m))
        cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        results[m] = res

        # Top left corner
        match_loc = max_loc if m in METHODS_MAX else min_loc
        print('[{0: <20}] match loc = {1}'.format(m, match_loc))
        bottom_right = (match_loc[0] + w, match_loc[1] + h)

        # draw box on template
        if draw:
            cv2.rectangle(img, match_loc, bottom_right, 255, 2)

            plt.clf()
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.show()

    best_method = compute_mse(results, TARGET_LOC)
    print('Best method is {}.'.format(best_method))


img_name = input("Input an image name: \n")
img      = '{}.jpg'.format(img_name)
template = '{}3.jpg'.format(img_name)
print('Using image src ' + img)
print('Using image template ' + template)

# Load images
img      = safe_imread(img)
template = safe_imread(template)
w, h = template.shape[::-1]

apply_methods(draw=False)


# apply template matching using SIFT - xfeatures does not work
# image = cv2.imread("images/adam.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# image = cv2.drawKeypoints(gray, kp, image)
# cv2.imwrite('sift_keypoints.jpg', image)


