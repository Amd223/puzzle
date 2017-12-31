import math
import multiprocessing
from collections import OrderedDict

import cv2
import numpy as np
from matplotlib import pyplot as plt

from puzzle.tools.utils import input_image, img_read
from puzzle.tools.crop import crop_interactive


# All the 6 methods for comparison in a list
METHODS_MAX = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']
METHODS_MIN = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
METHODS = METHODS_MAX + METHODS_MIN


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

    # Start multi-threaded jobs
    manager = multiprocessing.Manager()
    mse = manager.dict()
    jobs = []
    for m, r in results.items():
        p = multiprocessing.Process(target=do_single_mse, args=(m, r, mse))
        jobs.append(p)
        p.start()

    # Wait for completion
    for p in jobs:
        p.join()

    # Sort by ascending error
    mse = OrderedDict(sorted(mse.items(), key=lambda x: x[1]))
    for m, e in mse.items():
        print('Method {0: <20}: {1}'.format(m, e))

    return list(mse.items())[0][0]


def test_methods(img, template, template_pos, draw=True):
    print('[{0: <20}] templ loc = {1}'.format("", template_pos))

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

        # draw box on template
        if draw:
            w, h = template.shape[::-1]
            bottom_right = (match_loc[0] + w, match_loc[1] + h)
            cv2.rectangle(img, match_loc, bottom_right, 255, 2)

            plt.clf()
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.show()

    best_method = compute_mse(results, template_pos)
    print('Best method is {}.'.format(best_method))


img_path = input_image("Input an image name from '{}': \n")
print('Using image src : ' + img_path)

template_path, template_pos = crop_interactive(img_path, show_crop=False)

# Load images
img = img_read(img_path)
template = img_read(template_path)

test_methods(img, template, template_pos, draw=False)


# apply template matching using SIFT - xfeatures does not work
# image = cv2.imread("images/adam.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# image = cv2.drawKeypoints(gray, kp, image)
# cv2.imwrite('sift_keypoints.jpg', image)