import cv2
import numpy as np

from puzzle.tools.utils import input_image, img_read, input_directory



def get_histogram(img_piece):

    img_piece = np.array(img_piece)
    hist = cv2.calcHist([img_piece], None, [256], [0, 256])    #[0] indicates we are passing a grayscale image
                                                                    # None that we are passing the entire image                                                               # [256] size of the bins
    return hist

def compare_histograms(piece1, piece2, method=cv2.HISTCMP_CORREL):

    hist1 = get_histogram(piece1)
    hist2 = get_histogram(piece2)

    compare = cv2.compareHist(hist1, hist2, method)

    return compare


if __name__ == "__main__":
    img_path1 = input_directory()
    img_path2 = input_directory()
    print(compare_histograms(img_path1, img_path2, method=cv2.HISTCMP_CORREL))


# OPENCV_METHODS = (
#     ("Correlation", cv2.cv.CV_COMP_CORREL),
#     ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
#     ("Intersection", cv2.cv.CV_COMP_INTERSECT),
#     ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))
#
# # loop over the comparison methods
# for (methodName, method) in OPENCV_METHODS:
#     # initialize the results dictionary and the sort
#     # direction
#     results = {}
#     reverse = False
#
#     # if we are using the correlation or intersection
#     # method, then sort the results in reverse order
#     if methodName in ("Correlation", "Intersection"):
#         reverse = True
#
# for (k, hist) in index.items():
#     # compute the distance between the two histograms
#     # using the method and update the results dictionary
#     d = cv2.compareHist(index["doge.png"], hist, method)
#     results[k] = d
#
#     # sort the results
# results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)