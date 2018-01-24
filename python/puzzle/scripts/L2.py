import numpy as np

from puzzle.tools.crop import crop
from puzzle.tools.utils import input_image


def l2_distance(img, template):
    """
    :param img: original image
    :param template: cropped section of the image
    :return: l2 distance between images
    """
    distances = np.sqrt(np.sum(np.square(img - template), axis=1))

    return distances.sum()


img_path = input_image("Input an image name from '{}': \n")
print('Using image src : ' + img_path)

pieces = crop(img_path, (16, 16))

l2_list = []
l2_names = []

for i in range(len(pieces)):
    for k in range(len(pieces)):
        distance = l2_distance(pieces[i], pieces[k])
        if distance == 0:
            l2_names.append((i, k))
        l2_list.append(distance)

# return the pairs of pieces which were exact matches
print(l2_names)
