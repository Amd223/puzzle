import numpy as np

from puzzle.tools.utils import input_image, img_read
from puzzle.tools.crop import crop

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

pieces = crop(img_path,(16,16))

l2_list = []
l2_names = []
k = 0
i = 0

while i < len(pieces):
    while k < len(pieces):
        distance = l2_distance(pieces[i], pieces[k])
        if distance == 0:
            l2_names. append((i,k))
        l2_list.append(distance)
        k += 1
    k = 0
    i += 1

#return the pairs of pieces which were exact matches
print(l2_names)



