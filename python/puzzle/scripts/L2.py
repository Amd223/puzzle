import numpy as np
import os

from puzzle.scripts.merge_all_dir import merge
from puzzle.tools.utils import input_image, input_directory, img_read
from puzzle.tools.crop import crop


def l2_distance(img, template):
    """
    :param img: original image
    :param template: cropped section of the image
    :return: l2 distance between images
    """

    distances = np.sqrt(np.sum(np.square(img - template), axis=1))[:,None,None]

    return distances.sum()


if __name__ == "__main__":
    dir_in = input_directory()
    resized_dir = os.path.join(dir_in, "resized")
    #img = input_image(images_relpath=resized_dir)
    #img_path = os.path.join(resized_dir, img)

    # pieces = crop(img_path, (512, 512))
    # pieces = os.listdir(pieces)

    merged_path = '../../../images/reconstructed'

    pieces = resized_dir
    print(pieces)
    pieces = os.listdir(pieces)
    print(pieces)

    l2_list = []
    l2_names = []

    for i in range(len(pieces)):
        for k in range(len(pieces)):
            path1 = os.path.join(resized_dir, pieces[i])
            path2 = os.path.join(resized_dir, pieces[k])
            img1 = img_read(path1)
            img2 = img_read(path2)
            distance = l2_distance(img1, img2)

            """
            if distance < 1:
                merge(os.path.join(resized_dir, merged_path), img1, img2)
            elif distance == 0:
                l2_names.append((pieces[i], pieces[k]))
            l2_list.append(distance)
            """

    # return the pairs of pieces which were exact matches
    print(l2_names)