import os
import glob
import cv2


def img_read(img_path):
    """
    Safely load an image in OpenCV
    :param img_path: str
        Path to the image to load
    :return: file descriptor object
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("No image at '{}'".format(img_path))
    return img


def input_image(msg=None, images_relpath="../../../images"):
    """
    Prompts the user for an image in the default project images location
    :param images_relpath: str
        Relative path to the image directory
    :return: str
        Absolute path to the input image
    """
    if msg is None:
        msg = "Input an image name from '{}': "

    curr_dir = os.path.dirname(__file__)
    img_dir = os.path.realpath(os.path.join(curr_dir, images_relpath))

    # Prompt user
    img = input(msg.format(img_dir))
    img = os.path.join(img_dir, img)

    img_matches = glob.glob('{}.*'.format(img))
    if len(img_matches) == 0:
        raise ValueError("No image found at '{}'".format(img))

    return img_matches[0]
