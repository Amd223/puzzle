import os
import glob
import tempfile
from random import randint

from puzzle.tools.crop import crop_loulou
from puzzle.tools.utils import input_directory, img_read
import cv2




def crop_one(img_path, crop_dim, crop_pos=(0, 0), save=True):
    """
    Extracts a crop from a given image
    :param img_path: str
        Path to the image to crop from
    :param crop_dim: (int, int)
        Dimensions (width, height) of the crop
    :param crop_pos: (int, int)
        Position of the top-left corner (x, y) of the crop
    :return: str, path of the directory containing the cropped images.
    """
    crop_width, crop_height = crop_dim
    crop_x, crop_y = crop_pos

    # Load image
    img = img_read(img_path)

    crop_img = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    if not save:
        return crop_img


    # Create temp dir of outputs
    tmp_dir = tempfile.mkdtemp()
    _, img_extension = os.path.splitext(img_path)
    crop_name = os.path.join(tmp_dir, 'crop{}'.format(img_extension))
    cv2.imwrite(crop_name, crop_img)

    return crop_name

X_1 = []
X_2 = []
Y = []

def select_crops(img_path, crop_dim=(16,16), list1 = X_1, list2 = X_2, class_list = Y):


    x = randint(1, 13)
    y = randint(1, 14)

    crop_pos = (x, y)
    print(crop_pos)
    crop = crop_one(img_path, crop_dim, crop_pos, save=False)

    new_crop_pos = (x+1, y+1)
    print(new_crop_pos)
    crop2 = crop_one(img_path, crop_dim, new_crop_pos, save=False)

    X_1.append(crop)
    X_2.append(crop2)
    Y.append(0) # 0 indicates the 2 pieces should be next to each other

    return crop, crop2


if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)
    img_dir = os.path.realpath(os.path.join(curr_dir, '../../../images'))

    for image in glob.iglob(img_dir + '/**/*.jpg', recursive=True):
        select_crops(image)

    #cv2.imshow("img", select_crops(dir_in)[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
