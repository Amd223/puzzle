import os
import tempfile


import cv2

from puzzle.tools.utils import img_read, input_image

rel_path         = '../../../images'
rel_path_resized = '../../../images/resized'
rel_path_merged  = '../../../images/merged'
rel_path_merged_pieces  = '../../../pieces'



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

def crop(img_path, block_dim):
    """
    Crops an image into blocks of given width / height
    :param img_path: str
        Path of the image to crop from
    :param block_dim: (int, int)
        Tuple of block_width, block_height
    :return: str
        Path of the directory containing the cropped images.
    """
    block_height, block_width = block_dim

    # Create temp dir of outputs
    tmp_dir = tempfile.mkdtemp()

    # Load image
    img = img_read(img_path)
    height, width = img.shape[:2]

    dir_out = os.path.realpath(os.path.join(img_path, rel_path_merged_pieces))
    n = len(os.listdir(dir_out))
    print(n)

    # Iterate in the range(begin, end, step)
    for y in range(0, height, block_height):
        for x in range(0, width, block_width):
            n +=1
            crop_name2 = os.path.join(dir_out, str(n) + ".jpg")
            #crop_name = os.path.join(tmp_dir, 'img_y-%d_x-%d.png' % (y, x))
            crop_img  = img[y:y+block_height, x:x+block_width]
            cv2.imwrite(crop_name2, crop_img)

    return crop_name2

def crop_loulou(img_path, block_dim):
    """
    Crops an image into blocks of given width / height
    :param img_path: str
        Path of the image to crop from
    :param block_dim: (int, int)
        Tuple of block_width, block_height
    :return: str
        Path of the directory containing the cropped images.
    """

    block_height, block_width = block_dim

    # Create temp dir of outputs
    tmp_dir = tempfile.mkdtemp()

    # Load image
    img = img_read(img_path)
    height, width = img.shape[:2]

    # Iterate in the range(begin, end, step)
    for y in range(0, height, block_height):
        for x in range(0, width, block_width):
            crop_name = os.path.join(tmp_dir, 'img_y-%d_x-%d.png' % (y, x))
            crop_img  = img[y:y+block_height, x:x+block_width]
            cv2.imwrite(crop_name, crop_img)

    return tmp_dir

def crop_interactive(img_path=None, show_crop=True):
    """
    Interactively create a crop from a given (optional) image path
    :param img_path: str,
        Path to the image to crop from
    :param show_crop: bool,
        Whether to show the cropped image as a confirmation. Default is True
    :return: (str, (int, int))
        (Path to the cropped image, crop location)
    """
    image = None
    ref_pt = []
    SOURCE_IMG = 'Source image'

    if img_path is None:
        img_path = input_image()

    # load the image and save a clone of it
    image = img_read(img_path)
    clone = image.copy()

    def reset_ref():
        nonlocal ref_pt, image
        ref_pt = []
        image = clone.copy()
        cv2.imshow(SOURCE_IMG, image)

    def append_ref(pt):
        nonlocal ref_pt
        ref_pt.append(pt)

    def click_and_crop(event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            reset_ref()
            append_ref((x, y))

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            append_ref((x, y))

            # draw a rectangle around the region of interest
            white = 255
            [pt_a, pt_b] = ref_pt
            cv2.rectangle(image, pt_a, pt_b, white, 2)
            cv2.imshow(SOURCE_IMG, image)

    cv2.namedWindow(SOURCE_IMG)
    cv2.setMouseCallback(SOURCE_IMG, click_and_crop)

    try:
        # Keep looping until the 'q' key is pressed or 'c'
        while True:
            # Display the image and wait for a keypress
            cv2.imshow(SOURCE_IMG, image)
            key = cv2.waitKey(0) & 0xFF

            # 'r' key: reset the cropping region
            if key == ord("r"):
                image = clone.copy()

            # 'c' key: crop the region of interest
            elif key == ord("c") and len(ref_pt) == 2:
                [(pt_a_x, pt_a_y), (pt_b_x, pt_b_y)] = ref_pt
                top_left     = (min(pt_a_x, pt_b_x), min(pt_a_y, pt_b_y))
                bottom_right = (max(pt_a_x, pt_b_x), max(pt_a_y, pt_b_y))

                top_left_x, top_left_y = top_left
                bottom_right_x, bottom_right_y = bottom_right

                template = clone[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                if show_crop:
                    cv2.imshow("Template image", template)
                    cv2.waitKey(0)

                dim_x = abs(bottom_right_x - top_left_x)
                dim_y = abs(bottom_right_y - top_left_y)
                template_path = crop_one(img_path, (dim_x, dim_y), top_left)

                return template_path, top_left
    finally:
        cv2.destroyAllWindows()
