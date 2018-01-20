from scipy.ndimage import measurements
from puzzle.tools.crop import crop_interactive
from puzzle.tools.utils import img_read, input_image

img_path = input_image("Input an image name from '{}': \n")
print('Using image src : ' + img_path)

template_path, template_pos = crop_interactive(img_path, show_crop=False)

# Load images
img = img_read(img_path)
template = img_read(template_path)
print("dimensions: ", template.shape)

# load image and threshold to make sure it is binary
img = 1*(img<128)
template = 1*(template<128)

labels, nbr_objects = measurements.label(img)
labels2, nbr_objects2 = measurements.label(template)
print("Number of objects in original image:", nbr_objects)
print("Number of objects in original template:", nbr_objects2)



