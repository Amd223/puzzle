from PIL import Image
from pylab import *

pil_im = Image.open("adam.jpg")
pil_im_paste = Image.open("adam.jpg")
# pil_im = Image.open("adam.jpg").convert("L")


# crop a region from the image, rotate it and then paste it back in the original image
box = (100,100,400,400)
region = pil_im.crop(box)
region = region.transpose(Image.ROTATE_180)
pil_im_paste.paste(region, box)

# plots an image and four points with red stars at the x and y coordinates
im = array(pil_im)  # read image to array
imshow(im)
x = [100,100,400,400]  # initialise point x
y = [200,500,200,500]  # initialise point y
plot(x,y,'r*')  # plot the points with red star-markers
plot(x[:2],y[:2])  # line plot connecting the first two points
plot(x,y,'go-')  # green line with circle-markers
plot(x,y,'ks:')  # black dotted line with square-markers
title('Plotting: "adam.jpg"')  # add title
# axis('off')


im = array(Image.open('adam.jpg').convert('L'))
figure()  # create a new figure
gray()  # don’t use colors
contour(im, origin='image')  # show contours with origin upper left corner
axis('equal')
show()

# display graylevel image histogram
figure()
hist(im.flatten(), 128)  # image is flattened because 1-dimensional array as input
show()
