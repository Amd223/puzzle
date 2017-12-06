from pylab import *
from PIL import Image

pil_im = Image.open("adam.jpg")
# pil_im = Image.open("adam.jpg").convert("L")


# crop a region from the image, rotate it and then paste it back in the original image
box = (100, 100, 400, 400)
region = pil_im.crop(box)
region = region.transpose(Image.ROTATE_180)
pil_im.paste(region, box)
pil_im.show()


# plots an image and four points with red stars at the x and y coordinates
im = array(pil_im)  # read image to array
imshow(im)
print("Please click 2 points")
x = ginput(2)
print("you clicked:’,x show")

x = [100,100,400,400]  # initialise point x
y = [200,500,200,500]  # initialise point y
plot(x,y,'r*')  # plot the points with red star-markers
plot(x[:2],y[:2])  # line plot connecting the first two points
plot(x,y,'go-')  # green line with circle-markers
plot(x,y,'ks:')  # black dotted line with square-markers
title('Plotting: "adam.jpg"')  # add title
axis('off')



im_2 = array(Image.open('adam.jpg').convert('L'))
figure()  # create a new figure
gray()  # don’t use colors
contour(im_2, origin='image')  # show contours with origin upper left corner
axis('equal')

# display graylevel image histogram
figure()
hist(im_2.flatten(), 50)  # image is flattened because 1-dimensional array as input


print("Please click 3 time")
x = ginput(3)
print("You clicked", x)
show()
