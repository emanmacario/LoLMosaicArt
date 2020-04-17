from PIL import Image
import numpy as np
import os

"""
1. Read the tile images, which will replace the tiles in the original image.
2. Read the target image and split it into an M×N grid of tiles.
3. For each tile, find the best match from the input images.
4. Create the final mosaic by arranging the selected input images in an M×N grid.
"""


def preprocess(input_image):
    pass


def get_avg_rgb(image):
    """
    Given PIL image, return average (r, g, b) colour values
    :param image: PIL image
    :return: average RGB values
    """
    rasters = np.array(image)
    height, width, depth = rasters.shape
    # reshape the image
    reshaped = rasters.reshape(width*height, depth)
    averages = tuple(np.average(reshaped, axis=0))
    return averages


def split(image, M=30, N=18):
    """
    Given PIL image, returns a generator of m*n images
    :param image: PIL image
    :return: generator of MxN images
    """
    # Get size of image
    width, height = image.size
    # Get pixel offset values for width and height
    x_offset = int(width/M)
    y_offset = int(height/N)

    total = 0

    for i in range(M):
        for j in range(N):
            left = i*x_offset
            right = (i+1)*x_offset
            upper = j*y_offset
            lower = (j+1)*y_offset

            total += 1
            yield image.crop((left, upper, right, lower))

    # print(total)


def euclideanistance():
    pass

def main():
    image = Image.open('images/Aatrox_0.jpg')
    width, height = image.size
    print(width, height)

    get_avg_rgb(image)


def print_resolutions():
    resolutions = set()

    for f in os.listdir('images/'):
        im = Image.open('images/' + f)
        resolutions.add(im.size)

    print(resolutions)


if __name__ == "__main__":
    im = Image.open('images/Aatrox_0.jpg')
    t = 0
    for i in split(im):
        if t > 10:
            break
        i.show()
