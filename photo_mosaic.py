from collections import defaultdict
from PIL import Image
import numpy as np
import os

"""
1. Read the tile images, which will replace the tiles in the original image.
2. Read the target image and split it into an M×N grid of tiles.
3. For each tile, find the best match from the input images.
4. Create the final mosaic by arranging the selected input images in an M×N grid.
"""


def get_image_filenames():
    filenames = []
    for f in os.listdir('images/'):
        filenames.append(f)
    return filenames


def preprocess(image, M, N):
    """
    Crops a PIL image to ensure that its width and height
    in pixels are a multiple of M and N respectively
    :param input_image: PIL image
    :param M: width factor
    :param N: height factor
    :return: cropped image
    """
    # Get image size
    width, height = image.size

    # Calculate right and bottom pixel boundaries.
    # This process may be lossy, depending on the
    # precision of M and N
    left = 0
    right = width - (width % M)
    upper = 0
    lower = height - (height % N)
    return image.crop((left, upper, right, lower))


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


def split(image, M=15, N=9): # DEFAULT M=30, N=18
    """
    Given PIL image, returns a generator of m*n images
    :param image: PIL image
    :return: generator of MxN images
    """
    # First, preprocess the image
    image = preprocess(image, M, N)
    # Get size of the image
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


def euclidean_distance(rgb1, rgb2):
    """
    Calculates the Euclidean distance between two RGB vectors
    :param rgb1: first vector
    :param rgb2: second vector
    :return: distance
    """
    return np.linalg.norm(np.array(rgb1)-np.array(rgb2))


def process_base_images():
    """
    Calculates the average RGB values for all images
    that will constitute the photomosaic image
    :return: dictionary of RGB averages
    """
    base_rgb_avgs = defaultdict(tuple)
    filenames = get_image_filenames()
    total = len(filenames)
    complete = 0

    for f in filenames:
        im = Image.open('images/' + f)
        base_rgb_avgs[f] = get_avg_rgb(im)
        complete += 1

        if complete % 10 == 0:
            print("%4d/%4d complete" % (complete, total))

    # TODO: Write to JSON to avoid long processing times

    return base_rgb_avgs


def main():
    image = Image.open('images/Aatrox_0.jpg')
    width, height = image.size
    print(width, height)

    get_avg_rgb(image)



def index_image(image, base_rgb_avgs):
    """
    Assign a single (split) image to a base image
    :param image: a cropped image from the split image
    :return: filename denoting image with closest average RGB values
    """
    min_distance = np.inf
    index = ''
    rgb1 = get_avg_rgb(image)
    for im in base_rgb_avgs:
        rgb2 = base_rgb_avgs[im]
        distance = euclidean_distance(rgb1, rgb2)
        if distance < min_distance:
            min_distance = distance
            index = im
    assert index
    return index




def index_split_images(image, base_rgb_avs):
    """
    For each image in the split, assign a base image
    with the closest average RBG value to it
    :return: List of base image filenames
    """
    index = []
    for im in split(image):
        index.append(index_image(im, base_rgb_avgs))
    return index


def generate_mosaic(image, index, M=15, N=9):
    image = preprocess(image)
    width, height = image.size

    # Current index value
    n = 0

    # Create empty final mosaic image
    mosaic = Image.new('RGB', (width*N, height*M))

    # Fill in the mosaic with base images
    # by using the index as a reference
    for i in range(M):
        # Create a new row
        row = Image.new('RGB', (width*9, height))
        for j in range(N):
            # Append a base image to the row
            im = Image.open('images/' + index[n])
            im = preprocess(im, M, N)
            row.paste(im, (j*width, 0))
            n += 1

        # Append row to the mosaic
        mosaic.paste(row, (0, i*height))

    # Save the mosaic image
    mosaic.save('mosaics/mosaic.jpg')


def print_resolutions():
    resolutions = set()

    for f in os.listdir('images/'):
        im = Image.open('images/' + f)
        resolutions.add(im.size)

    print(resolutions)


if __name__ == "__main__":
    """
    im = Image.open('images/Aatrox_0.jpg')
    width, height = im.size
    print(width)
    print(height)
    print(len(get_image_filenames()))
    """


    base_rgb_avgs = process_base_images()
    print(len(base_rgb_avgs))
    print(base_rgb_avgs)