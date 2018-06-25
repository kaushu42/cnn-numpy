import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def convolve(source, filter, stride = 1, mode = 'normal'):
    '''
        Convolve: used to perform convolution between an image and a filter.

        Returns:
            result: The result of convolution
        Args:
            source: The source matrix with which the filter is convolved
            filter: The matrix used to detect features in the image_size
    '''
    image_height, image_width, channels = source.shape
    filter_size = filter.shape[0] # FIlter must be a square matrix

    height = (image_height - filter_size)//stride + 1
    width = (image_width - filter_size)//stride + 1
    result = np.zeros((height, width))

    for i in range(0, height):
        for j in range(0, width):
                current_region = source[i:i + filter_size, j:j + filter_size, :]
                temp = current_region * filter
                result[i, j] += temp.sum()
    return result

def main():
    src = mpimg.imread('pic.jpg')
    print(src.shape)
    # plt.imshow(src)
    # plt.show()
    start = time.time()

    filter = np.array((([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),
                        ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),
                        ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),))
    # exit()
    result = convolve(src, filter)
    print(result.shape)

    end = time.time()
    print('Time taken: ', end - start)

    plt.imshow(result)
    plt.show()
if __name__ == '__main__':
    main()
