import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from activations import relu

def convolve_single(source, filter, stride = 1, mode = 'normal'):
    '''
        convolve_single: used to perform convolution between a single image and a filter.

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

    for i in range(0, height, stride):
        for j in range(0, width, stride):
                current_region = source[i:i + filter_size, j:j + filter_size, :]
                temp = current_region * filter
                result[i, j] += temp.sum()
    return result

def convolve(source, filter, stride = 1, mode = 'normal'):
    '''
        Convolve: used to perform convolution between an image array and a filter.

        Returns:
            result: The result of convolution
        Args:
            source: The source matrix with which the filter is convolved
            filter: The matrix used to detect features in the image_size
            stride: Defines how much to slide the filter by
            mode: The mode of convolution. Can be 'same' or 'normal'
    '''
    samples, image_height, image_width, channels = source.shape
    filter_size = filter.shape[0] # FIlter must be a square matrix

    height = (image_height - filter_size)//stride + 1
    width = (image_width - filter_size)//stride + 1
    result = np.zeros((samples, height, width))
    temp = np.zeros(samples)


    for i in range(0, height, stride):
        for j in range(0, width, stride):
                current_region = source[:, i:i + filter_size, j:j + filter_size, :]
                temp = current_region * filter
                x = temp.sum(axis = (1, 2, 3))
                result[:, i, j] += x
    return result

def max_pool(source, pool_size, stride = 'auto'):
    '''
        max_pool: used to perform max pooling of an image.

        Returns:
            result: The result of max pooling
        Args:
            source: The source matrix to max pool
            pool_size : The size of the pooling matrix
            stride: Used to set the pool distance
    '''
    image_height, image_width = source.shape
    if stride == 'auto':
        stride = pool_size

    height = (image_height - pool_size)//stride + 1
    width = (image_width - pool_size)//stride + 1
    result = np.zeros((height, width))

    for i in range(0, height, stride):
        for j in range(0, width, stride):
                current_region = source[i:i + pool_size, j:j + pool_size]
                result[i, j] = current_region.max()
    return result

def visualize(input, n):
    x = 0
    for i in input:
        plt.imshow(i)
        plt.show()
        x += 1
        if x == n:
            break

def main():
    src1, src2, src3, src4 = mpimg.imread('pic.jpg'),mpimg.imread('pic2.jpg'), mpimg.imread('pic.jpg'),mpimg.imread('pic2.jpg')
    src = np.array([src1, src2, src3, src4])
    # del src1, src2
    print(src.shape)
    # visualize(src, 3)

    filter = np.array((([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),
                        ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),
                        ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]),))
                        # result = convolve(src1, filter)

    start = time.time()
    result = convolve(src, filter)
    # result = relu(result)
    end = time.time()
    print('Time taken for vectorized: ', end - start)

    start = time.time()
    result = np.array([convolve_single(src1, filter), convolve_single(src2, filter), convolve_single(src3, filter), convolve_single(src4, filter)])
    print(result.shape)
    end = time.time()
    print('Time taken for non vectorized: ', end - start)
    exit()
    plt.imshow(result[0, :, :, 0], cmap = 'gray')
    plt.show()
    plt.imshow(result[1, :, :, 0], cmap = 'gray')
    plt.show()
    exit()
    print(result.shape)

    start = time.time()
    result = max_pool(result, pool_size = 2)
    end = time.time()
    print('Time taken: ', end - start)

    plt.imshow(result, cmap = 'gray')
    plt.show()
    print(result.shape)
if __name__ == '__main__':
    main()
