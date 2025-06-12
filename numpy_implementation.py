import numpy as np

def conv2d_gray(image, kernel):
    H, W = image.shape # H = Height of an input image / W = Width of an input image
    KH, KW = kernel.shape # KH = Height of a Kernel / KW = Width of a Kernel

    # output image size
    output_H = H - KH + 1
    output_W = W - KH + 1
    # intiailize output image with zeros
    output = np.zeros((output_H, output_W))

    # convolution
    for i in range(output_H):
        for j in range(output_W):
            output[i,j] = np.sum(image[i : i+KH, j : j+KW] * kernel)

    return output