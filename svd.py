import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def matrix_show(data):
    plt.imshow(data, cmap=plt.cm.gray)
    plt.show()


def image2matrix(filename):
    im = Image.open(filename)
    width, height = im.size
    im = im.convert("L")
    data = np.matrix(im.getdata(), dtype='float')
    new_data = np.reshape(data, (height, width))
    return new_data


def matrix2image(data):
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


def add_noise(data):
    data = data + np.random.normal(255, size=data.shape)
    return data


def svd(data):
    M, segma, VT = np.linalg.svd(data)
    segma_len = len(segma)
    segma[segma_len - 10:] = 0
    segma = np.eye(M.shape[1], VT.shape[0]) * segma.reshape(segma_len, 1)
    ori_data = np.dot(M, np.dot(segma, VT))

    return ori_data


if __name__ == "__main__":
    data = image2matrix('images.jpg')

    data = add_noise(data)
    matrix_show(data)

    data = svd(data)
    matrix_show(data)
