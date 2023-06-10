import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal as signal
from. import dataset

def gkernel(l, sig):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

img = cv2.imread('JFK.png', 0)
# img = img.astype(np.float32) / 255.0
k = gkernel(9, 2)
denoise_img = signal.convolve2d(img, k, mode='same', boundary='symm')

plt.imshow(img, cmap='gray')
plt.show()

plt.imshow(denoise_img, cmap='gray')
plt.show()