<<<<<<< Updated upstream
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.image as mpimg
from scipy.fftpack import fft2, fftfreq, fftshift, ifft2
from scipy import ndimage

# RGB to grayscale formula: Y' = 0.2989 R + 0.5870 G + 0.1140 B
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# Loading image
img = mpimg.imread('image.png')
# Convert to grayscale
img = rgb2gray(img)
plt.figure("Original image")
plt.imshow(img, plt.cm.gray)
plt.axis('off')
plt.title('Original image')

# Fourier transform
img_fft = fft2(img)

def spectrum_plot(img_fft):
    # Plotting spectrum
    plt.figure("Spectrum")
    plt.imshow(np.abs(img_fft), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.show()

spectrum_plot(img_fft)

# Filter in FFT
def filter_fft(img_fft):
    keep_fraction = 0.1
    im_fft2 = img_fft.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    plt.figure("Filtered spectrum")
    plt.imshow(np.abs(im_fft2), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Filtered Fourier transform')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.show()
    return im_fft2

img_fft2 = filter_fft(img_fft)

# Reconstructing image from filtered FFT, keeping real part for displaying the image
img_new = ifft2(img_fft2).real
plt.figure("Filtered image")
plt.imshow(img_new, plt.cm.gray)
plt.axis('off')
plt.title('Filtered image')
plt.show()
=======
from matplatlib import pyplot as plt
import utils.addnoise, utils.rgb2gray
import Filters.fft_denoiser, Filters.gauss_filter, Filters.metrics, Filters.NLM, Filters.median_filter
>>>>>>> Stashed changes

def image_load(path):
    return plt.imread(path))

