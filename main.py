import cv2
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np

from utils.addnoise import periodic_noise
from utils.rgb2gray import rgb2gray
import Filters.fft_denoiser, Filters.gauss_filter, Filters.metrics, Filters.NLM, Filters.median_filter
from Filters.NLM import NLMeans
from Filters.notch_filter import notch_reject_filter


def image_load(path):
    return plt.imread(path)


def multiple_plot(img1, img2, img3, img4, img5) -> None:
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original')
    plt.subplot(2, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Gaussian')
    plt.subplot(2, 3, 3)
    plt.imshow(img3, cmap='gray')
    plt.title('Median')
    plt.subplot(2, 3, 4)
    plt.imshow(img4, cmap='gray')
    plt.title('NLM')
    plt.subplot(2, 3, 5)
    plt.imshow(img5, cmap='gray')
    plt.title('FFT')
    plt.tight_layout()
    plt.show()


def plotter(img1, img2, title) -> None:
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def periodic_noise_demo(img: np.ndarray) -> None:
    # Convert

    # Apply periodic noise
    image = img.copy()
    img_periodic = periodic_noise(image)

    # Get the magnitude spectrum
    ft = fft2(img_periodic)
    fs = fftshift(ft)
    fft_mag = 20 * np.log(np.abs(fs))

    # Create notch reject filters
    H1 = notch_reject_filter(fs.shape, 15, 176, 176)
    H2 = notch_reject_filter(fs.shape, 15, 0, 350 - 176)
    H3 = notch_reject_filter(fs.shape, 15, 350 - 176, 0)
    H4 = notch_reject_filter(fs.shape, 15, -350 + 176, 176)

    # Apply the filters to the image fft
    H = H1 * H2 * H3 * H4
    filtered_ft_mag = fft_mag * H

    # Show the original image and fft
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(2, 2, 1)
    plt.imshow(img_periodic, cmap='gray')
    plt.title('Original Image')

    fig.add_subplot(2, 2, 2)
    plt.imshow(fft_mag, cmap='gray')
    plt.title('Original FT')

    # Show the filtered image and filtered fft
    fig.add_subplot(2, 2, 3)
    plt.imshow(ifft2(ifftshift(fs * H)).real, cmap='gray')
    plt.title('Filtered Image')

    fig.add_subplot(2, 2, 4)
    plt.imshow(filtered_ft_mag, cmap='gray')
    plt.title('Filtered FT')

    plt.show()


def main() -> None:
    # Load image
    img = image_load('./dataset/JFK.png')
    # Convert to grayscale
    img = rgb2gray(img)

    # Apply filters
    kernel_size = 10
    window_size = 5
    sigma = 2  # Sigma is the standard deviation of the Gaussian distribution

    # Gaussian Filter
    kernel = Filters.gauss_filter.gaussian_kernel(kernel_size, sigma)
    img_gauss = Filters.gauss_filter.convolve(img.copy(), kernel)
    plotter(img, img_gauss, 'Gaussian filter')

    # Median Filter
    median_filter_image = Filters.median_filter.median_filter(img.copy(), window_size)
    plotter(img, median_filter_image, 'Median filter')

    # Average Filter
    average_filtered_image = Filters.median_filter.average_filter(img.copy(), window_size)
    plotter(img, average_filtered_image, 'Average filter')

    # NLM Filter
    denoiser = NLMeans()
    gauss_noise = denoiser.solve(img.copy(), 27)
    plotter(img, gauss_noise, 'NLM filter')

    # FFT Filter
    fft_filter_image = Filters.fft_denoiser.denoiser(img.copy())

    plotter(img, fft_filter_image, 'FFT filter')

    # Plot all filters
    multiple_plot(img, img_gauss, median_filter_image, gauss_noise, fft_filter_image)

    # Image with periodic noise
    img2 = cv2.imread('./dataset/bw_landscape.jpeg', 0)
    periodic_noise_demo(img2)


if __name__ == '__main__':
    main()
