from matplotlib import pyplot as plt
import utils.addnoise as addnoise
from utils.rgb2gray import rgb2gray
import Filters.fft_denoiser, Filters.gauss_filter, Filters.metrics, Filters.NLM, Filters.median_filter
from Filters.NLM import NLMeans
def image_load(path):
    return plt.imread(path)

def multiple_plot(img1, img2, img3, img4, img5)-> None:
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 5, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 5, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Gaussian')
    plt.subplot(1, 5, 3)
    plt.imshow(img3, cmap='gray')
    plt.title('Median')
    plt.subplot(1, 5, 4)
    plt.imshow(img4, cmap='gray')
    plt.title('NLM')
    plt.subplot(1, 5, 5)
    plt.imshow(img5, cmap='gray')
    plt.title('FFT')
    plt.show()

def plotter(img1, img2, title) -> None:
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title)
    plt.show()

def main() -> None:
    # Load image
    img = image_load('./dataset/JFK.png')
    # Convert to grayscale
    img = rgb2gray(img)

    # Apply filters
    kernel_size = 10
    window_size = 5
    sigma = 2 # Sigma is the standard deviation of the Gaussian distribution

    # Gaussian Filter
    kernel = Filters.gauss_filter.gaussian_kernel(kernel_size, sigma)
    img_gauss = Filters.gauss_filter.convolve(img.copy(), kernel)
    plotter(img, img_gauss, 'Gaussian filter')

    # Median Filter
    median_filter_image = Filters.median_filter.median_filter(img.copy(), window_size)
    average_filtered_image = Filters.median_filter.average_filter(img.copy(), window_size)
    plotter(median_filter_image, average_filtered_image, 'Median filter')

    # NLM Filter
    denoiser = NLMeans()
    gauss_noise = denoiser.solve(img.copy(), 27)
    plotter(img, gauss_noise, 'NLM filter')

    # FFT Filter
    fft_filter_image = Filters.fft_denoiser.denoiser(img.copy())

    # Plot all filters
    multiple_plot(img, img_gauss, median_filter_image, gauss_noise, fft_filter_image)

if __name__ == '__main__':
    main()

