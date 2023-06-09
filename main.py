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

# references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html, https://homepages.inf.ed.ac.uk/rbf/HIPR2/median.htm, https://www.youtube.com/watch?v=-eyICcPd-zE
# Median Filter (Spartial Non-Linear Filter)
def median_filter(image, kernel_size):
    # Get the dimensions of the image
    height, width = image.shape
    
    # Calculate the kernel offset
    k = kernel_size // 2
    
    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Loop through each pixel in the image
    for i in range(height):
        for j in range(width):
            # Create an empty list to store the pixel values of the neighbors
            neighbors = []
            
            # Loop through the kernel window
            for m in range(-k, k+1):
                for n in range(-k, k+1):
                    # Check if the neighbor pixel is within the image boundaries
                    if i+m >= 0 and i+m < height and j+n >= 0 and j+n < width:
                        # Append the pixel value to the neighbors list
                        neighbors.append(image[i+m, j+n])
            
            # Calculate the median value of the neighbors
            median = np.median(neighbors)
            
            # Set the filtered pixel value in the output image
            filtered_image[i, j] = median
    # Plot the filtered image
    plt.figure("Filtered image")
    plt.imshow(filtered_image, plt.cm.gray)
    plt.title('Median Filtered Image')
    plt.axis('off')
    plt.show()
    return filtered_image

# references:https://www.mathworks.com/help/coder/gs/averaging-filter.html https://en.wikipedia.org/wiki/Image_segmentation
# Average Filter (Spartial Linear Filter)
def average_filter(image, window_size):
    # Get image dimensions
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    # Scale the image values from [0.0, 1.0] to [0, 255]
    scaled_image = (image * 255).astype(np.uint8)

    # Create an output image with the same dimensions as the input image
    filtered_image = np.zeros_like(scaled_image, dtype=np.uint8)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate the average value for the window centered at the current pixel
            sum_channels = np.zeros(channels)
            count = 0
            for j in range(-window_size // 2, window_size // 2 + 1):
                for i in range(-window_size // 2, window_size // 2 + 1):
                    # Check if the pixel is within the image boundaries
                    if 0 <= y + j < height and 0 <= x + i < width:
                        sum_channels += scaled_image[y + j, x + i]
                        count += 1

            # Calculate the average value for each channel
            avg_channels = sum_channels // count

            # Set the filtered pixel value in the output image
            filtered_image[y, x] = avg_channels.astype(np.uint8)

    # Plot the filtered image
    plt.figure("Filtered image")
    plt.imshow(filtered_image, plt.cm.gray)
    plt.title('Average Filtered Image')
    plt.axis('off')
    plt.show()

    # Return the filtered image
    return filtered_image

# Define the kernel size for the median filter
kernel_size = 3  # Adjust this value according to your needs

# Define the kernel size for the median filter
window_size = 5  # Adjust this value according to your needs

# Apply the median filter
median_filtered_image = median_filter(img, kernel_size)

# Apply the median filter
average_filtered_image = average_filter(img, window_size)




