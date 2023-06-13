import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal as signal
from . import dataset

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel

def convolve(image, kernel):
    # Get kernel size
    ksize = kernel.shape[0]

    # Get image dimensions
    height, width = image.shape

    # Calculate padding size
    pad = ksize // 2

    # Create padded image
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')

    # Initialize output image
    output = np.zeros_like(image)

    # Convolution
    for i in range(height):
        for j in range(width):
            output[i, j] = np.sum(padded_image[i:i+ksize, j:j+ksize] * kernel)

    return output

# Example usage:
# Load the image
image = cv2.imread('JFK.png', cv2.IMREAD_GRAYSCALE)

# Convert the image to float
image = image.astype(float)

# Normalize the image to the range [0, 1]
image /= 255.0

# Define the kernel size and sigma
kernel_size = 7
sigma = 2.0

# Generate the Gaussian kernel
kernel = gaussian_kernel(kernel_size, sigma)

# Apply convolution with the Gaussian kernel
filtered_image = convolve(image, kernel)

# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Gaussian)')
plt.axis('off')

plt.tight_layout()
plt.show()
