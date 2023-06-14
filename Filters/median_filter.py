import numpy as np
import matplotlib.pyplot as plt


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

