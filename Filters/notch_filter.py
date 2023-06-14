import numpy as np
import matplotlib.pyplot as plt


def notch_reject_filter(shape, radius=9, x=0, y=0):
    """
    Create a notch reject filter. Use for periodic noise removal.
    :param shape: shape of the image FFT
    :param radius: radius of the filter
    :param x: x offset of the filter
    :param y: y offset of the filter
    :return: notch reject filter to be multiplied with the FFT
    """

    P = shape[0]
    Q = shape[1]

    # Initialize filter with zeros
    mask = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get distance from point D(u,v) and the antipodal point D(-u,-v) from the center
            dist = np.sqrt((u - P / 2 + x) ** 2 + (v - Q / 2 + y) ** 2)
            antipodal_dist = np.sqrt((u - P / 2 - x) ** 2 + (v - Q / 2 - y) ** 2)

            if dist <= radius or antipodal_dist <= radius:
                mask[u, v] = 0.0
            else:
                mask[u, v] = 1.0

    return mask

# Example:

# Get the magnitude spectrum
# fft_image = fft2(image)
# fft_image = fftshift(fft_image)
# fft_mag = 20 * np.log(np.abs(fft))

# Create notch reject filters
# H1 = notch_reject_filter(fs.shape, 9, 176, 176)
# H2 = notch_reject_filter(fs.shape, 9, 0, 350-176)
# H3 = notch_reject_filter(fs.shape, 9, 350-176, 0)
# H4 = notch_reject_filter(fs.shape, 9, -350+176, 176)

# Apply the filters to the image FFT
# fft_image = fft_image * H1 * H2 * H3 * H4

# Show the filtered image
# plt.figure("Filtered Image")
# plt.imshow(np.abs(ifft2(ifftshift(fft_image))), plt.cm.gray)
# plt.title('Filtered Image')
# plt.show()
