import numpy as np

def spike_detector(ft_img: np.ndarray, k_size = (21, 21), threshold: float = 5.0):
    """
    Detect spikes in an image by calculate z value of each pixel. If z value is greater than threshold, the pixel is True
    :param ft_img: fft image
    :param k_size: size of the kernel
    :param threshold: z threshold for spike detection
    :return: binary image with detected spikes marked as True, and list of coordinate of spikes
    """
    # Assure the kernel have a pixel at center
    if k_size[0] % 2 == 0 or k_size[1] % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Pad the image with zeros
    clone1 = np.pad(ft_img, ((k_size[0]//2, k_size[0]//2), (k_size[1]//2, k_size[1]//2)), "constant", constant_values=255)
    clone2 = ft_img.copy()

    width = len(ft_img[0])
    height = len(ft_img[1])

    points = []

    for u in range (0, width):
        for v in range (0, height):
            # If the distance u,v is near the center, skip
            if abs(u - width//2) < k_size[0]//2 and abs(v - height//2) < k_size[1]//2:
                clone2[u, v] = 0
                continue

            # Calculate the standard deviation of all pixels in the kernel, except the center pixel
            # The center pixel is the pixel at (u, v)
            # omg O(n) = n^4 :(
            # Loop through the kernel
            values = []
            for i in range(-k_size[0]//2, k_size[0]//2):
                for j in range(-k_size[1]//2, k_size[1]//2):
                    if i == 0 and j == 0:
                        continue
                    values.append(clone1[u + i, v + j])

            # Calculate the mean
            m = np.mean(values)
            # Calculate the standard deviation
            s = np.std(values)
            # Calculate z value
            z = (ft_img[u, v] - m) / s

            if z > threshold:
                clone2[u, v] = 1
                points.append((u, v))
            else:
                clone2[u, v] = 0

    return clone2, points