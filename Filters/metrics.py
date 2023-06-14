import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


def MSE(image1, image2):
    """
  Mean Squared Error
  :param image1: image1
  :param image2: image2
  :rtype: float
  :return: MSE value
  """

    # Calculating the Mean Squared Error
    mse = np.mean(np.square(image1.astype(np.float) - image2.astype(np.float)))

    return mse


def PSNR(image1, image2, peak=255):
    """
  Peak signal-to-noise ratio
  :param image1: image1
  :param image2: image2
  :param peak: max value of pixel 8-bit image (255)
  :rtype: float
  :return: PSNR value
  """

    # Calculating the Mean Squared Error
    mse = MSE(image1, image2)

    # Calculating the Peak Signal Noise Ratio
    psnr = 10 * np.log10(peak ** 2 / mse)

    return psnr
