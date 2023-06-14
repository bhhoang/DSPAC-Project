from skimage.util import random_noise
import numpy as np

class noisy_system():
  def example(self,img,**kwargs):
    """
    An example function to test expected return.
    You can read more about skimage.util.random_noise at https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
    """
    noisy_image = random_noise(img,**kwargs)
    noisy_image = np.uint8(noisy_image*255)
    return noisy_image

  def create_salt_and_pepper_noise(self,img,amount=0.05):
    """
    function to create salt and pepper noise
    :param image: input image
    :rtype: uint8 (w,h)
    :return: noisy image
    
    """

    # Converting pixel values from 0-255 to 0-1 float
    img = img/255

    # Getting the dimensions of the image
    h = img.shape[0]
    w = img.shape[1]

    # Setting the ratio of salt and pepper in salt and pepper noised image
    s = 0.5
    p = 0.5

    # Initializing the result (noisy) image
    result = img.copy()

    # Adding salt noise to the image
    salt = np.ceil(amount * img.size * s)
    vec = []
    for i in img.shape:
      vec.append(np.random.randint(0, i-1, int(salt)))
    
    result[vec] = 1

    # Adding pepper noise to the image
    pepper = np.ceil(amount * img.size * p)
    vec = []
    for i in img.shape:
      vec.append(np.random.randint(0, i-1, int(salt)))
    
    result[vec] = 0

    # Converting the result back to uint8
    result = np.uint8(result*255)

    return result

  def create_gaussian_noise(self,img,mean=0,var=0.01):
    """
    function to create gaussian noise
    :param image: input image
    :rtype: uint8 (w,h)
    :return: noisy image
    """

    # Converting pixel values from 0-255 to 0-1 float
    img = img/255

    # Initializing the result (noisy) image
    result = img.copy()

    # Adding gaussian noise to the image
    gauss = np.random.normal(mean, var**0.5, img.shape)
    result = result + gauss
    result = np.clip(result, 0, 1)

    # Converting the result back to uint8
    result = np.uint8(result*255)

    return result