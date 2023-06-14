import numpy as np

# RGB to grayscale formula: Y' = 0.2989 R + 0.5870 G + 0.1140 B
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
