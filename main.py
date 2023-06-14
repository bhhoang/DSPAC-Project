from matplatlib import pyplot as plt
import utils.addnoise, utils.rgb2gray
import Filters.fft_denoiser, Filters.gauss_filter, Filters.metrics, Filters.NLM, Filters.median_filter

def image_load(path):
    return plt.imread(path)

