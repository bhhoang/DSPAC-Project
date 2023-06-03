import numpy as np

def pcm_to_decibel(pcm_value, bit_depth):
    max_amplitude = 2 ** (bit_depth - 1) - 1
    amplitude = pcm_value / max_amplitude
    decibel = 20 * np.log10(abs(amplitude))
    return decibel

def decibel_to_pcm(decibel, bit_depth):
    max_amplitude = 2 ** (bit_depth - 1) - 1
    amplitude = 10 ** (decibel / 20)
    pcm_value = amplitude * max_amplitude
    return pcm_value
