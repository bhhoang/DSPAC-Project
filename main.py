import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

file_path = './Example/guitar.wav'  # Define file path

# Define variable
threshold = 20  # dB
ratio = 10  # :1
gain_reduction = 0  # dB
gain_increase = 0  # dB
attack = 0  # ms
release = 0  # ms

# Read file
samplerate, data = wav.read(file_path)

print ("Sample rate: ", samplerate, "Hz")
print ("Number of channels: ", len(data[0]))
print ("Number of samples: ", len(data))
print ("Duration: ", len(data)/samplerate, "s")

n = len(data)
fs = samplerate

def pcm_to_decibel(pcm_value, bit_depth) -> float:
    if pcm_value == 0:
        return 0
    max_amplitude = 2 ** (bit_depth - 1) - 1
    amplitude = pcm_value / max_amplitude
    decibel = 20 * np.log10(np.abs(amplitude))
    return decibel

channel_1 = np.array(data[:,0])
channel_2 = np.array(data[:,1])

channel_1_fft = np.fft.fft(channel_1) # FFT
abs_channel_1_fft = np.abs(channel_1_fft[:n//2]) # Spectrum

channel_2_fft = np.fft.fft(channel_2) # FFT
abs_channel_2_fft = np.abs(channel_2_fft[:n//2]) # Spectrum