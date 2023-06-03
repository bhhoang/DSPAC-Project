from matplotlib import pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

file_path = './Example/guitar.wav' # Define file path

# Define variable
threshold = 20 # dB
ratio = 10 # :1
gain_reduction = 0 # dB
gain_increase = 0 # dB
attack = 0 # ms
release = 0 # ms

# Read file
samplerate, data = wav.read(file_path)

print ("Sample rate: ", samplerate, "Hz")
print ("Number of channels: ", len(data[0]))
print ("Number of samples: ", len(data))
print ("Duration: ", len(data)/samplerate, "s")

n = len(data)
fs = samplerate

channel_1 = np.array(data[:,0])
channel_2 = np.array(data[:,1])

channel_1_fft = np.fft.fft(channel_1) # FFT
abs_channel_1_fft = np.abs(channel_1_fft[:n//2]) # Spectrum

plt.figure("2 Channel Spectrum signal") # Plot spectrum
plt.subplot(1,1,1)
plt.plot(np.linspace(0, fs/2, n//2), abs_channel_1_fft) # Plot spectrum
plt.title("Channel 1 Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum magnitude (dB)")
plt.grid()

channel_2_fft = np.fft.fft(channel_2) # FFT
abs_channel_2_fft = np.abs(channel_2_fft[:n//2]) # Spectrum

plt.subplot(1,1,2)
plt.plot(np.linspace(0, fs/2, n//2), abs_channel_2_fft, color='red') # Plot spectrum
plt.title("Channel 2 Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum magnitude (dB)")
plt.grid()

plt.show()
