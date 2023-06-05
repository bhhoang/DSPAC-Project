import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.io import wavfile


def pcm_to_decibel(pcm_value, bit_depth) -> float:
    if pcm_value == 0:
        return 0
    max_amplitude = 2 ** (bit_depth - 1) - 1
    amplitude = pcm_value / max_amplitude
    decibel = 20 * np.log10(np.abs(amplitude))
    return decibel

# Plot data in time domain

fig1 = plt.figure(figsize=(9, 9))
sr, data = wavfile.read("Example\\guitar.wav")
length = len(data)/sr
time = np.linspace(0, length, len(data))

fig1.add_subplot(211)
right_channel = data[:, 0]
plt.title("Time domain - Right channel")
plt.specgram(right_channel, Fs=sr, vmin=-20, vmax=80)
plt.xlim(0, length)
plt.colorbar()

fig1.add_subplot(212)
left_channel = data[:, 1]
plt.title("Time domain - Left channel")
plt.specgram(left_channel, Fs=sr, vmin=-20, vmax=80)
plt.xlim(0, length)
plt.colorbar()
plt.tight_layout()
plt.show()

# Plot data in frequency domain

fig2 = plt.figure(figsize=(10, 5))
ax = fig2.add_subplot(211)
right_channel = data[:, 0]
fourier_data = sp.fft.rfft(right_channel)

db = []
for pcm_val in fourier_data:
    db.append(pcm_to_decibel(pcm_val, 16))
db = np.array(db)

freq_spectrum = sp.fft.rfftfreq(right_channel.size, 1/sr)

plt.title("Frequency domain - Right channel")
plt.xlabel("Frequency (Hz) - Symmetrical log scale")
plt.ylabel("Decibel (dB)")
ax.set_xscale('symlog')
plt.stem(freq_spectrum, db, markerfmt=" ")

ax = fig2.add_subplot(212)
plt.xlabel("Frequency (Hz) - Normal scale")
plt.ylabel("Decibel (dB)")
plt.stem(freq_spectrum, db, markerfmt=" ")
plt.tight_layout()
plt.show()