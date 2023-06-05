import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import main 

#Plot data in spectrum

plt.figure("2 Channel Spectrum signal") # Plot spectrum
plt.subplot(1,2,1)
plt.plot(np.linspace(0, main.fs/2, main.n//2), main.abs_channel_1_fft) # Plot spectrum
plt.title("Channel 1 Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum magnitude (dB)")
plt.grid()

plt.subplot(1,2,2)
plt.plot(np.linspace(0, main.fs/2, main.n//2), main.abs_channel_2_fft, color='red') # Plot spectrum
plt.title("Channel 2 Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum magnitude (dB)")
plt.grid()

# Plot data in time domain

fig1 = plt.figure(figsize=(9, 9))
length = len(main.data)/main.samplerate
time = np.linspace(0, length, len(main.data))

fig1.add_subplot(211)
right_channel = main.data[:, 0]
plt.title("Time domain - Right channel")
plt.specgram(right_channel, Fs=main.samplerate, vmin=-20, vmax=80)
plt.xlim(0, length)
plt.colorbar()

fig1.add_subplot(212)
left_channel = main.data[:, 1]
plt.title("Time domain - Left channel")
plt.specgram(left_channel, Fs=main.samplerate, vmin=-20, vmax=80)
plt.xlim(0, length)
plt.colorbar()
plt.tight_layout()
plt.show()

# Plot data in frequency domain

fig2 = plt.figure(figsize=(10, 5))
ax = fig2.add_subplot(211)
right_channel = main.data[:, 0]
fourier_data = sp.fft.rfft(right_channel)

db = []
for pcm_val in fourier_data:
    db.append(main.pcm_to_decibel(pcm_val, 16))
db = np.array(db)

freq_spectrum = sp.fft.rfftfreq(right_channel.size, 1/main.samplerate)

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