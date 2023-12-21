import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

# Load sample data from a WAV file
#sample_rate, data = scipy.io.wavfile.read('/home/yun/DATASETS/ema/Raw_data/Haskins/F01/wav/F01_B10_S07_R01_N.wav')
sample_rate = 500
data = np.load('/home/yun/bootphon_articulatory_inverison/Preprocessed_data_HY/fsew0/ema/fsew0_264.npy')
data = data[:, 1]
print(data.shape)
times = np.arange(len(data))/sample_rate
print(times)

# Apply a 50 Hz low-pass filter to the original data
filtered = lowpass(data, 5, sample_rate)

# Code used to display the result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
ax1.plot(times, data)
ax1.set_title("Original Signal")
ax1.margins(0, .1)
ax1.grid(alpha=.5, ls='--')
ax2.plot(times, filtered)
ax2.set_title("Low-Pass Filter (10 Hz)")
ax2.grid(alpha=.5, ls='--')
plt.tight_layout()
plt.show()
plt.savefig("filters.png") 
plt.show()