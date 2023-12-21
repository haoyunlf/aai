import librosa
import numpy as np
print(librosa.__version__)

y, sr = librosa.load('/home/yun/DATASETS/ema/Raw_data/Haskins/F01/wav/F01_B02_S52_R02_N.wav')

y = 0.5 * y / np.max(y)
yt, index = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=128)
print(librosa.get_duration(y=y, sr=sr), librosa.get_duration(y=yt, sr=sr))
dur = librosa.get_duration(y=y, sr=sr)
print('sr: {}'.format(sr))
t = [max(0, index[0]/sr-0.1), index[1]/sr]

print(index, t)
print(len(y))
print(len(yt))

#xtrm_temp_ema = [int(xtrm[0] * self.sampling_rate_ema), min(int((xtrm[1] * self.sampling_rate_ema) + 1), len(ema))]

# xtrm_temp_mfcc = [int(xtrm[0] / self.hop_time),
# int(np.ceil(xtrm[1] / self.hop_time))]

xtrm_temp_mfcc = [int(t[0] / 0.01), int(np.ceil(t[1] / 0.01))]
xtrm_temp_ema = [int(t[0] * 100), int(np.ceil(t[1] * 100))]

print(xtrm_temp_ema, xtrm_temp_mfcc)

# mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]

# ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]