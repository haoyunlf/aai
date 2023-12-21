import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from statistics import mean

def check_wavdur(folder_path):
    # 存储所有音频时长的列表
    durations = []
    
    # 遍历文件夹
    for subdir in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subdir, 'wav')
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(subfolder_path, file)
                    # 使用librosa提取音频时长
                    y, sr = librosa.load(file_path)
                    duration = librosa.get_duration(y=y, sr=sr)
                    durations.append(duration)
    
    # 画出音频时长分布的直方图
    plt.hist(durations, bins=20, alpha=0.7, color='b')
    plt.xlabel('Duration (s)')
    plt.ylabel('Count')
    plt.title('Distribution of Audio Durations')
    plt.show()
    plt.savefig('dur_haskins.png')
    
    # 打印最大最小值
    print("Mean:", mean(durations))
    print("Max:", max(durations))
    print("Min:", min(durations))

def compute_coef(xarr, yarr):
    r = np.corrcoef(xarr, yarr)
    #a = np.concatenate([r, r], axis=1)
    print('R', r)
    print(np.diagonal(r, offset=2)) #offset需要是xarr的行数
    return r
#check_wavdur('/home/yun/DATASETS/ema/Raw_data/Haskins')

xarr = np.array([[1,1,3,4],[5,1,1,1]])
yarr = np.array([[1,1,5,6],[7,6,4,2]])
print(np.corrcoef([1,1,3,4], [1,1,5,6]))
print(np.corrcoef([5,1,1,1], [7,6,4,2]))
print(compute_coef(xarr, yarr))
print(np.corrcoef([1,1,3,4], [7,6,4,2]))