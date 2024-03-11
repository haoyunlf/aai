#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import scipy.signal
import scipy.interpolate
import scipy.io as sio
from Preprocessing.tools_preprocessing import get_fileset_names, get_delta_features, split_sentences

from os.path import dirname
import numpy as np
import scipy.signal
import scipy.interpolate
import librosa
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
from Preprocessing.class_corpus import Speaker
import glob

import soundfile as sf
from tqdm import tqdm
import pickle

root_path = dirname(dirname(os.path.realpath(__file__)))


def detect_silence(ma_data):
    """
    :param ma_data: one "data" file containing the beginning and end of one sentence
    :return: the beginning and end (in seconds) of the entence
    We test 2 cases since the "ma_data" are not all organized in the same order.
    """
    for k in [5, 6]:
        try:
            mon_debut = ma_data[0][k][0][0][1][0][1]
            ma_fin = ma_data[0][k][0][-1][1][0][0]
        except:
            pass
    return [mon_debut, ma_fin]


class Speaker_Haskins(Speaker):
    """
    class for 1 speaker of Haskins, child of the Speaker class (in class_corpus.py),
    then inherits of some preprocessing scripts and attributes
    """
    def __init__(self, sp, path_to_raw, N_max=0):
        """
        :param sp:  name of the speaker
        :param N_max:  # max of files we want to preprocess (0 is for All files), variable useful for test
        """
        super().__init__(sp)  # gets the attributes of the Speaker class
        self.root_path = path_to_raw
        self.path_files_treated = os.path.join(root_path, "Preprocessed_data", self.speaker)
        self.path_files_brutes = os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "data")
        self.path_files_hy = os.path.join(root_path, "Preprocessed_data", self.speaker)
        self.EMA_files = sorted([name[:-4] for name in os.listdir(self.path_files_brutes) if "palate" not in name])
        self.N_max = N_max

    def read_ema_and_wav(self, display=False):
        """
        :param k: index wrt EMA_files list of the file to read
        :return: ema positions for 12 arti (K',12) , acoustic features (K,429); where K in the # of frames.
        read and reorganize the ema traj,
        calculations of the mfcc with librosa , + Delta and DeltaDelta, + 10 context frames
        # of acoustic features per frame: 13 ==> 13*3 = 39 ==> 39*11 = 429.
        parameters for mfcc calculation are defined in class_corpus
        """
        order_arti_haskins = ['td_x', 'td_y', 'tb_x', 'tb_y', 'tt_x', 'tt_y', 'ul_x', 'ul_y', "ll_x", "ll_y",
                              "ml_x", "ml_y", "li_x", "li_y", "jl_x", "jl_y"]
        order_arti = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                      'ul_x', 'ul_y', 'll_x', 'll_y']

        N = len(self.EMA_files)
        ema_all = []
        
        for k in range(N):
            try:
                data = sio.loadmat(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))[self.EMA_files[k]][0]
            except:
                print('Could not read from {}'.format(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))) #HY
                continue
    
            ema = np.zeros((len(data[1][2]), len(order_arti_haskins)))
    
            for arti in range(1, len(data)):  # lecture des trajectoires articulatoires dans le dicionnaire
                ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
                ema[:, arti * 2 - 1] = data[arti][2][:, 2]
            new_order_arti = [order_arti_haskins.index(col) for col in order_arti]
            ema = ema[:, new_order_arti]
    
            wav, sr = librosa.load(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav",
                                                self.EMA_files[k] + ".wav"), sr=self.sampling_rate_wav_wanted)
            wav = 0.5 * wav / np.max(wav)
    
            marge = 0.1 #original:0
            xtrm = detect_silence(data)
            xtrm = [max(xtrm[0] - marge, 0), xtrm[1] + marge]
    
            xtrm_temp_ema = [int(np.floor(xtrm[0] * self.sampling_rate_ema)),
                             int(min(np.floor(xtrm[1] * self.sampling_rate_ema) + 1, len(ema)))]
            xtrm_temp_wav = [int(np.floor(xtrm[0] * self.sampling_rate_wav_wanted)),
                             int(min(np.floor(xtrm[1] * self.sampling_rate_wav_wanted) + 1, len(wav)))]
            new_ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
            new_wav = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]
            if not os.path.exists(os.path.join(self.root_path, "Preprocessed_data", self.speaker, "wav")):
                os.makedirs(os.path.join(self.root_path, "Preprocessed_data", self.speaker, "wav"))
            sf.write(os.path.join(self.root_path, "Preprocessed_data", self.speaker, "wav", self.EMA_files[k] + ".wav"), new_wav, self.sampling_rate_wav_wanted)
    
            ema_smooth = self.low_pass_butter(new_ema, self.sampling_rate_ema)
            ema_norm = scipy.stats.zscore(ema_smooth, axis=0)
            ema_all.append({'id': self.EMA_files[k], 'data': ema_norm})

        save_path = os.path.join(self.root_path, "Preprocessed_data", self.speaker, 'ema.pkl')
        with open(save_path, 'wb') as f:
              pickle.dump(ema_all, f)


    def trim_wav(self):
        N = len(self.EMA_files)

        for k in range(N):
        #在已经有wav的情况下，想要按照remove_silence的原则重新读取并保存去除静音的wav
            try:
                data = sio.loadmat(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))[self.EMA_files[k]][0]
            except:
                print('Could not read from {}'.format(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))) #HY
                continue

            wav, sr = librosa.load(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav",
                                            self.EMA_files[k] + ".wav"), sr=self.sampling_rate_wav_wanted)
            dur = librosa.get_duration(y=wav, sr=sr)

            xtrm = detect_silence(data)
            marge = 0.1
            xtrm = [max(xtrm[0] - marge, 0), xtrm[1] + marge]

            xtrm_temp_wav = [int(np.floor(xtrm[0] * self.sampling_rate_wav_wanted)),
                         int(min(np.floor(xtrm[1] * self.sampling_rate_wav_wanted) + 1, len(wav)))]
            new_wav = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]

            if not os.path.exists(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav")):
                os.makedirs(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav"))
            sf.write(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav", self.EMA_files[k] + ".wav"), new_wav, self.sampling_rate_wav_wanted)
    

    def check_alignment(self):
        N = len(self.EMA_files)
        if self.N_max != 0:
            N = self.N_max

        for i in range(N):
            path_wav = os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker,
                                              "wav", self.EMA_files[i] + '.wav')
            wav, sr = librosa.load(path_wav, sr=None)
            ema = np.load(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i] + ".npy"))
            print('EMA len {}, WAV len {}, EMA time {}, WAV time {}'.format(len(ema), len(wav),
                    len(ema)/self.sampling_rate_ema, len(wav)/self.sampling_rate_wav))
        
def Preprocessing_general_haskins(N_max, path_to_raw):
    """
    :param N_max: #max of files to treat (0 to treat all files), useful for test
    go through all the speakers of Haskins
    """
    corpus = 'Haskins'
    speakers_Has = get_speakers_per_corpus(corpus)
    for sp in speakers_Has :
        print("In progress Haskins ",sp)
        speaker = Speaker_Haskins(sp, path_to_raw=path_to_raw, N_max=N_max)
        speaker.Read_general_speaker()
        speaker.Preprocessing_general_speaker()
        print("Done Haskins ",sp)


if __name__ == "__main__":

    spks = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
    for spk in spks:
        print('Processing', spk)
        speaker = Speaker_Haskins(spk, path_to_raw="/nvme/yun/aai/")
        speaker.read_ema_and_wav()
