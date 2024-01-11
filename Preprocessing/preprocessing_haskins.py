#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    script to read data from the Haskins database
    It's free and available here "https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h/folder/30415804819"
    There are 8 speakers, the rawfiles for the speaker X need to be in "Raw_data/Haskins/X".
    For one sentence the acoustic & arti data are in a matlab file "data".
    The extraction&preprocessing are done for one speaker after the other.
    For one sentence of speaker X the script saves 3 files in "Preprocessed_data/X"  : mfcc (K,429), ema (K,18),
    ema_final (K,18) [same as ema but normalized]; where K depends on the duration of the recording
    The script write a .wav file in "Raw_data/Haskins/X/wav".
    The script also saves for each the "norm values" [see class_corpus, calculate_norm_values()]
"""
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
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self.path_files_hy = os.path.join(root_path, "Preprocessed_data_HY", self.speaker)
        self.EMA_files = sorted([name[:-4] for name in os.listdir(self.path_files_brutes) if "palate" not in name])
        self.N_max = N_max


    def create_missing_dir(self):
        """
        delete all previous preprocessing, create needed directories
        """
        if not os.path.exists(os.path.join(self.path_files_treated, "ema")):
            os.makedirs(os.path.join(self.path_files_treated, "ema"))
        if not os.path.exists(os.path.join(self.path_files_treated, "mfcc")):
            os.makedirs(os.path.join(self.path_files_treated, "mfcc"))
        if not os.path.exists(os.path.join(self.path_files_treated, "ema_final")):
            os.makedirs(os.path.join(self.path_files_treated, "ema_final"))

        #HY
        if not os.path.exists(os.path.join(self.path_files_hy, "ema")):
            os.makedirs(os.path.join(self.path_files_hy, "ema"))
        if not os.path.exists(os.path.join(self.path_files_hy, "mfcc")):
            os.makedirs(os.path.join(self.path_files_hy, "mfcc"))
        if not os.path.exists(os.path.join(self.path_files_hy, "mfcc_final")):
            os.makedirs(os.path.join(self.path_files_hy, "mfcc_final"))
        if not os.path.exists(os.path.join(self.path_files_hy, "ema_final")):
            os.makedirs(os.path.join(self.path_files_hy, "ema_final"))

        # We are going tp create the wav files form the matlab format files given for haskins
        if not os.path.exists(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav")):
            os.makedirs(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav"))

        files = glob.glob(os.path.join(self.path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "mfcc", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "ema_final", "*"))
        files += glob.glob(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav", "*"))
        for f in files:
            os.remove(f)

    def read_ema_and_wav(self, k, display=False):
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

        # add exception by HY
        try:
            data = sio.loadmat(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))[self.EMA_files[k]][0]
        except:
            print('Could not read from {}'.format(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))) #HY
            return None, None

        ema = np.zeros((len(data[1][2]), len(order_arti_haskins)))

        for arti in range(1, len(data)):  # lecture des trajectoires articulatoires dans le dicionnaire
            ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
            ema[:, arti * 2 - 1] = data[arti][2][:, 2]
        new_order_arti = [order_arti_haskins.index(col) for col in order_arti]
        ema = ema[:, new_order_arti]

        # We create wav files form intensity matlab files
        wav_data = data[0][2][:, 0]
        sf.write(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker,
                                              "wav", self.EMA_files[k] + ".wav"), wav_data, self.sampling_rate_wav)
        # print('wav saved to', os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav",
        #                                     self.EMA_files[k] + ".wav"))
        wav, sr = librosa.load(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav",
                                            self.EMA_files[k] + ".wav"), sr=self.sampling_rate_wav_wanted)
        # np.save(os.path.join(root_path, "Raw_data", corpus, speaker, "wav",
        #                      EMA_files[k]), wav)
        wav = 0.5 * wav / np.max(wav)
        dur = librosa.get_duration(y=wav, sr=sr)
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sampling_rate_wav_wanted, n_mfcc=self.n_coeff,
                                       n_fft=self.frame_length, hop_length=self.hop_length).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((self.window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * self.window
        mfcc = np.concatenate([frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)

        marge = 0.1 #original:0
        xtrm = detect_silence(data)
        xtrm = [max(xtrm[0] - marge, 0), xtrm[1] + marge]

        xtrm_temp_ema = [int(np.floor(xtrm[0] * self.sampling_rate_ema)),
                         int(min(np.floor(xtrm[1] * self.sampling_rate_ema) + 1, len(ema)))]
        xtrm_temp_mfcc = [int(np.floor(xtrm[0] / self.hop_time)),
                          int(np.ceil(xtrm[1] / self.hop_time))]
        ema_cut = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
        mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]
        # print(ema_cut.shape[0]/self.sampling_rate_ema, mfcc.shape[0]*self.hop_time) <- is the same

        n_frames_wanted = mfcc.shape[0]
        ema_cut = scipy.signal.resample(ema_cut, num=n_frames_wanted)
        # HY: for this corpus the resampled rate is not changed, because the sr of ema and mfcc are the same

        if display:
            #times1 = np.arange(dur)
            #times1 = np.arange(len(wav))/self.sampling_rate_wav    
            times1 = np.arange(len(ema[:, 0]))/self.sampling_rate_ema
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
            #ax1.plot(times, ema_ttx)
            librosa.display.waveshow(wav, sr=self.sampling_rate_wav_wanted, ax=ax1, color="blue")
            ax1.set_title("Original Wav: "+ self.EMA_files[k])
            ax1.margins(0, .1)
            ax1.grid(alpha=.5, ls='--')
            ax2.plot(times1, ema[:, 0])
            ax2.set_title("Orinigal EMA: " + self.EMA_files[k] + ' TT_x')
            ax2.grid(alpha=.5, ls='--')
            plt.tight_layout()
            plt.show()
            plt.savefig("before_remove_silence.png") 

            #times2 = np.arange(xtrm)
            times2 = np.arange(len(ema_cut[:, 0]))/self.sampling_rate_ema
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
            new_wav = wav[int(xtrm[0] * self.sampling_rate_wav_wanted) : int(xtrm[1] * self.sampling_rate_wav_wanted)]
            librosa.display.waveshow(new_wav, sr=self.sampling_rate_wav_wanted, ax=ax1, color="blue")
            ax1.set_title("Cut Wav: "+ self.EMA_files[k])
            ax1.margins(0, .1)
            ax1.grid(alpha=.5, ls='--')
            ax2.plot(times2, ema_cut[:, 0])
            ax2.set_title("Cut EMA: " + self.EMA_files[k] + ' TT_x')
            ax2.grid(alpha=.5, ls='--')
            plt.tight_layout()
            plt.show()
            plt.savefig("after_remove_silence.png") 
            
        return ema_cut, mfcc

    def trim_wav(self):
        N = len(self.EMA_files)

        for k in range(N):
        #在已经有wav的情况下，想要按照remove_silence的原则重新读取并保存去除静音的wav
            try:
                data = sio.loadmat(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))[self.EMA_files[k]][0]
            except:
                print('Could not read from {}'.format(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))) #HY
                continue

            # if os.path.isfile(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav", self.EMA_files[k] + ".wav")):
            #     print('wav {} exists'.format(self.EMA_files[k] + ".wav"))
            #     continue
            wav, sr = librosa.load(os.path.join(self.root_path, "Raw_data", self.corpus, self.speaker, "wav",
                                            self.EMA_files[k] + ".wav"), sr=self.sampling_rate_wav_wanted)
            dur = librosa.get_duration(y=wav, sr=sr)

            xtrm = detect_silence(data)
            marge = 0.1
            xtrm = [max(xtrm[0] - marge, 0), xtrm[1] + marge]

            xtrm_temp_wav = [int(np.floor(xtrm[0] * self.sampling_rate_wav_wanted)),
                         int(min(np.floor(xtrm[1] * self.sampling_rate_wav_wanted) + 1, len(wav)))]
            new_wav = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]

            #print(len(wav), len(new_wav), len(new_wav)/self.sampling_rate_wav_wanted)
            if not os.path.exists(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav")):
                os.makedirs(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav"))
            sf.write(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "wav", self.EMA_files[k] + ".wav"), new_wav, self.sampling_rate_wav_wanted)
                            
            
    def Read_general_speaker(self, display=False): #HY
        self.create_missing_dir()
        N = len(self.EMA_files)
        if self.N_max != 0:
            N = int(self.N_max)

        for i in range(N):
            ema, mfcc = self.read_ema_and_wav(i, display=display)
            if ema is not None:
                np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i]), ema)
                np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
                
    def Preprocessing_general_speaker(self, display=False):
        """
        Go through the sentences one by one.
            - reads ema data and turn it to a (K,18) array where arti are in a precise order, interploate missing values,
        smooth the trajectories, remove silences at the beginning and the end, undersample to have 1 position per
        frame mfcc, add it to the list of EMA traj for this speaker
            - reads the wav file, calculate the associated acoustic features (mfcc+delta+ deltadelta+contextframes) ,
        add it to the list of the MFCC FEATURES for this speaker.
        Then calculate the normvalues based on the list of ema/mfcc data for this speaker
        Finally : normalization and last smoothing of the trajectories.
        Final data are in Preprocessed_data/speaker/ema_final.npy and  mfcc.npy
        """

        N = len(self.EMA_files)
        if self.N_max != 0:
            N = int(self.N_max)

        for i in range(N):
            if os.path.isfile(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i] + ".npy")):
                ema = np.load(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i] + ".npy"))
                ema_smooth = self.low_pass_butter(ema) #HY: using scipy butter low-pass filter
                ema_norm = scipy.stats.zscore(ema_smooth, axis=0)
                np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema_final",
                                self.EMA_files[i]), ema_norm)
                mfcc = np.load(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "mfcc", self.EMA_files[i] + ".npy"))
                mfcc_norm = scipy.stats.zscore(mfcc, axis=0)
                #print(True if len(mfcc_norm) == len(ema_norm) else False)
                np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "mfcc_final",
                                self.EMA_files[i]), mfcc_norm)
                
                if display:
                    ema_ttx = ema[:, 0]
                    times = np.arange(len(ema_ttx))/self.sampling_rate_ema
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
                    ax1.plot(times, ema_ttx)
                    ax1.set_title("Original Signal: " + self.EMA_files[i] + ' TT_x')
                    ax1.margins(0, .1)
                    ax1.grid(alpha=.5, ls='--')
                    ax2.plot(times, ema_smooth[:, 0])
                    ax2.set_title("Low-Pass Filter (10 Hz): " + self.EMA_files[i] + ' TT_x')
                    ax2.grid(alpha=.5, ls='--')
                    plt.tight_layout()
                    plt.show()
                    plt.savefig("filters.png") 
            '''HY
            ema_VT = self.add_vocal_tract(ema)
            ema_VT_smooth = self.smooth_data(ema_VT)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema", self.EMA_files[i]), ema_VT)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final",
                                 self.EMA_files[i]), ema_VT_smooth)
            self.list_EMA_traj.append(ema_VT_smooth)
            self.list_MFCC_frames.append(mfcc)
        self.calculate_norm_values()

        for i in range(N):
            ema_VT_smooth = np.load(os.path.join(
                root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i] + ".npy"))
            mfcc = np.load(os.path.join(
                root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i] + ".npy"))
            ema_VT_smooth_norma, mfcc = self.normalize_sentence(i, ema_VT_smooth, mfcc)
            new_sr = 1 / self.hop_time
            ema_VT_smooth_norma = self.smooth_data(ema_VT_smooth_norma, new_sr)

            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i]),
                    ema_VT_smooth_norma)
        #  split_sentences(speaker)
        get_fileset_names(self.speaker)
        '''

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


#Test :
#Preprocessing_general_haskins(N_max=50)
if __name__ == "__main__":
    # sp = "F01"
    # speaker = Speaker_Haskins(sp,path_to_raw="/home/yun/DATASETS/ema/")
    # speaker.trim_wav()

    spks = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
    for spk in spks:
        print('Processing', spk)
        speaker = Speaker_Haskins(spk, path_to_raw="/home/yun/DATASETS/ema/")
        speaker.trim_wav()
    # speaker.Read_general_speaker()
    # speaker.Preprocessing_general_speaker()
    
    # speaker.check_alignment()
    # speaker.Preprocessing_general_speaker(display=True)
    # corpus = 'Haskins'
    # speakers_Has = get_speakers_per_corpus(corpus)
    # for sp in speakers_Has :
    #     #print("In progress mocha ",sp)
    #     speaker = Speaker_Haskins(sp,path_to_raw="/home/yun/DATASETS/ema/", N_max=0)
    #     speaker.check_alignment()