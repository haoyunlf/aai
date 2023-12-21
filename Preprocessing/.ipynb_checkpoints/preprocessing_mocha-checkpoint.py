#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    script to read data from the mocha database.
    It's free and available here "http://data.cstr.ed.ac.uk/mocha", we used the following speakers :
    "fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"
    data for speaker X has to be in "Raw_data/X" and for each sentence have 2 files ( .ema, .wav) and .lab
    if available.
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import scipy.signal
import scipy.interpolate
from Preprocessing.tools_preprocessing import get_fileset_names, get_delta_features, split_sentences

from os.path import dirname
import numpy as np
import scipy.signal

import scipy.interpolate
import librosa
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
from Preprocessing.class_corpus import Speaker
import glob

root_path = dirname(dirname(os.path.realpath(__file__)))

import time
import matplotlib.pyplot as plt

class Speaker_mocha(Speaker):
    """
    class for the speaker of MNGU0, child of the Speaker class (in class_corpus.py),
    then inherits of some preprocessing scripts and attributes
    """
    def __init__(self, sp, path_to_raw, N_max=0 ):
        """
        :param sp:  name of the speaker
        :param N_max:  # max of files we want to preprocess (0 is for All files), variable useful for test
        """
        super().__init__(sp)  # gets the attributes of the Speaker class
        self.root_path = path_to_raw
        self.N_max = N_max
        self.path_files_treated = os.path.join(root_path, "Preprocessed_data", self.speaker)
        self.path_files_brutes = os.path.join(self.root_path, "Raw_data", "mocha", self.speaker)
        self.path_files_hy = os.path.join(root_path, "Preprocessed_data_HY", self.speaker) #HY

        self.EMA_files = sorted([name for name in os.listdir(self.path_files_brutes) if "palate" not in name])
        self.EMA_files = sorted([name[:-4] for name in self.EMA_files if name.endswith('.ema')])
        self.n_columns = 20
        self.wav_files = sorted([name[:-4] for name in os.listdir(self.path_files_brutes) if name.endswith('.wav')])
        self.sp_with_trans = ["fsew0", "msak0", "mjjn0", "ffes0"] #speakers for which we have transcription ( ie we can remove silence)


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
        if not os.path.exists(os.path.join(self.path_files_hy, "ema_final")):
            os.makedirs(os.path.join(self.path_files_hy, "ema_final"))

        if not os.path.exists(os.path.join(self.path_files_brutes, "wav_cut")):
            os.makedirs(os.path.join(self.path_files_brutes, "wav_cut"))
        files = glob.glob(os.path.join(self.path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "mfcc", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "ema_final", "*"))
        files += glob.glob(os.path.join(self.path_files_brutes, "wav_cut", "*"))

        for f in files:
            os.remove(f)

    def read_ema_file(self,k):
        """
        read the ema file, first preprocessing,
        :param k: utterance index (wrt the list "EMA_files")
        :return: npy array (K,12 or 14) , K depends on the duration of the recording, 12 trajectories (or 14 when
        the velum is provided , ie for 4 speakers)
        """
        path_ema_file = os.path.join(self.path_files_brutes, self.EMA_files[k] + ".ema")
        with open(path_ema_file, 'rb') as ema_annotation:
            column_names = [0] * self.n_columns
            for line in ema_annotation:
                line = line.decode('latin-1').strip("\n")
                if line == 'EST_Header_End':
                    break
                elif line.startswith('NumFrames'):
                    n_frames = int(line.rsplit(' ', 1)[-1])
                    #print('n_frames', n_frames)
                elif line.startswith('Channel_'):
                    col_id, col_name = line.split(' ', 1)
                    column_names[int(col_id.split('_', 1)[-1])] = col_name.replace(" ",
                                                                                   "")  # v_x has sometimes a space
            ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, -1)
            #print(ema_data.shape)
            #print('Time in seconds:', ema_data[-5:])
            # #print(column_names) #HY
            
            # cols_index = [column_names.index(col) for col in self.articulators]
            # ema_data = ema_data[:, cols_index]
            # ema_data = ema_data / 100  # met en mm, initallement en 10^-1m

            # if np.isnan(ema_data).sum() != 0:
            #     print("nombre de nan ", np.isnan(ema_data).sum())
            #     # Build a cubic spline out of non-NaN values.
            #     spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(ema_data).ravel()),
            #                                       ema_data[~np.isnan(ema_data)], k=3)
            #     # Interpolate missing values and replace them.
            #     for j in np.argwhere(np.isnan(ema_data)).ravel():
            #         ema_data[j] = scipy.interpolate.splev(j, spline)
            return ema_data

    def remove_silences(self,ema, mfcc, k):
        """
          :param k:  utterance index (wrt the list EMA_files)
          :param ema: the ema list of traj
          :param mfcc: the mfcc features
          :return: the data (ema and mfcc) without the silence at the beginning and end of the recording
         For some speakers we have annotation that gives in second when the voice starts and ends.
         For those speakers : get those extremity , calculates the equivalence in # of ema points and # of mfcc frames
         add by HY: for other speakers, do it by using librosa.effects.trim
          """
        marge = 0.1  # original: 0
        if self.speaker in self.sp_with_trans:
            path_annotation = os.path.join(self.path_files_brutes, self.wav_files[k] + '.lab')
            with open(path_annotation) as file:
                labels = [
                    row.strip('\n').strip('\t').replace(' 26 ', '').split(' ')
                    for row in file
                ]
            xtrm = [max(float(labels[0][1]) - marge, 0), float(labels[-1][0]) + marge]
            xtrm_temp_ema = [int(xtrm[0] * self.sampling_rate_ema),
                             min(int((xtrm[1] * self.sampling_rate_ema) + 1), len(ema))]

            xtrm_temp_mfcc = [int(xtrm[0] / self.hop_time),
                              int(np.ceil(xtrm[1] / self.hop_time))]

            mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]

            ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
        else:
        #TODO add trim later 正在改
            print("Remove silence based on trim")
            print("MFCC sr: {}, EMA sr: {}".format(1/self.hop_time, self.sampling_rate_ema))
            path_wav = os.path.join(self.path_files_brutes, self.wav_files[k] + '.wav')
            y, sr = librosa.load(path_wav)
            dur = librosa.get_duration(y=y, sr=sr)
            signal, index = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=128)
            xtrm = [max(0, index[0]/sr - marge), min(index[1]/sr + marge, dur)]
            print("Trimmed time interval:", xtrm)
            xtrm_temp_mfcc = [int(xtrm[0] / self.hop_time), int(np.ceil(xtrm[1] / self.hop_time))]
            xtrm_temp_ema = [int(xtrm[0] * self.sampling_rate_ema), min(int(np.ceil(xtrm[1] * self.sampling_rate_ema)), len(ema))]
            print(xtrm_temp_mfcc, xtrm_temp_ema)
            print("Original mfcc:", mfcc.shape)
            mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]
            print("Trimmed mfcc:", mfcc.shape)
            print("Orignial ema:", ema.shape)
            ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
            print("Trimmed ema:", ema.shape)
        return ema, mfcc

    def from_wav_to_mfcc(self,wav):
        """
       :param wav: list of intensity points of the wav file
       :return: the acoustic features( K,429); where K in the # of frames.
       calculations of the mfcc with librosa , + Delta and DeltaDelta, + 10 context frames
       # of acoustic features per frame: 13 ==> 13*3 = 39 ==> 39*11 = 429.
       parameters for mfcc calculation are defined in class_corpus
       """
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sampling_rate_wav, n_mfcc=self.n_coeff,
                                       n_fft=self.frame_length, hop_length=self.hop_length
                                       ).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((self.window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * self.window
        mfcc = np.concatenate([frames[j:j + len(mfcc)] for j in range(full_window)], axis=1)  # add context
        return mfcc

    def Read_general_speaker(self): #HY
        self.create_missing_dir()

        N = len(self.EMA_files)
        if self.N_max != 0:
            N = self.N_max

        for i in range(N):
            ema = self.read_ema_file(i)
            #print('a', ema.shape, len(ema)/self.sampling_rate_ema)#HY
            np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i]), ema) #HY

    def Check_general_speaker(self): #HY
        self.create_missing_dir()

        N = len(self.EMA_files)
        if self.N_max != 0:
            N = self.N_max

        print('Ema_len\tEma_len/Ema_sr\tEma_start_sec\tEma_end_sec\tEma_dur_sec\tWav_sec\tDiff(Wav_sec-Ema_len/EMA_sr)\tDiff(Wav_sec-Ema_dur)')
        for i in range(N):
            ema = self.read_ema_file(i)
            path_wav = os.path.join(self.path_files_brutes, self.wav_files[i] + '.wav')
            wav, sr = librosa.load(path_wav, sr=None)
            dur = librosa.get_duration(y=wav, sr=sr)
            print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4}\t{:.4}\t{:.4}\t{:.4}'.format(len(ema), len(ema)/500, ema[0][0], ema[-1][0], ema[-1][0]-ema[0][0], dur,
                                                                dur - len(ema)/500, dur - (ema[-1][0]-ema[0][0])))#HY

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
        #self.create_missing_dir()

        N = len(self.EMA_files)
        if self.N_max != 0:
            N = self.N_max

        for i in range(N):
            # ema = self.read_ema_file(i)
            # np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i]), ema) #HY
            ema = np.load(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema", self.EMA_files[i] + ".npy"))
            print('b', ema.shape, len(ema)/self.sampling_rate_ema)
            ema_smooth = self.low_pass_butter(ema) #HY: using scipy butter low-pass filter
            print('c', ema_smooth.shape, len(ema_smooth)/self.sampling_rate_ema)

            # add display option
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
            
            path_wav = os.path.join(self.path_files_brutes, self.wav_files[i] + '.wav')
            wav, sr = librosa.load(path_wav, sr=None)
            dur = librosa.get_duration(y=wav, sr=sr)
            print('wav dur: {}'.format(dur))
            wav = 0.5 * wav / np.max(wav)
            mfcc = self.from_wav_to_mfcc(wav)
            print('EMA sr {}, WAV sr {} {}, EMA len {}, WAV len {}, EMA time {}, WAV time {}'.format(
                self.sampling_rate_ema, self.sampling_rate_wav, sr, len(ema_smooth), len(wav),
                len(ema_smooth)/self.sampling_rate_ema, len(wav)/self.sampling_rate_wav))

            if display:
                marge = 0.1
                times = np.arange(len(ema_ttx))/self.sampling_rate_ema
                #times2 = np.arange(len(wav))/self.sampling_rate_wav        
                print(len(times), len(wav), len(ema_smooth[:, 0]))
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
                #ax1.plot(times, ema_ttx)
                librosa.display.waveshow(wav, sr=sr, ax=ax1, color="blue")
                ax1.set_title("Original Wav: "+ self.EMA_files[i])
                ax1.margins(0, .1)
                ax1.grid(alpha=.5, ls='--')
                ax2.plot(times, ema_smooth[:, 0])
                ax2.set_title("Filtered EMA: " + self.EMA_files[i] + ' TT_x')
                ax2.grid(alpha=.5, ls='--')
                plt.tight_layout()
                plt.show()
                plt.savefig("before_remove_silence.png") 
                
            ema_smooth, mfcc = self.remove_silences(ema_smooth, mfcc, i)
            
            if display:
                ema_ttx = ema_smooth[:, 0]
                dur = librosa.get_duration(y=wav, sr=sr)
                signal, index = librosa.effects.trim(wav, top_db=30, frame_length=512, hop_length=128)
                xtrm = [max(0, index[0]/sr - marge), min(index[1]/sr + marge, dur)]
                times = np.arange(len(ema_ttx))/self.sampling_rate_ema
                print(len(times), len(signal), len(ema_smooth[:, 0]))
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3))
                #ax1.plot(times, ema_ttx)
                librosa.display.waveshow(signal, sr=sr, ax=ax1, color="blue")
                ax1.set_title("Trimmed Wav: " + self.EMA_files[i])
                ax1.margins(0, .1)
                ax1.grid(alpha=.5, ls='--')
                ax2.plot(times, ema_smooth[:, 0])
                ax2.set_title("Trimmed EMA: " + self.EMA_files[i] + ' TT_x')
                ax2.grid(alpha=.5, ls='--')
                plt.tight_layout()
                plt.show()
                plt.savefig("after_remove_silence.png") 

            ema_smooth, mfcc = self.synchro_ema_mfcc(ema_smooth, mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "ema_final", self.EMA_files[i]), ema_smooth) #HY
            np.save(os.path.join(root_path, "Preprocessed_data_HY", self.speaker, "mfcc", self.EMA_files[i]), mfcc) #HY
        '''HY
            ema_VT = self.add_vocal_tract(ema)
            ema_VT_smooth = self.smooth_data(ema_VT)  # smooth for a better calculation of norm values
            path_wav = os.path.join(self.path_files_brutes, self.wav_files[i] + '.wav')
            wav, sr = librosa.load(path_wav, sr=None)
            wav = 0.5 * wav / np.max(wav)
            mfcc = self.from_wav_to_mfcc(wav)
            ema_VT_smooth, mfcc = self.remove_silences(ema_VT_smooth, mfcc, i)
            ema_VT_smooth, mfcc = self.synchro_ema_mfcc(ema_VT_smooth, mfcc)

            ema_VT, rien = self.remove_silences(ema_VT, mfcc, i)
            ema_VT, rien = self.synchro_ema_mfcc(ema_VT, mfcc)

            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema", self.EMA_files[i]), ema_VT)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i]), ema_VT_smooth)
            self.list_EMA_traj.append(ema_VT_smooth)
            self.list_MFCC_frames.append(mfcc)

        self.calculate_norm_values()

        for i in range(N):
            ema_pas_smooth = np.load(
                os.path.join(root_path, "Preprocessed_data", self.speaker, "ema", self.EMA_files[i] + ".npy"))
            ema_VT_smooth = np.load(
                os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i] + ".npy"))
            mfcc = np.load(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i] + ".npy"))
            ema_VT_smooth_norma, mfcc = self.normalize_sentence(i, ema_VT_smooth, mfcc)
            ema_pas_smooth_norma, rien = self.normalize_sentence(i, ema_pas_smooth, mfcc)
            new_sr = 1 / self.hop_time   # we did undersampling of ema traj for 1 point per frame mfcc
                                        # so about 1 point every hoptime sec.
            ema_VT_smooth_norma = self.smooth_data(ema_VT_smooth_norma, new_sr)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema", self.EMA_files[i]), ema_pas_smooth_norma)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i]),
                    ema_VT_smooth_norma)

        #  split_sentences(speaker)   #possibility to cut to long sentences
        get_fileset_names(self.speaker)
        '''
        
def Preprocessing_general_mocha(N_max, path_to_raw):
    """
    :param N_max: #max of files to treat (0 to treat all files), useful for test
    go through all the speakers of Haskins
    """
    corpus = 'mocha'
    speakers_mocha = get_speakers_per_corpus(corpus)
    for sp in speakers_mocha :
        print("In progress mocha ",sp)
        speaker = Speaker_mocha(sp,path_to_raw=path_to_raw, N_max=N_max)
        speaker.Preprocessing_general_speaker()
        print("Done mocha ",sp)


#Test :
#Preprocessing_general_mocha(N_max=50)
if __name__ == "__main__":
    sp = "msak0"
    speaker = Speaker_mocha(sp,path_to_raw="/home/yun/DATASETS/ema/", N_max=0)
    speaker.Check_general_speaker()
    # corpus = 'mocha'
    # speakers_mocha = get_speakers_per_corpus(corpus)
    # for sp in speakers_mocha :
    #     #print("In progress mocha ",sp)
    #     speaker = Speaker_mocha(sp,path_to_raw="/home/yun/DATASETS/ema/", N_max=1)
    #     speaker.Read_general_speaker()
    #     speaker.Preprocessing_general_speaker(display=True)
    #     #speaker.check_alignment()
    #     #print("Done mocha ",sp)