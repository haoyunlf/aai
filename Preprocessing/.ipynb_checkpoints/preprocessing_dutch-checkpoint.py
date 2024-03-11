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

def scale_intensity(wav, ref_rms):
    "Scale the test wav intensity based on reference rms"
    return ref_rms/np.max(librosa.feature.rms(y=wav), axis=1) * wav
    
def low_pass_butter(ema, sr):
    """
    HY: added another smoothing method, straightforwardly based on scipy butterworth filter
    :param ema: one ema trajectory
    :param sr: sampling rate of the ema trajectory
    :return:  the smoothed ema trajectory
    """
    cutoff = 10
    sos = scipy.signal.butter(5, cutoff, 'lowpass', fs=sr, output='sos')
    filtered_data = np.concatenate([np.expand_dims(scipy.signal.sosfiltfilt(sos, col), axis=0) for col in ema.T],
                                   axis=0).T
    return filtered_data

def filter_data(dir):
    "Select the test data based on file name (only words and TIMIT sentences)"
    outputs = []
    for file in os.listdir(dir):
        name, ext = os.path.splitext(file)[0], os.path.splitext(file)[-1]
        if ext == '.wav' and os.path.exists(os.path.join(dir, name+'.mat')):
            labels = name.split('_')
            #print(labels)
            if labels[2].isupper():
                outputs.append(file)
              #print(labels)
            elif labels[2].find('sen') == 0:
                if labels[3].find('TIMIT') == 0:
                    outputs.append(file)
    
    return outputs

def read_ema_and_wav(input_path, output_path, spks, channel=0, display=False):
    """
    channel: the wanted channel in the wav file
    Save preprocessed wav and ema.pkl
    """
    order_arti_dutch = ['tb_x', 'tb_y', 'tt_x', 'tt_y', 'li_x', 'li_y', 'ul_x', 'ul_y', "ll_x", "ll_y",
                        'pl_x', 'pl_y', 'ft_x', 'ft_y', "ml_x", "ml_y", "mr_x", "mr_y", "ui_x", "ui_y"]

    # order_arti = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
    #               'ul_x', 'ul_y', 'll_x', 'll_y']
    order_arti = ['tt_x', 'tt_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                  'ul_x', 'ul_y', 'll_x', 'll_y']

    sampling_rate_ema = 400
    marge = 0.1
    ref_intensity = 0.1476

    for spk in spks:
      ema_all = []

      for file in filter_data(os.path.join(input_path, spk)):
          name, ext = os.path.splitext(file)[0], os.path.splitext(file)[-1]
          try:
              data = sio.loadmat(os.path.join(input_path, spk, name+'.mat'))['mf'][0]
          except:
              print('Could not read from {}'.format(os.path.join(input_path, spk, file)))
              continue #

          ema = np.zeros((len(data[1][2]), len(order_arti_dutch)))

          for arti in range(1, len(data)):
              ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
              ema[:, arti * 2 - 1] = data[arti][2][:, 2]
          new_order_arti = [order_arti_dutch.index(col) for col in order_arti]
          ema = ema[:, new_order_arti]

          wav_data, sr = librosa.load(os.path.join(input_path, spk, name+'.wav'), sr=16000, mono=False)
          wav = wav_data[channel]
          new_wav = scale_intensity(wav, ref_intensity)

          signal, index = librosa.effects.trim(new_wav, top_db=30, frame_length=512, hop_length=128)
          xtrm = index / sr
          xtrm = [max(0,  xtrm[0] - marge), xtrm[1] + marge]

          xtrm_temp_ema = [int(np.floor(xtrm[0] * sampling_rate_ema)),
                          int(min(np.floor(xtrm[1] * sampling_rate_ema) + 1, len(ema)))]
          xtrm_temp_wav = [int(np.floor(xtrm[0] * sr)),
                          int(min(np.floor(xtrm[1] * sr) + 1, len(wav_data[channel])))]
          new_wav_data = new_wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]
          ema_data = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]

          ema_smooth = low_pass_butter(ema_data, sampling_rate_ema)
          ema_norm = scipy.stats.zscore(ema_smooth, axis=0)
          ema_all.append({'id': name, 'data': ema_norm})

          if not os.path.exists(os.path.join(output_path, spk, "wav")):
              os.makedirs(os.path.join(output_path, spk, "wav"))
          sf.write(os.path.join(output_path, spk, "wav", name + ".wav"), new_wav_data, sr)

          save_path = os.path.join(output_path, spk, 'ema.pkl')
          with open(save_path, 'wb') as f:
            pickle.dump(ema_all, f)


if __name__ == "__main__":
    spks = ["SPOKECS01", "SPOKECS03"]
    root_path = "/nvme/yun/aai/Raw_data"
    output_path = "/nvme/yun/aai/Preprocessed_data"
    read_ema_and_wav(root_path, output_path, spks)
    read_ema_and_wav(root_path, output_path, ["pilot_ema_data"], channel=1)
