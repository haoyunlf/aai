import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

import os
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForCTC
import pickle

def extract_pretrained(root_path, spks, model_name, layer_range):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name in ["facebook/wav2vec2-base"]:
        layer_number = 13
    elif model_name in ["facebook/wav2vec2-large", "facebook/wav2vec2-large-960h", "facebook/mms-300m"]:
        layer_number = 25
    elif model_name in ["facebook/mms-1b", "facebook/mms-1b-all"]:
        layer_number = 49

    if model_name == "facebook/mms-1b-all-nld":
        layer_number = 49
        model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="nld", ignore_mismatched_sizes=True)
        feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b-all")
    else:
        model = Wav2Vec2Model.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    print(device)
    print(model)
    model.to(device)

    for spk in spks:
        wav_dir = os.path.join(root_path, spk, 'wav')
        pkl_output = [[] for _ in range(layer_number)]
        for wav_name in tqdm(sorted(os.listdir(wav_dir))):
            if os.path.splitext(wav_name)[-1] == '.wav':
                wav_path = os.path.join(wav_dir, wav_name)
                wav, sr = librosa.load(wav_path, sr=16000)

                input = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                with torch.no_grad():
                    outputs = model(input, output_hidden_states=True).hidden_states

                for i, layer in enumerate(outputs):
                    name = model_name.split('/')[-1] + "_layer_" + str(i)
                    emb = layer.squeeze().detach().cpu().numpy()
                    pkl_output[i].append({'id': os.path.splitext(wav_name)[0], 'data': emb})

        print('\n')
        if not os.path.exists(os.path.join(root_path, spk, "feature")):
            os.makedirs(os.path.join(root_path, spk, "feature"))
        for i in layer_range:
            save_path = os.path.join(root_path, spk, "feature", model_name.split('/')[-1]+'_layer_'+str(i)+'.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(pkl_output[i], f)
            print('Write to', save_path)
               
#
if __name__ == "__main__":
    #main()
    root_path = '/data/data2/yun/aai_expr/Preprocessed_data'
    spks = ["F01", "F02", "F03", "M01", "M02", "M03"]
    model_name = "facebook/mms-1b-all-nld"
    layer_range = range(41, 49)
    extract_pretrained(root_path, spks, model_name, layer_range)
