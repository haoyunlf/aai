import s3prl.hub as hub
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

import os
import librosa

def extract_haskins(root_path, spks, model, layers_range):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    for spk in spks:
        raw_wav_dir = os.path.join(raw_path, spk, 'wav')
        wav_dir = os.path.join(root_path, spk, 'wav')
        for wav_name in tqdm(sorted(os.listdir(wav_dir))):
            if os.path.splitext(wav_name)[-1] == '.wav':
                # print(wav_name)
                wav_path = os.path.join(wav_dir, wav_name)
                raw_wav_path = os.path.join(raw_wav_dir, wav_name)
                wav, sr = librosa.load(wav_path, sr=None)
                raw_wav, raw_sr = librosa.load(raw_wav_path, sr=None)
                dur = librosa.get_duration(y=wav, sr=sr)
                dur_raw = librosa.get_duration(y=raw_wav, sr=raw_sr)
                wav = [torch.from_numpy(wav).to(device)]
    
                with torch.no_grad():
                    output = model(wav)

                # print('ema, mfcc, emb, dur, raw_dur')
                for i in range(layers_range[0], layers_range[1] + 1):
                    layer = "hidden_state_" + str(i)
                    emb = output[layer][0].detach().cpu().numpy()
                    # info = output['_hidden_states_info']
                    # emb = output["hidden_states"]

                    # ### 确认是否与ema数据的长度对的上
                    # ema_path = os.path.join(root_path, spk, 'ema_final', os.path.splitext(wav_name)[0]+'.npy')
                    # mfcc_path = os.path.join(root_path, spk, 'mfcc_final', os.path.splitext(wav_name)[0]+'.npy')
                    # ema = np.load(ema_path)
                    # mfcc = np.load(mfcc_path)

                    # print(ema.shape[0]/100, mfcc.shape[0]/100, emb.shape[0]/50, dur, dur_raw)
                    
                    save_path = os.path.join(root_path, spk, model_name+'_layer_'+str(i))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    #print(save_path)
                    np.save(os.path.join(save_path, os.path.splitext(wav_name)[0]), emb)

def check_align(root_path, raw_path, spks):
    for spk in spks:
        raw_wav_dir = os.path.join(raw_path, spk, 'wav')
        wav_dir = os.path.join(root_path, spk, 'wav')
        for wav_name in tqdm(os.listdir(wav_dir)):
            if os.path.splitext(wav_name)[-1] == '.wav':
                wav_path = os.path.join(wav_dir, wav_name)
                raw_wav_path = os.path.join(raw_wav_dir, wav_name)
                wav, sr = librosa.load(wav_path, sr=None)
                raw_wav, raw_sr = librosa.load(raw_wav_path, sr=None)
                print('Wav:', sr, len(wav), len(wav)/sr)
                print('Raw wav:', raw_sr, len(raw_wav), len(raw_wav)/raw_sr)
                

def main():
    parser = ArgumentParser(
        prog="Wav2Vec2 Featurizer",
        description="Runs full featurization of wav files for downstream usage.",
    )
    parser.add_argument("-i", "--input_dir", default="../wav", type=str)
    parser.add_argument("-o", "--output_dir", default="../feats-seg/{model}/layer-{l}/{speaker}", type=str)
    parser.add_argument("-n", "--output_name", default="layer-{layer}.npy")
    parser.add_argument("-m", "--model", default="facebook/wav2vec2-base-960h", type=str)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(device)

    wavfiles = sorted([name[:-4] for name in os.listdir(args.input_dir) if name.endswith('.wav') and "palate" not in name])
    print('Input dir:', args.input_dir)
    print(f"Featurizing {len(wavfiles):,} wav files")
    output_dir = os.path.join(args.output_dir, args.model)
    print('Output dir:', output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model
    model = getattr(hub, model_name)()
    # layers = 
    
    if torch.cuda.is_available():
        model.to(device)

    for spk in spks:
        if not os.path.exists(os.path.join(output_dir, spk, model_name)):
            os.makedirs(os.path.join(output_dir, spk, model_name))
        wav_dir = os.path.join(input_dir, spk, 'wav')
        for wav_name in tqdm(os.listdir(wav_dir)):
          if os.path.splitext(wav_name)[-1] == '.wav':
            wav_path = os.path.join(wav_dir, wav_name)
            wav, sr = librosa.load(wav_path)
            wav = [torch.from_numpy(wav).to(device)]
        
            with torch.no_grad():
                emb = model(wav)["hidden_states"][0].detach().cpu().numpy()
        
                np.save()

if __name__ == "__main__":
    #main()
    root_path = '/data2/yun/aai_expr/Preprocessed_data_HY'
    raw_path = '/home/yun/DATASETS/ema/Raw_data/Haskins'
    spks = ["F02", "F03", "F04", "M01", "M02", "M03", "M04"]
    model_name = "wav2vec2"
    layers_range = [0, 12]
    model = getattr(hub, model_name)()
    extract_haskins(root_path, spks, model, layers_range)
    # check_align(root_path, raw_path, spks)