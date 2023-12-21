import os
import librosa
from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Processor
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

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

    wavfiles = sorted([name[:-4] for name in os.listdir(args.input_dir) if name.endswith('.wav') and "palate" not in name])
    print('Input dir:', args.input_dir)
    print(f"Featurizing {len(wavfiles):,} wav files")
    output_dir = os.path.join(args.output_dir, args.model)
    print('Output dir:', output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    for wavfile in tqdm(wavfiles, ncols=80):
        wav_path = os.path.join(args.input_dir, wavfile+'.wav')
        print(wav_path)
        y, rate = librosa.load(wav_path)
        input = librosa.resample(y, orig_sr=rate, target_sr=16000)
        input = processor(input.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        feat_path = os.path.join(args.output_dir, model_name)
        if torch.cuda.is_available():
            input = input.cuda()

        with torch.no_grad():
            hidden_state = model(input).last_hidden_state.squeeze(0).detach().cpu().numpy()

        np.save(feat_path, hidden_state, wavfile)

if __name__ == "__main__":
    main()