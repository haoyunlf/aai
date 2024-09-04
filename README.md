# Exploring Self-Supervised Speech Representations for Cross-lingual Acoustic-to-Articulatory Inversion

This is the code for Interspeech 2024 paper "Exploring Self-Supervised Speech Representations for Cross-lingual Acoustic-to-Articulatory Inversion".
Our code is partly based on https://github.com/bootphon/articulatory_inversion and https://github.com/karkirowle/articulatory_inversion

 > **Abstract**:Acoustic-to-articulatory inversion (AAI) is the process of inferring vocal tract movements from acoustic speech signals. Despite its diverse potential applications, AAI research in languages other than English is scarce due to the challenges of collecting articulatory data. In recent years, self-supervised learning (SSL) based representations have shown great potential for addressing low-resource tasks. We utilize wav2vec 2.0 representations and English articulatory data for training AAI systems and investigates their effectiveness for a different language: Dutch. Results show that using mms-1b features can reduce the cross-lingual performance drop to less than 30%. We found that increasing model size, selecting intermediate rather
than final layers, and including more pre-training data improved AAI performance. By contrast, fine-tuning on an ASR task did not. Our results therefore highlight promising prospects for implementing SSL in AAI for languages with limited articulatory data.

## Data
The English data (Haskins Production Rate Corpus) can be downloaded at https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h/folder/30415804819

The Dutch data is not publicly available.

## Usage
- Preprocessing/extract_ssl.py: extract pretrained ssl features
- Preprocessing/preprocessing_dutch.py: preprocessing Dutch EMA data
- Preprocessing/preprocessing_haskins.py: preprocessing English EMA data
- train_kf.py: train k-fold validation models

## Citation
```bash
@inproceedings{hao24c_interspeech,
  title     = {Exploring Self-Supervised Speech Representations for Cross-lingual Acoustic-to-Articulatory Inversion},
  author    = {Yun Hao and Reihaneh Amooie and Wietse {de Vries} and Thomas Tienkamp and Rik {van Noord} and Martijn Wieling},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4603--4607},
  doi       = {10.21437/Interspeech.2024-1740},
}
```
