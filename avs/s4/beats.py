import os
import sys
import json
import torch
import librosa
import pandas as pd
from config import cfg

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from beats_model.BEATs import BEATs, BEATsConfig


if __name__ == '__main__':
    print("beats start")

    with open(cfg.BEATS.ONTOLOGY, 'r') as f:
        id_name = json.load(f)
    id_name_dict = {}
    for item in id_name:
        id_name_dict[item['id']] = item['name']

    checkpoint = torch.load(cfg.BEATS.CKPT_2M)
    beats_cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(beats_cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    output_dict = {}
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_path = cfg.BEATS.OUTPUT_PATH

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
        audio_wav_path = os.path.join(cfg.DATA.AUDIO_WAV_DIR, category, video_name + '.wav')
        audio_input_16khz, _ = librosa.load(audio_wav_path, sr=16000)
        audio_input_16khz = torch.tensor(audio_input_16khz).unsqueeze(0)
        padding_mask = torch.zeros(1, 10000).bool()

        probs = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
        silence_id = next((k for k, v in checkpoint['label_dict'].items() if v == '/m/028v0c'), None) # 235
        probs[:, silence_id] = 0

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
            top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
            top5_name = [id_name_dict[label] for label in top5_label]
            output_dict[video_name] = top5_name

    print(len(output_dict))

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_dict, outfile, ensure_ascii=False, indent=4)
    
    print("beats finish")
