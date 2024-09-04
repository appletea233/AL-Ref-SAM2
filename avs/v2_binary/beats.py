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

    checkpoint_20k = torch.load(cfg.BEATS.CKPT_20K)
    beats_cfg_20k = BEATsConfig(checkpoint_20k['cfg'])
    BEATs_model_20k = BEATs(beats_cfg_20k)
    BEATs_model_20k.load_state_dict(checkpoint_20k['model'])
    BEATs_model_20k.eval()

    checkpoint_2m = torch.load(cfg.BEATS.CKPT_2M)
    beats_cfg_2m = BEATsConfig(checkpoint_2m['cfg'])
    BEATs_model_2m = BEATs(beats_cfg_2m)
    BEATs_model_2m.load_state_dict(checkpoint_2m['model'])
    BEATs_model_2m.eval()

    output_dict_20k = {}
    output_dict_2m = {}
    output_dict = {}
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_path = cfg.BEATS.OUTPUT_PATH

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_v2 = df_all[df_all['label'] == cfg.LABEL]
    df_v2_test = df_v2[df_v2['split'] == cfg.SPLIT]

    for index in range(len(df_v2_test)):
        print("index:", index)
        df_one_video = df_v2_test.iloc[index]
        video_name = df_one_video.iloc[1]
        audio_wav_path = os.path.join(cfg.DATA.DIR, video_name, 'audio.wav')
        audio_input_16khz, _ = librosa.load(audio_wav_path, sr=16000)
        audio_input_16khz = torch.tensor(audio_input_16khz).unsqueeze(0)
        padding_mask = torch.zeros(1, 10000).bool()

        # 20k
        probs = BEATs_model_20k.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
        silence_id = next((k for k, v in checkpoint_20k['label_dict'].items() if v == '/m/028v0c'), None) # 235
        probs[:, silence_id] = 0

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
            top5_label = [checkpoint_20k['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
            top5_name = [id_name_dict[label] for label in top5_label]
            output_dict_20k[video_name] = top5_name

        # 2m
        probs = BEATs_model_2m.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
        silence_id = next((k for k, v in checkpoint_2m['label_dict'].items() if v == '/m/028v0c'), None) # 235
        probs[:, silence_id] = 0

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
            top5_label = [checkpoint_2m['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
            top5_name = [id_name_dict[label] for label in top5_label]
            output_dict_2m[video_name] = top5_name

    print("output_dict_20k:", len(output_dict_20k))
    print("output_dict_2m:", len(output_dict_2m))

    for index in range(len(df_v2_test)):
        print("index:", index)
        df_one_video = df_v2_test.iloc[index]
        video_name = df_one_video.iloc[1]
        labels_20k = output_dict_20k[video_name]
        labels_2m = output_dict_2m[video_name]
        labels = list(set(labels_20k + labels_2m))

        output_dict[video_name] = labels

    print("output_dict:", len(output_dict))

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_dict, outfile, ensure_ascii=False, indent=4)

    print("beats finish")
