import os
import sys
import ast
import json
import tqdm
import torch
import librosa
import itertools
import pandas as pd
import soundfile as sf
from config import cfg

from languagebind import to_device, LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor

def split_wav(original_wav, intervals, save_path):
    y, sr = librosa.load(original_wav, sr=None)
    total_y = y.shape[0]
    segment_paths = []
    for i, (s, e) in enumerate(intervals):
        start = s * sr
        end = min(e * sr, total_y)
        segment = y[start:end]
        output_path = os.path.join(save_path, f'segment_{s}_{e}.wav')
        sf.write(output_path, segment, sr)
        print(f"Saved {output_path}")
        segment_paths.append([s, e, output_path])
    return segment_paths


if __name__ == '__main__':
    print("languagebind 1 start")

    label_dict = {'dog': 'dog barking', 'cat': 'cat meow', 'gun':'gun shot'}
    reverse_label_dict = {v:k for k, v in label_dict.items()}

    device = torch.device('cuda')
    pretrained_ckpt = 'LanguageBind/LanguageBind_Audio_FT'
    model = LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    model = model.to(device)
    tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    audio_process = LanguageBindAudioProcessor(model.config, tokenizer)
    model.eval()

    with open(cfg.GPT_STAGE_1.OUTPUT_PATH, 'r') as f:
        name_label_keyframe = json.load(f)

    with open(cfg.LANGUAGEBIND.AUDIO_SEGMENT_FILE, 'r') as f:
        name_intervals = json.load(f)

    label_for_second_dict = dict()

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    for video_id in tqdm.tqdm(range(len(df_test))):
        df_one_video = df_test.iloc[video_id]
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]

        label_keyframe = ast.literal_eval(name_label_keyframe[video_name])
        raw_label_text = list(label_keyframe.keys())

        label_for_second_dict[video_name] = dict()

        T = 10 if video_label == "v2" else 5

        if len(raw_label_text) == 0:
            labels_for_second = [[] for i in range(T)]
            label_for_second_dict[video_name]['label4second'] = labels_for_second
            continue
        if len(raw_label_text) == 1:
            labels_for_second = [[raw_label_text[0]] for i in range(T)]
            label_for_second_dict[video_name]['label4second'] = labels_for_second
            continue

        label_text = []
        for label in raw_label_text:
            if label in label_dict:
                label_text.append(label_dict[label])
            else:
                label_text.append(label)

        label_text_combine = [t for t in label_text]
        for l in range(2, len(label_text) + 1):
            combinations = list(itertools.combinations(label_text, l))
            for comb in combinations:
                label_text_combine.append('; '.join(comb))

        # split audio files
        audio_path = os.path.join(cfg.DATA.BASE_DIR, video_label, video_name, 'audio.wav')
        audio_segment_path = os.path.join(cfg.LANGUAGEBIND.AUDIO_SEGMENT_WAV_DIR, video_label, video_name)
        os.makedirs(audio_segment_path, exist_ok=True)
        intervals = name_intervals[video_name]
        audio_segment_files = split_wav(audio_path, intervals, audio_segment_path)
        audio_segment_roots = [l[2] for l in audio_segment_files]

        data = to_device(audio_process(audio_segment_roots, label_text_combine, return_tensors='pt'), device)

        with torch.no_grad():
            out = model(**data)
        similarity = out.text_embeds @ out.image_embeds.T
        top1 = similarity.max(0)[1]
        select_labels = [label_text_combine[t] for t in top1]
        select_labels_list = []
        for select_label in select_labels:
            select_label = select_label.split('; ')
            for j in range(len(select_label)):
                if select_label[j] in reverse_label_dict.keys():
                    select_label[j] = reverse_label_dict[select_label[j]]
            select_labels_list.append(select_label)
        labels_for_second = []
        for idx, inter in enumerate(intervals):
            for k in range(inter[0], inter[1]):
                labels_for_second.append(select_labels_list[idx])
        if len(labels_for_second) < T:
            N = T - len(labels_for_second)
            for k in range(N):
                labels_for_second.append(labels_for_second[-1])
        label_for_second_dict[video_name]['label4second'] = labels_for_second
        print(labels_for_second)

    with open(cfg.LANGUAGEBIND.LABEL4SECOND, 'w') as f:
        json.dump(label_for_second_dict, f, indent=4)

    print("languagebind 1 finish")
