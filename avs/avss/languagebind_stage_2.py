import sys
import ast
import json
import torch
from config import cfg

from languagebind import to_device, LanguageBind, transform_dict, LanguageBindImageTokenizer


if __name__ == '__main__':
    print("languagebind 2 start")

    device = torch.device('cuda')
    clip_type = {
        'image': 'LanguageBind_Image',
    }
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    inputs = {}
    inputs['image'] = {}
    inputs['image']['pixel_values'] = torch.zeros((71, 3, 224, 224))
    inputs['image'] = to_device(inputs['image'], device)

    objs_list = cfg.OBJECTS

    with open(cfg.GPT_STAGE_1.OUTPUT_PATH, 'r') as f:
        name_label_keyframe = json.load(f)

    labels = set()
    for video_name, label_keyframe in name_label_keyframe.items():
        label_keyframe = ast.literal_eval(label_keyframe)
        labels.update(label_keyframe.keys())

    label_dict = {}
    for label in labels:
        if label not in objs_list:
            language = [label] + objs_list

            inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), device)

            with torch.no_grad():
                embeddings = model(inputs)

            language_embed = embeddings['language']
            similarity = language_embed[0] @ language_embed.T # 71
            similarity = similarity[1:]
            max_index = torch.argmax(similarity)
            obj = objs_list[max_index]
            label_dict[label] = obj
            print(label, obj)

    print("label_dict", label_dict)
    with open(cfg.LANGUAGEBIND.LABEL_REPLACEMENT, 'w', encoding='utf-8') as outfile:
        json.dump(label_dict, outfile, ensure_ascii=False, indent=4)

    print("languagebind 2 finish")
