import os
import json
import torch
import pandas as pd
from PIL import Image
from config import cfg
from torchvision import transforms
from utils.color_utils import get_v2_pallete, save_color_mask

if __name__ == '__main__':
    print("merge start")

    objs_list = cfg.OBJECTS
    v2_pallete = get_v2_pallete(cfg.DATA.LABEL2IDX)

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    os.makedirs(cfg.MERGE.VIS_OUTPUT_DIR, exist_ok=True)

    with open(cfg.LANGUAGEBIND.LABEL4SECOND, 'r') as f:
        name_label4second = json.load(f)

    with open(cfg.LANGUAGEBIND.LABEL_REPLACEMENT, 'r') as f:
        label_replacement = json.load(f)

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]
        origin_w, origin_h = Image.open(os.path.join(cfg.DATA.BASE_DIR, video_label, video_name, "frames", "0.jpg")).convert('RGB').size

        choose_lists = name_label4second[video_name]['label4second'] # [['dog'], ['dog', 'cat'], [...], [...], [...]]

        T = 10 if video_label == "v2" else 5

        all_pred_masks = []
        for img_id in range(1, T + 1):
            choose_list = choose_lists[img_id - 1] # list
            if len(choose_list) == 0:
                all_pred_masks.append(torch.zeros(origin_h, origin_w))
            else:
                pred_mask = torch.zeros(origin_h, origin_w)
                for label in choose_list:
                    mask_path = os.path.join(cfg.SAM2.VIS_OUTPUT_DIR, video_label, video_name, label, f"{img_id}_video_mask.png")
                    mask = Image.open(mask_path).convert('L')
                    transform = transforms.ToTensor()
                    mask = (transform(mask) != 0).squeeze(0)
                    if label in label_replacement:
                        label = label_replacement[label]
                    label_index = objs_list.index(label) + 1
                    pred_mask[torch.where(mask)] = label_index
                all_pred_masks.append(pred_mask)

        all_pred_masks = torch.stack(all_pred_masks, dim=0)   # [T, h, w]

        save_dir = os.path.join(cfg.MERGE.VIS_OUTPUT_DIR, video_label, video_name)
        save_color_mask(all_pred_masks, save_dir, v2_pallete, T)

    print("merge finish")
