import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from config import cfg

if __name__ == '__main__':
    print("merge start")

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    os.makedirs(cfg.MERGE.VIS_OUTPUT_DIR, exist_ok=True)

    with open(cfg.LANGUAGEBIND.LABEL4SECOND, 'r') as f:
        name_label4second = json.load(f)

    T = 5

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name = df_one_video.iloc[0]
        origin_w, origin_h = Image.open(os.path.join(cfg.DATA.JPG_IMG_DIR, video_name, "1.jpg")).convert('RGB').size

        choose_lists = name_label4second[video_name]['label4second'] # [['dog'], ['dog', 'cat'], [...], [...], [...]]

        all_pred_masks = []
        for img_id in range(1, T + 1):
            choose_list = choose_lists[img_id - 1] # list
            if len(choose_list) == 0:
                all_pred_masks.append(torch.zeros((origin_h, origin_w), dtype=torch.bool))
            else:
                pred_mask = torch.zeros((origin_h, origin_w), dtype=torch.bool) # bool
                for label in choose_list:
                    mask_path = os.path.join(cfg.SAM2.VIS_OUTPUT_DIR, video_name, label, f"{img_id}_video_mask.png")
                    mask = Image.open(mask_path).convert('L')
                    mask = np.array(mask).astype(bool) # (h, w), [0, 255] -> [False, True]
                    pred_mask = torch.logical_or(pred_mask, torch.from_numpy(mask))
                all_pred_masks.append(pred_mask)

        all_pred_masks = torch.stack(all_pred_masks, dim=0)   # [T, h, w], bool

        save_dir = os.path.join(cfg.MERGE.VIS_OUTPUT_DIR, video_name)
        os.makedirs(save_dir, exist_ok=True)
        for f in range(all_pred_masks.shape[0]): # T
            img_E = Image.fromarray(all_pred_masks[f].cpu().numpy().astype(np.uint8) * 255)
            img_E.save(os.path.join(save_dir, f"{f + 1}_video_mask.png"))

    print("merge finish")
