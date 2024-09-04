import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import cfg
from PIL import Image
from utils.utility import show_masks
from torchvision.ops import box_convert

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

if __name__ == '__main__':
    print("sam2 start")

    sam2_checkpoint = cfg.SAM2.CKPT
    model_cfg = cfg.SAM2.CONFIG
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
    image_predictor = SAM2ImagePredictor(sam2)

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_v2 = df_all[df_all['label'] == cfg.LABEL]
    df_v2_test = df_v2[df_v2['split'] == cfg.SPLIT]

    os.makedirs(cfg.SAM2.VIS_OUTPUT_DIR, exist_ok=True)

    with open(cfg.GDINO.OUTPUT_PATH, 'r') as f:
        name_label_keyframe_bboxes = json.load(f) # 0~1, cx,cy,w,h

    with open(cfg.GPT_STAGE_2.OUTPUT_PATH, 'r') as f:
        name_label_keybox = json.load(f)

    T = 10

    for index in range(len(df_v2_test)):
        print("index:", index)
        df_one_video = df_v2_test.iloc[index]
        video_name = df_one_video.iloc[1]

        origin_w, origin_h = Image.open(os.path.join(cfg.DATA.DIR, video_name, "frames", "0.jpg")).convert('RGB').size

        label_keyframe_bboxes = name_label_keyframe_bboxes[video_name]
        if len(label_keyframe_bboxes) == 0:
            continue
        for label, keyframe_bboxes in label_keyframe_bboxes.items():
            keyframe = keyframe_bboxes['keyframe']
            bboxes = keyframe_bboxes['bboxes']

            save_dir = os.path.join(cfg.SAM2.VIS_OUTPUT_DIR, video_name, label)
            os.makedirs(save_dir, exist_ok=True)

            if len(bboxes) == 0:
                all_pred_masks = torch.zeros(T, origin_h, origin_w, dtype=torch.bool) # [T, h, w]
            else:
                box_id = 0 if len(bboxes) == 1 else (int(name_label_keybox[video_name][label]) - 1)
                if box_id < 0 or box_id >= len(bboxes):
                    box_id = 0

                box = torch.Tensor(bboxes[box_id])
                box = box * torch.Tensor([origin_w, origin_h, origin_w, origin_h])
                input_box = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                # generate mask for the keyframe
                frame_dir = os.path.join(cfg.DATA.DIR, video_name, "frames")
                frame_path = os.path.join(frame_dir, f"{keyframe - 1}.jpg")
                image = Image.open(frame_path)
                image = np.array(image.convert("RGB"))
                image_predictor.set_image(image)
                masks, scores, _ = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                ) # 1 x H x W
                show_masks(image, masks, scores, box_coords=input_box)
                plt.savefig(os.path.join(save_dir, f"{keyframe}_image_mask.jpg"))
                plt.close()

                ann_frame_idx = keyframe - 1 # the frame index we interact with
                ann_obj_id = 1 # give a unique id to each object we interact with (it can be any integers)

                inference_state = predictor.init_state(video_path=frame_dir)
                predictor.reset_state(inference_state)

                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=masks[0],
                )

                all_pred_masks_dict = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    out_mask_logit = (out_mask_logits[0][0] > 0) # [h, w], bool
                    all_pred_masks_dict[out_frame_idx] = out_mask_logit

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                    out_mask_logit = (out_mask_logits[0][0] > 0) # [h, w], bool
                    all_pred_masks_dict[out_frame_idx] = out_mask_logit

                all_pred_masks = []
                for out_frame_idx in range(T):
                    all_pred_masks.append(all_pred_masks_dict[out_frame_idx])
                all_pred_masks = torch.stack(all_pred_masks, dim=0) # [T, h, w], bool

            for f in range(all_pred_masks.shape[0]): # T
                img_E = Image.fromarray(all_pred_masks[f].cpu().numpy().astype(np.uint8) * 255)
                img_E.save(os.path.join(save_dir, f"{f + 1}_video_mask.png"))

    print("sam2 finish")
