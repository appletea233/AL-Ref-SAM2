import os
import ast
import cv2
import json
import torch
import pandas as pd
from config import cfg
from groundingdino.util.inference import load_model, load_image, predict, annotate

if __name__ == '__main__':
    print("gdino start")

    model = load_model(cfg.GDINO.CONFIG, cfg.GDINO.CKPT)

    output_dict = {}
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_path = cfg.GDINO.OUTPUT_PATH
    vis_output_dir = cfg.GDINO.VIS_OUTPUT_DIR
    os.makedirs(vis_output_dir, exist_ok=True)

    with open(cfg.GPT_STAGE_1.OUTPUT_PATH, 'r') as f:
        name_label_keyframe = json.load(f)

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]
        output_dict[video_name] = {}
        T = 10 if video_label == "v2" else 5

        label_keyframe = ast.literal_eval(name_label_keyframe[video_name]) # dict
        if len(label_keyframe) == 0:
            continue
        for label, keyframe in label_keyframe.items():
            if keyframe < 1 or keyframe > T:
                keyframe = 1

            IMAGE_PATH = os.path.join(cfg.DATA.BASE_DIR, video_label, video_name, "frames", f"{keyframe - 1}.jpg")
            TEXT_PROMPT = label
            BOX_TRESHOLD = cfg.GDINO.BOX_TRESHOLD
            TEXT_TRESHOLD = cfg.GDINO.TEXT_TRESHOLD

            image_source, image = load_image(IMAGE_PATH)

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            # boxes: [N, 4]
            # logits: [N]
            # phrases: a list of N phrases
            if len(boxes) >= 3:
                logits, top_indices = torch.topk(logits, 3)
                boxes = boxes[top_indices]
                phrases = [phrases[i] for i in top_indices.tolist()]

            output_dict[video_name][label] = {}
            output_dict[video_name][label]['keyframe'] = keyframe
            output_dict[video_name][label]['bboxes'] = boxes.tolist()
            # output_dict[video_name][label]['logits'] = logits.tolist()

            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            annotated_frame_path = os.path.join(vis_output_dir, f"{video_label}_{video_name}_{label}_{keyframe}_dino.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_dict, outfile, ensure_ascii=False, indent=4)

    print("gdino finish")
