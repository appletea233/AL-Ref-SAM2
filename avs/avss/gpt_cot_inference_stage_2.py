import os
import re
import json
import openai
import threading
import time
import base64
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from config import cfg

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPTAgent():
    def __init__(self, api_key, api_base):
        openai.api_key = api_key
        openai.api_base = api_base

    def get_response(self, prompts, user_id, image_path):
        try:
            base64_image=encode_image(image_path)
            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompts},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                        ]
                    },
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error occurred while requesting ChatGPT: {e}")
            return None

def process_string(input_string):
    match_brackets = re.search(r'\[(.*?)\]', input_string)
    if match_brackets:
        input_string = match_brackets.group(1)
        match_number = re.search(r'\d+', input_string)
        if match_number:
            number = match_number.group(0)
        else:
            number = 1
    else:
        number = 1
    return number

def sub_processor(lock, pid, cfg, sub_df_test):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(sub_df_test),
            position=pid,
            desc=text,
            ncols=0
        )

    API_KEY = cfg.GPT.GPT_API_KEY
    API_BASE = cfg.GPT.GPT_API_BASE
    img_dir = cfg.DATA.CONCAT_2_JPG_IMG_DIR
    dino_path = cfg.GDINO.OUTPUT_PATH

    chatgpt_client = GPTAgent(API_KEY, API_BASE)

    with open(dino_path, 'r') as f:
        name_label_keyframe_bboxes = json.load(f) # 0~1, cx,cy,w,h

    for index in range(len(sub_df_test)):
        df_one_video = sub_df_test.iloc[index]
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]
        label_keyframe_bboxes = name_label_keyframe_bboxes[video_name]
        if len(label_keyframe_bboxes) == 0: # {}
            with lock:
                output_dict[video_name] = {}
            continue

        for label, keyframe_bboxes in label_keyframe_bboxes.items():
            keyframe = keyframe_bboxes['keyframe']
            bboxes = keyframe_bboxes['bboxes']

            if len(bboxes) <= 1:
                continue

            img_path = os.path.join(img_dir, video_label, video_name, label, "box.jpg")

            colors = ['green', 'magenta', 'orange']
            box_string = ""
            for b_id in range(len(bboxes)):
                color = colors[b_id]
                box_string += f'id: {b_id+1}, color: {color}, normalized box center: ({bboxes[b_id][0]}, {bboxes[b_id][1]}), normalized box width: {bboxes[b_id][2]}, normalized box height: {bboxes[b_id][3]}. '

            if video_label == "v2":
                prompts_prefix = "The given image is composed of 10 frames from a video. The first row is frames 1 to 5 from left to right, and the second row is frames 6 to 10 from left to right. The frame number is marked with a circle in the upper left corner of each frame."
            else:
                prompts_prefix = "The given image is composed of 5 frames from a video spliced from left to right, and the frame number is marked with a circle in the upper left corner of each frame."

            prompts = f"{prompts_prefix} Now we have several candidates of category {label} and one of them is making sound in the video, and we mark these candidates with numbered bounding boxes on frame {keyframe}. The coordinates and scale of the bounding boxes are {box_string}You should select the bounding box of the object that is most likely to make sound. Here are three tips for you to select the precise bounding box of sounding object: (1) When the sound-producing object is a stringed instrument like violin, the selected box should only contains the body of the instrument, without including the bow. (2) The bounding box you choose should encompass the entire sound-producing object rather than merely a part of it, and the scale of the former is usually larger than the latter. (3) It is usually a salient object or a moving object in the video frame that makes sound in the video. Your output should be in the format of [id], and replace id with the selected box number from 1 to {len(bboxes)}. Do not include any additional information."

            user_id = "test"
            response = chatgpt_client.get_response(prompts, user_id, image_path=img_path)
            if response:
                number = process_string(response)
                response = str(number)
            else:
                response = "1"
            print(video_name, label, response)

            with lock:
                if video_name not in output_dict:
                    output_dict[video_name] = {}
                tmp_dict = output_dict[video_name]
                tmp_dict[label] = response
                output_dict[video_name] = tmp_dict

        with lock:
            progress.update(1)

    with lock:
        progress.close()


if __name__ == "__main__":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_path = cfg.GPT_STAGE_2.OUTPUT_PATH

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    # create subprocess
    thread_num = cfg.GPT_STAGE_2.NTHREAD
    global output_dict
    output_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(df_test)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print("gpt cot inference stage 2 start")
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_df_test = df_test.iloc[i * per_thread_video_num:]
        else:
            sub_df_test = df_test.iloc[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, cfg, sub_df_test))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    output_dict = dict(output_dict)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_dict, outfile, ensure_ascii=False, indent=4)

    print("Total inference time: %.4f s" %(total_time))
    print("gpt cot inference stage 2 finish")
