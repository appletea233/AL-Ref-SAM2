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
                model="gpt-4-turbo-2024-04-09",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompts},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
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
    colon_space_positions = [m.start() for m in re.finditer(r': ', input_string)]

    result_dict = {}

    for pos in colon_space_positions:
        match_number = re.search(r'^\d+', input_string[pos + 2:])
        if match_number:
            number = match_number.group(0)

            match_word = re.search(r'["\']([^"\']+)["\']$', input_string[:pos])
            if match_word:
                full_word = match_word.group(1)
                first_word = re.split(r'[,;]', full_word.strip())[0]
                if first_word:
                    result_dict[first_word] = int(number)

    return result_dict

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
    img_dir = cfg.DATA.CONCAT_1_JPG_IMG_DIR
    beats_path = cfg.BEATS.OUTPUT_PATH
    objs_list = cfg.OBJECTS

    chatgpt_client = GPTAgent(API_KEY, API_BASE)

    with open(beats_path, 'r') as f:
        name_labels = json.load(f)

    for index in range(len(sub_df_test)):
        df_one_video = sub_df_test.iloc[index]
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]
        img_path = os.path.join(img_dir, video_label, video_name, "concat.jpg")

        beats_labels = name_labels[video_name] # a list of audio labels

        if video_label == "v2":
            prompts_prefix = "The given image is composed of 10 frames from a video. The first row is frames 1 to 5 from left to right, and the second row is frames 6 to 10 from left to right. The frame number is marked with a circle in the upper left corner of each frame."
        else:
            prompts_prefix = "The given image is composed of 5 frames from a video spliced from left to right, and the frame number is marked with a circle in the upper left corner of each frame."

        prompts = f"{prompts_prefix} Using an audio classification model, we obtained the audio labels with the highest confidence in the video: {beats_labels}. Please process these audio labels based on the content of the image, filtering out audio labels that do not exist in the video or are abstract labels that cannot be associated with specific objects. Additionally, merge audio labels that represent the same object. Then, according to the retained audio labels, output the category of one or more objects in the video that may be making sounds. The category of sound-producing object should be selected from the following list as much as possible: {objs_list}. Please note that the 'keyboard' in this list refers to a typing keyboard, not a musical keyboard. For each selected sound-producing object category, select a video frame that is easiest to distinguish the sound-producing object clearly. Your output should be in the format of ['obj': id] or ['obj1': id1, 'obj2': id2, ...]. Replace obj with the selected sound-producing object category and id with the selected video frame number. Please note that objs in the output results need to be different. Do not include any additional information."

        user_id = "test"
        response = chatgpt_client.get_response(prompts, user_id, image_path=img_path)
        if response:
            result_dict = process_string(response)
            response = str(result_dict)
        else:
            response = "{}"
        print(video_label, response)

        with lock:
            output_dict[video_name] = response
            progress.update(1)

    with lock:
        progress.close()


if __name__ == "__main__":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_path = cfg.GPT_STAGE_1.OUTPUT_PATH

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    # create subprocess
    thread_num = cfg.GPT_STAGE_1.NTHREAD
    global output_dict
    output_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(df_test)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print("gpt cot inference stage 1 start")
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
    print("gpt cot inference stage 1 finish")
