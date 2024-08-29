import os
import json
from pathlib import Path
from PIL import Image
import tqdm
from tqdm import tqdm
import time
import re
import openai
import base64

import multiprocessing as mp
import threading

from opts import get_args_parser

def main(args):
    print("Inference only supports for batch size = 1") 

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # load data
    root = Path(args.davis_path) # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())    
    
    # create subprocess
    thread_num = args.nthread
    global results_dict
    results_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data, 
                                                   save_path_prefix, 
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    # sub_processor(lock, 0, args, data, save_path_prefix, img_folder, video_list)

    end_time = time.time()
    total_time = end_time - start_time

    results_dict = dict(results_dict)
    with open('cot_one_stage.json', 'w') as f:
        json.dump(results_dict, f)

    print("Total inference time: %.4f s" %(total_time))

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
                            {
                                "type": "text", 
                                "text": prompts
                            },
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
            print(f"请求ChatGPT时发生错误: {e}")
            return None

def sub_processor(lock, pid, args, data, save_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    
    # start inference
    num_all_frames = 0 

    # 1. For each video
    chatgpt_client = GPTAgent(args.api_key, args.api_base)
    for video in video_list:
        metas = [] # list[dict], length is number of expressions

        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        result_dict = dict()
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)
            video_dir = os.path.join(img_folder, video_name)
            origin_w, origin_h = Image.open(os.path.join(img_folder, video_name, frames[0] + ".jpg")).convert('RGB').size

            concat_video_dir = os.path.join(args.output_dir, 'concat', video_name, 'stage_2', str(i))
            output_json_dir = os.path.join(concat_video_dir, 'cot_two_stage.json')

            if os.path.exists(output_json_dir):
                continue

            with open(os.path.join(concat_video_dir, f'box_info_combined_{args.concat_frame_num}_step_{args.concat_frame_step}.json'), "r") as f:
                json_info = json.load(f)

            for id, res_dict in enumerate(json_info['res_list']):
                concat_image_path = os.path.join(concat_video_dir, f'combined_{args.concat_frame_num}_step_{args.concat_frame_step}_id_{id}.jpg')
                boxes_info = res_dict.get('boxes_info', None)

                if boxes_info is not None:
                    target_frame_id = res_dict['target_frame_id']
                    boxes = boxes_info['boxes']
                    logits = boxes_info['logits']
                    phrases = boxes_info['phrases']

                    boxes_num = len(logits)
                    if boxes_num == 1:
                        box_des = 'one objects are marked with colored boxes: a green box (label 1)'
                    elif boxes_num == 2:
                        box_des = 'two objects are marked with colored boxes: a green box (label 1) and a pink box (label 2)'
                    elif boxes_num == 3:
                        box_des = 'three objects are marked with colored boxes: a green box (label 1), a pink box (label 2), and an orange box (label 3)'
                    elif boxes_num == 4:
                        box_des = 'four objects are marked with colored boxes: a green box (label 1), a pink box (label 2), an orange box (label 3), and a blue box (label 4)'
                    elif boxes_num == 5:
                        box_des = 'five objects are marked with colored boxes: a green box (label 1), a pink box (label 2), and an orange box (label 3), a blue box (label 4), and a red box (label 5)'
                    else:
                        assert 1 == 2

                    colors = ["green", "pink", "orange", 'blue', 'red', 'yellow']
                    box_phrase_des = ""
                    for p_id, phrase in enumerate(phrases):
                        if p_id == len(phrases) - 1 and p_id != 0:
                            box_phrase_des += 'and '
                        label = p_id + 1
                        box_phrase_des += f"the {colors[p_id]} box (label {label}) as \"{phrase}\""
                        if p_id != len(phrases) - 1:
                            box_phrase_des += ', '
                    prompt = f"""
    The above content is an image that contains frames 1 to 5 of a video, with the frame numbers labeled in the top-left corner. In the {target_frame_id} frame, three objects are marked with colored boxes: {box_des}. Grounding Dino (a grounding model) identifies {box_phrase_des}.
    Please follow these steps: 
    1. **Describe the Scene:** 
        - Describe the video and each frame.
        - Describe each object in the frame.
    2. **Describe the Objects within Each Box:**
        - Describe the objects in the above boxes and their relationships.
        - Grounding model results are not necessarily accurate; do not fully trust the model. Its results can be used as a reference. Provide your own understanding based on the image
    3. **Analyze the Provided Description:**
        - Given the description: "{exp}", analyze its syntax, identifying the main object described in the sentence.
        - Adhere to syntax analysis principles, and do not assume that an object is the main subject simply because it has extensive description.
        - Note the heuristic: the main object often appears earlier in the sentence.
        - This analysis will help you distinguish the box that needs to be selected from the image.
    4. **Identify the Object that Best Matches the Description:**
        - Ensure you select the precise bounding box of the referring object by following these tips:
            1. Include only the main object described, excluding other objects.
            2. Include the whole main object.
            3. Do not include other objects mentioned in the description that are not the main object.
    5. **Output the Result:**
        - Output the single number in list "[]" format.
        - If no objects match the description, output [-1].
        - Provide reasoning for selecting or not selecting each box."""
                    response = chatgpt_client.get_response(prompt, 'test', image_path=concat_image_path)
                    response = response.replace("*", "").replace("\\", "")
                    matches = re.findall(r'\[(.*?)\]', response)

                    try:
                        target_object_id = int(matches[-1])
                    except:
                        target_object_id = -1
                        # print(f"error with video {video} exp {i}") 

                    res_dict.pop('prompt')
                    res_dict.pop('response')
                    res_dict.pop('image_path')
                    res_dict['response_two_stage'] = response
                    res_dict['target_object_id'] = target_object_id

            with open(output_json_dir, 'w') as f:
                json.dump(json_info, f)

        with lock:
            progress.update(1)

    with lock:
        progress.close()
  
  
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, 'davis', args.prefix)
    print(f"output dir: {args.output_dir}")
    main(args)
