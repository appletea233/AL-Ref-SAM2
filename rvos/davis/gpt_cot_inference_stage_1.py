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

            concat_video_dir = os.path.join(args.output_dir, 'concat', video_name, 'stage_1')
            concat_images = os.listdir(concat_video_dir)
            concat_image_paths = [os.path.join(concat_video_dir, image_name) for image_name in concat_images if f'combined_{args.concat_frame_num}_step_{args.concat_frame_step}' in image_name]

            results = dict(
                exp_id=i,
                exp=exp,
                video_dir=video_dir,
                res_list=[],
            )
            for concat_image_path in concat_image_paths:
                prompt = f"I have input an image stitched together from frames of a video, each frame is marked with an ID in the upper left corner.Please first describe in detail the events happening in the video and then help me select the single frame that best demonstrate the \"{exp}\" and may results a good segmentation result of the object previously described, and return their IDs in the upper left corner to me in a list surrounded by []."
                response = chatgpt_client.get_response(prompt, 'test', image_path=concat_image_path)
               
                try:
                    matches = re.findall(r'\[(.*?)\]', response)
                    target_frame_id = int(matches[-1])
                except:
                    target_frame_id = -1

                res = dict(
                    prompt=prompt,
                    response=response,
                    image_path=concat_image_path,
                    target_frame_id=target_frame_id,
                )
                results["res_list"].append(res)
            result_dict[i] = results
        
        with open(os.path.join(concat_video_dir, 'cot_one_stage.json'), 'w') as f:
            json.dump(result_dict, f)
        results_dict[video] = result_dict

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
