import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import threading

from opts import get_args_parser


def main(args):
    print("Inference only supports for batch size = 1") 

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = output_dir
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
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    # for i in range(thread_num):
    #     if i == thread_num - 1:
    #         sub_video_list = video_list[i * per_thread_video_num:]
    #     else:
    #         sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
    #     p = mp.Process(target=sub_processor, args=(lock, i, args, data, 
    #                                                save_path_prefix, 
    #                                                img_folder, sub_video_list))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
    sub_processor(lock, 0, args, data, save_path_prefix, img_folder, video_list)

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import os

def concat_frames(image_paths, output_path, image_num, step, id):
    images = [mpimg.imread(x) for x in image_paths]
    h, w, c = images[0].shape

    fig, axs = plt.subplots(len(images), 1, figsize=(5, 5 * h/w * len(images)))
    for i, (ax, img) in enumerate(zip(axs, images), start=1):
        ax.imshow(img)
        circ = Circle((50, 50), 35, fill=True, edgecolor='white', linewidth=2)
        ax.text(50, 50, str(i), color='white', fontsize=15, ha='center', va='center')
        ax.add_patch(circ)
        ax.axis('off')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_path, f'combined_{image_num}_step_{step}_id_{id}.jpg'))
    plt.close()

def sub_processor(lock, pid, args, data, save_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    # with lock:
    #     progress = tqdm(
    #         total=len(video_list),
    #         position=pid,
    #         desc=text,
    #         ncols=0
    #     )

    # start inference
    num_all_frames = 0 

    # 1. For each video
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
        video_name = meta[0]["video"]
        frames = meta[0]["frames"]

        video_len = len(frames)
        video_dir = os.path.join(img_folder, video_name)
        origin_w, origin_h = Image.open(os.path.join(img_folder, video_name, frames[0] + ".jpg")).convert('RGB').size
        output_path = os.path.join(args.output_dir, 'concat', video_name, 'stage_1')
        os.makedirs(output_path, exist_ok=True)

        for id, start_id in enumerate(list(range(0, video_len, args.concat_frame_step * args.concat_frame_num))):
            end_id = start_id + args.concat_frame_step * args.concat_frame_num
            if end_id > video_len:
                end_id = video_len
            image_paths = [os.path.join(img_folder, video_name, frames[i] + ".jpg") for i in range(start_id, end_id, args.concat_frame_step)]  
            if len(image_paths) == 1:
                continue

            try:
                concat_frames(image_paths, output_path, args.concat_frame_num, args.concat_frame_step, id)
            except:
                breakpoint()
        # with lock:
        #     progress.update(1)
    result_dict[str(pid)] = num_all_frames
    # with lock:
    #     progress.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, 'davis', args.prefix,)
    print(f"output dir: {args.output_dir}")
    main(args)
