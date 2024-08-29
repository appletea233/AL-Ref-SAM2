import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm
from tqdm import tqdm
import time

import multiprocessing as mp
import threading
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict, annotate

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils import show_anns, show_mask, show_points, show_masks
from opts import get_args_parser


def main(args):
    print("Inference only supports for batch size = 1") 

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'output_stage_1', split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # load data
    root = Path(args.ytvos_path) # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    if 'youtube' in args.ytvos_path:
        valid_test_videos = set(data.keys())
        # for some reasons the competition's validation expressions dict contains both the validation (202) & 
        # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
        # the validation expressions dict:
        test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
        with open(test_meta_file, 'r') as f:
            test_data = json.load(f)['videos']
        test_videos = set(test_data.keys())
        valid_videos = valid_test_videos - test_videos
        video_list = sorted([video for video in valid_videos])
        assert len(video_list) == 202, 'error: incorrect number of validation videos'
    else:
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

    torch.cuda.set_device(pid % args.ngpu)

    # grounding dino model
    model = load_model(
        args.gdino_config, 
        args.gdino_ckpt,
        device='cuda'
    )
    BOX_TRESHOLD = 0.001
    TEXT_TRESHOLD = 0.1

    # sam2 for video
    predictor = build_sam2_video_predictor(
        args.sam2_config, 
        args.sam2_ckpt,
        device="cuda",
    )

    # sam2 for image
    sam2 = build_sam2(
        args.sam2_config, 
        args.sam2_ckpt, 
        device='cuda', 
        apply_postprocessing=False
    )
    image_predictor = SAM2ImagePredictor(sam2)

    # start inference
    num_all_frames = 0 
    model.eval()

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

        concat_video_dir = os.path.join(args.output_dir, 'concat', meta[0]['video'], 'stage_1')
        # 2. For each expression
        with open(os.path.join(concat_video_dir, 'cot_one_stage.json'), 'r') as f:
            result_dict = json.load(f)
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)
            video_dir = os.path.join(img_folder, video_name)
            origin_w, origin_h = Image.open(os.path.join(img_folder, video_name, frames[0] + ".jpg")).convert('RGB').size

            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if os.path.exists(save_path):
                continue
            
            results = result_dict[str(i)]
            key_frames = []
            for t, segment in enumerate(results['res_list']):
                target_frame_id = segment['target_frame_id']
                if target_frame_id <= 0 or target_frame_id > args.concat_frame_num:
                    continue
                real_freame_id = t * args.concat_frame_step * args. concat_frame_num + (target_frame_id - 1) * args.concat_frame_step
                if real_freame_id >= len(frames):
                    continue
                key_frames.append(real_freame_id)
            if len(key_frames) == 0:
                key_frames = [0]

            # init sam2
            inference_state = predictor.init_state(video_path=video_dir)
            predictor.reset_state(inference_state)
            
            for key_frame in key_frames:
                with torch.no_grad():
                    img_path = os.path.join(video_dir, frames[key_frame] + ".jpg")
                    image_source, image = load_image(img_path)
                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=exp,
                        box_threshold=BOX_TRESHOLD,
                        text_threshold=TEXT_TRESHOLD
                    )

                    if len(phrases) == 0:
                        continue
                    max_logit_box_id = logits.argmax()
                    boxes = boxes[max_logit_box_id].unsqueeze(0)
                    logits = logits[max_logit_box_id].unsqueeze(0)
                    phrases = [phrases[max_logit_box_id]]

                    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                    # cv2.imwrite("annotated_image.jpg", annotated_frame)

                    input_boxes = boxes * torch.Tensor([origin_w, origin_h, origin_w, origin_h])
                    input_boxes = box_convert(boxes=input_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                    input_box = input_boxes[0]

                    image = Image.open(img_path)
                    image = np.array(image.convert("RGB"))
                    image_predictor.set_image(image)
                    masks, scores, _ = image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=key_frame,
                        obj_id=1,
                        mask=masks[0],
                    )
            
            start_frame = 0
            key_frame_num = len(key_frames)
            start_frame_idx = key_frames[key_frame_num // 2]


            all_pred_masks_dict = {}

            with torch.no_grad():
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_idx):
                    out_mask_logit = (out_mask_logits[0][0] > 0) # [h, w], bool
                    all_pred_masks_dict[out_frame_idx] = out_mask_logit

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True, start_frame_idx=start_frame_idx):
                    out_mask_logit = (out_mask_logits[0][0] > 0) # [h, w], bool
                    all_pred_masks_dict[out_frame_idx] = out_mask_logit

                pass
            keys = list(all_pred_masks_dict.keys())
            keys.sort()
            all_pred_masks = [all_pred_masks_dict[frame_idx] for frame_idx in keys]
            all_pred_masks = torch.stack(all_pred_masks, dim=0)   # (video_len, h, w) 
            all_pred_masks = all_pred_masks.cpu().numpy()
            
            # save binary image
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(video_len):
                frame_name = frames[j]
                mask = all_pred_masks[j].astype(np.float32) 
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

        with lock:
            progress.update(1)

    with lock:
        progress.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'youtube' in args.ytvos_path:
        dataset = 'ytvos'
    elif 'mevis' in args.ytvos_path:
        dataset = 'mevis'
    args.output_dir = os.path.join(args.output_dir, dataset, args.prefix)
    print(f"output dir: {args.output_dir}")
    main(args)
