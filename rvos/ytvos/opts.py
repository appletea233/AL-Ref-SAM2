import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('davis inference scripts.', add_help=False)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--nthread', default=1, type=int)
    parser.add_argument('--api_key', default="YOUR API KEY", type=str)
    parser.add_argument('--api_base', default="YOUR API BASE", type=str)
    parser.add_argument('--concat_frame_num', default=5, type=int)
    parser.add_argument('--concat_frame_step', default=10, type=int)
    parser.add_argument('--gd_box_threshold', default=0.25, type=float)
    parser.add_argument('--gd_text_threshold', default=0.2, type=float)
    parser.add_argument('--max_box_num', default=3, type=int)
    parser.add_argument('--anno_num', default=4, type=int)
    parser.add_argument('--ytvos_path', default="YOUTUBE-VOS PATH", type=str)
    parser.add_argument('--split', default="valid", type=str)
    parser.add_argument('--prefix', default="baseline", type=str)
    parser.add_argument('--gdino_config', default="GDINO_CONFIG", type=str)
    parser.add_argument('--gdino_ckpt', default="GDINO_CKPT", type=str)
    parser.add_argument('--sam2_config', default="SAM2_CONFIG", type=str)
    parser.add_argument('--sam2_ckpt', default="SAM2_CKPT", type=str)
    parser.add_argument('--output_dir', default="OUTPUT_DIR", type=str)

    return parser