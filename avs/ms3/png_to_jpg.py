import os
import pandas as pd
from PIL import Image
from config import cfg

if __name__ == '__main__':
    print("png_to_jpg start")

    jpg_base_dir = cfg.DATA.JPG_IMG_DIR
    os.makedirs(jpg_base_dir, exist_ok=True)
    
    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]
    T = 5

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name = df_one_video.iloc[0]
        frame_dir = os.path.join(cfg.DATA.PNG_IMG_DIR, video_name)
        for img_id in range(1, T + 1):
            png_path = os.path.join(frame_dir, f"{video_name}.mp4_{img_id}.png")
            rgb_image = Image.open(png_path).convert('RGB')
            jpg_dir = os.path.join(jpg_base_dir, video_name)
            os.makedirs(jpg_dir, exist_ok=True)
            jpg_path = os.path.join(jpg_dir, f"{img_id}.jpg")
            rgb_image.save(jpg_path)
    
    print("png_to_jpg finish")
