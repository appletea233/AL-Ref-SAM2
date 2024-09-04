import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle
from config import cfg

def box(jpg_dir, category, video_name, keyframe, bboxes, box_jpg_dir):
    T = 5
    image_paths = [os.path.join(jpg_dir, category, video_name, f"{i}.jpg") for i in range(1, T + 1)]
    images = [mpimg.imread(path) for path in image_paths]

    h, w, c = images[0].shape

    fig, axs = plt.subplots(1, T, figsize=(5 * T, 5 * h/w))

    colors = ['green', 'magenta', 'orange']

    for i, (ax, img) in enumerate(zip(axs, images), start=1):
        ax.imshow(img)
        circ = Circle((20, 20), 15, fill=True, edgecolor='white', linewidth=2)
        ax.text(20, 20, str(i), color='white', fontsize=20, ha='center', va='center')
        ax.add_patch(circ)
        ax.axis('off')

        if i == keyframe:
            for j, bbox in enumerate(bboxes, start=1):
                cx, cy, bw, bh = bbox
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                color = colors[j - 1]
                rect = Rectangle((x, y), bw * w, bh * h, linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                circ = Circle((x + bw * w - 8, y + bh * h - 8), 8, fill=True, edgecolor='white', facecolor=color, linewidth=2)
                ax.add_patch(circ)
                ax.text(x + bw * w - 8, y + bh * h - 8, str(j), color='white', fontsize=15, ha='center', va='center')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0)
    save_dir = os.path.join(box_jpg_dir, category, video_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "box.jpg"))
    plt.close()

if __name__ == "__main__":
    print("concat stage 2 start")

    box_jpg_dir = cfg.DATA.CONCAT_2_JPG_IMG_DIR
    os.makedirs(box_jpg_dir, exist_ok=True)

    with open(cfg.GDINO.OUTPUT_PATH, 'r') as f:
        name_label_keyframe_bboxes = json.load(f) # 0~1, cx,cy,w,h

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name, category = df_one_video.iloc[0], df_one_video.iloc[2]
        label_keyframe_bboxes = name_label_keyframe_bboxes[video_name]
        if len(label_keyframe_bboxes) == 0: # {}
            continue

        label = next(iter(label_keyframe_bboxes))
        keyframe = label_keyframe_bboxes[label]['keyframe']
        bboxes = label_keyframe_bboxes[label]['bboxes']
        if len(bboxes) <= 1:
            continue
        box(cfg.DATA.JPG_IMG_DIR, category, video_name, keyframe, bboxes, box_jpg_dir)

    print("concat stage 2 finish")
