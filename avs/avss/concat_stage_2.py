import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle
from config import cfg

def box(jpg_dir, video_label, video_name, label, keyframe, bboxes, box_jpg_dir):
    if video_label == "v2":
        T = 10
        image_paths = [os.path.join(jpg_dir, video_label, video_name, "frames", f"{i}.jpg") for i in range(T)]
        images = [mpimg.imread(path) for path in image_paths]

        h, w, c = images[0].shape

        fig, axs = plt.subplots(2, 5, figsize=(5 * 5, 5 * 2 * h/w))

        colors = ['green', 'magenta', 'orange']

        for i, (ax, img) in enumerate(zip(axs.flat, images), start=1):
            ax.imshow(img)
            circ = Circle((50, 50), 40, fill=True, edgecolor='white', linewidth=2)
            ax.text(50, 50, str(i), color='white', fontsize=15, ha='center', va='center')
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
                    circ = Circle((x + bw * w - 30, y + bh * h - 30), 30, fill=True, edgecolor='white', facecolor=color, linewidth=2)
                    ax.add_patch(circ)
                    ax.text(x + bw * w - 30, y + bh * h - 30, str(j), color='white', fontsize=15, ha='center', va='center')
    else:
        T = 5
        image_paths = [os.path.join(jpg_dir, video_label, video_name, "frames", f"{i}.jpg") for i in range(T)]
        images = [mpimg.imread(path) for path in image_paths]

        h, w, c = images[0].shape

        fig, axs = plt.subplots(1, T, figsize=(5 * T, 5 * h/w))

        colors = ['green', 'magenta', 'orange']

        for i, (ax, img) in enumerate(zip(axs, images), start=1):
            ax.imshow(img)
            circ = Circle((50, 50), 40, fill=True, edgecolor='white', linewidth=2)
            ax.text(50, 50, str(i), color='white', fontsize=15, ha='center', va='center')
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
                    circ = Circle((x + bw * w - 30, y + bh * h - 30), 30, fill=True, edgecolor='white', facecolor=color, linewidth=2)
                    ax.add_patch(circ)
                    ax.text(x + bw * w - 30, y + bh * h - 30, str(j), color='white', fontsize=15, ha='center', va='center')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    save_dir = os.path.join(box_jpg_dir, video_label, video_name, label)
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
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]
        label_keyframe_bboxes = name_label_keyframe_bboxes[video_name]
        if len(label_keyframe_bboxes) == 0: # {}
            continue

        for label, keyframe_bboxes in label_keyframe_bboxes.items():
            keyframe = keyframe_bboxes['keyframe']
            bboxes = keyframe_bboxes['bboxes']
            if len(bboxes) <= 1:
                continue
            box(cfg.DATA.BASE_DIR, video_label, video_name, label, keyframe, bboxes, box_jpg_dir)

    print("concat stage 2 finish")
