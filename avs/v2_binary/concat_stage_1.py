import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from config import cfg

def concat(jpg_dir, video_name, concat_jpg_dir):
    T = 10
    image_paths = [os.path.join(jpg_dir, video_name, "frames", f"{i}.jpg") for i in range(T)]
    images = [mpimg.imread(path) for path in image_paths]

    h, w, c = images[0].shape

    fig, axs = plt.subplots(2, 5, figsize=(5 * 5, 5 * 2 * h/w))

    for i, (ax, img) in enumerate(zip(axs.flat, images), start=1):
        ax.imshow(img)
        circ = Circle((50, 50), 40, fill=True, edgecolor='white', linewidth=2)
        ax.text(50, 50, str(i), color='white', fontsize=15, ha='center', va='center')
        ax.add_patch(circ)
        ax.axis('off')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    save_dir = os.path.join(concat_jpg_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "concat.jpg"))
    plt.close()


if __name__ == "__main__":
    print("concat stage 1 start")

    concat_jpg_dir = cfg.DATA.CONCAT_1_JPG_IMG_DIR
    os.makedirs(concat_jpg_dir, exist_ok=True)

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_v2 = df_all[df_all['label'] == cfg.LABEL]
    df_v2_test = df_v2[df_v2['split'] == cfg.SPLIT]

    for index in range(len(df_v2_test)):
        print("index:", index)
        df_one_video = df_v2_test.iloc[index]
        video_name = df_one_video.iloc[1]
        concat(cfg.DATA.DIR, video_name, concat_jpg_dir)

    print("concat stage 1 finish")
