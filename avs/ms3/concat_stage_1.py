import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from config import cfg

def concat(jpg_dir, video_name, concat_jpg_dir):
    T = 5
    image_paths = [os.path.join(jpg_dir, video_name, f"{i}.jpg") for i in range(1, T + 1)]
    images = [mpimg.imread(path) for path in image_paths]

    h, w, c = images[0].shape

    fig, axs = plt.subplots(1, T, figsize=(5 * T, 5 * h/w))

    for i, (ax, img) in enumerate(zip(axs, images), start=1):
        ax.imshow(img)
        circ = Circle((20, 20), 15, fill=True, edgecolor='white', linewidth=2)
        ax.text(20, 20, str(i), color='white', fontsize=20, ha='center', va='center')
        ax.add_patch(circ)
        ax.axis('off')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0)
    save_dir = os.path.join(concat_jpg_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "concat.jpg"))
    plt.close()


if __name__ == "__main__":
    print("concat stage 1 start")

    concat_jpg_dir = cfg.DATA.CONCAT_1_JPG_IMG_DIR
    os.makedirs(concat_jpg_dir, exist_ok=True)

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name = df_one_video.iloc[0]
        concat(cfg.DATA.JPG_IMG_DIR, video_name, concat_jpg_dir)

    print("concat stage 1 finish")
