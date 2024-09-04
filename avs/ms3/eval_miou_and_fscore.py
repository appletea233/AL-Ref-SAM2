import os
import pandas as pd
from PIL import Image
from config import cfg
from torchvision import transforms
from utils.pyutils import AverageMeter
from utils.utility import mask_iou, Eval_Fmeasure

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


if __name__ == '__main__':
    print("eval start")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    avg_miou = AverageMeter(['miou']) # MJ
    avg_F = AverageMeter(['F_score']) # MF

    df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
    df_test = df_all[df_all['split'] == cfg.SPLIT]

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    detailed_f = open(cfg.EVAL.DETAILED_OUTPUT_PATH, 'w')
    T = 5
    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name = df_one_video.iloc[0]
        gt_dir = os.path.join(cfg.DATA.GT_MASK_DIR, video_name)
        result_dir = os.path.join(cfg.MERGE.VIS_OUTPUT_DIR, video_name)

        for img_id in range(1, T + 1):
            gt_path = os.path.join(gt_dir, f"{video_name}_{img_id}.png")
            result_path = os.path.join(result_dir, f"{img_id}_video_mask.png")

            gt_mask = load_image_in_PIL_to_Tensor(gt_path, mode='1', transform=mask_transform)
            result_mask = load_image_in_PIL_to_Tensor(result_path, mode='1', transform=mask_transform)
            gt_mask = gt_mask.cuda()
            result_mask = result_mask.cuda()

            miou = mask_iou(result_mask, gt_mask)
            avg_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(result_mask, gt_mask, None)
            avg_F.add({'F_score': F_score})
            print('index: {}, img_id: {}, iou: {}, F_score: {}'.format(index, img_id, miou, F_score))
            detailed_f.write(f'video_name: {video_name}, img_id: {img_id}, iou: {miou}, F_score: {F_score}\n')

    detailed_f.close()
    # breakpoint()
    miou = (avg_miou.pop('miou'))
    F_score = (avg_F.pop('F_score'))
    print('miou:', miou.item())
    print('F_score:', F_score)

    f = open(cfg.EVAL.OUTPUT_PATH, 'a')
    f.write(f'{cfg.MERGE.VIS_OUTPUT_DIR}\n')
    f.write(f'miou / F_score: {miou.item()} / {F_score}\n')
    f.close()

    print("eval finish")