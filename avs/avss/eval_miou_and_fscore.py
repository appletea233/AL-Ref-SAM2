import os
import torch
import pandas as pd
from config import cfg
from torchvision import transforms
from utils.pyutils import AverageMeter
from utils.color_utils import get_v2_pallete, load_color_mask_in_PIL_to_Tensor, calc_color_miou_fscore


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

    v2_pallete = get_v2_pallete(cfg.DATA.LABEL2IDX)

    detailed_f = open(cfg.EVAL.DETAILED_OUTPUT_PATH, 'w')

    N_CLASSES = 71
    miou_pc = torch.zeros((N_CLASSES)) # miou value per class (total sum)
    Fs_pc = torch.zeros((N_CLASSES)) # f-score per class (total sum)
    cls_pc = torch.zeros((N_CLASSES)) # count per class

    for index in range(len(df_test)):
        print("index:", index)
        df_one_video = df_test.iloc[index]
        video_name, video_label = df_one_video.iloc[1], df_one_video.iloc[6]
        gt_dir = os.path.join(cfg.DATA.BASE_DIR, video_label, video_name, "labels_rgb")
        result_dir = os.path.join(cfg.MERGE.VIS_OUTPUT_DIR, video_label, video_name)

        T = 10 if video_label == "v2" else 5

        gt_masks = []
        result_masks = []
        for img_id in range(1, T + 1):
            gt_path = os.path.join(gt_dir, f"{img_id - 1}.png")
            result_path = os.path.join(result_dir, f"{img_id}_video_mask.png")
            gt_color_mask = load_color_mask_in_PIL_to_Tensor(gt_path, v2_pallete)
            result_color_mask = load_color_mask_in_PIL_to_Tensor(result_path, v2_pallete)
            gt_masks.append(gt_color_mask)
            result_masks.append(result_color_mask)

        gt_masks = torch.stack(gt_masks, dim=0)
        result_masks = torch.stack(result_masks, dim=0)

        _miou_pc, _fscore_pc, _cls_pc, _ = calc_color_miou_fscore(result_masks, gt_masks, T)
        # compute miou, J-measure
        miou_pc += _miou_pc
        cls_pc += _cls_pc
        # compute f-score, F-measure
        Fs_pc += _fscore_pc

        batch_iou = miou_pc / cls_pc
        batch_iou[torch.isnan(batch_iou)] = 0
        batch_iou = torch.sum(batch_iou) / torch.sum(cls_pc != 0)
        batch_fscore = Fs_pc / cls_pc
        batch_fscore[torch.isnan(batch_fscore)] = 0
        batch_fscore = torch.sum(batch_fscore) / torch.sum(cls_pc != 0)
        print('index: {}, iou: {}, F_score: {}, cls_num: {}'.format(index, batch_iou, batch_fscore, torch.sum(cls_pc != 0).item()))
        detailed_f.write(f'index: {index}, iou: {batch_iou}, F_score: {batch_fscore}, cls_num: {torch.sum(cls_pc != 0).item()}\n')

    detailed_f.close()
    # breakpoint()
    miou_pc = miou_pc / cls_pc
    print(f"[test miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
    miou_pc[torch.isnan(miou_pc)] = 0
    miou = torch.mean(miou_pc).item()
    miou_noBg = torch.mean(miou_pc[:-1]).item()
    f_score_pc = Fs_pc / cls_pc
    print(f"[test fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
    f_score_pc[torch.isnan(f_score_pc)] = 0
    f_score = torch.mean(f_score_pc).item()
    f_score_noBg = torch.mean(f_score_pc[:-1]).item()

    f = open(cfg.EVAL.OUTPUT_PATH, 'a')
    f.write(f'{cfg.MERGE.VIS_OUTPUT_DIR}\n')
    print('test | cls {}, miou: {}, miou_noBg: {}, F_score: {}, F_score_noBg: {}'.format(torch.sum(cls_pc !=0).item(), miou, miou_noBg, f_score, f_score_noBg))
    f.write(f'test | cls {torch.sum(cls_pc !=0).item()}, miou: {miou}, miou_noBg: {miou_noBg}, F_score: {f_score}, F_score_noBg: {f_score_noBg}\n')
    f.close()

    print("eval finish")
