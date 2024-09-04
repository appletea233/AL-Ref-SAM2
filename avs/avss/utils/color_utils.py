import os
import json
import torch
import numpy as np
from PIL import Image

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete

def save_color_mask(pred_masks, save_dir, v_pallete, T=10):
    # pred_masks: [T, h, w]
    N_CLASSES = 71
    os.makedirs(save_dir, exist_ok=True)

    T, H, W = pred_masks.shape

    pred_masks = pred_masks.cpu().numpy()

    pred_rgb_masks = np.zeros((pred_masks.shape + (3,)), np.uint8)
    for cls_idx in range(N_CLASSES):
        rgb = v_pallete[cls_idx]
        pred_rgb_masks[pred_masks == cls_idx] = rgb
    pred_rgb_masks = pred_rgb_masks.reshape(T, H, W, 3)

    for video_id in range(len(pred_rgb_masks)): # T
        one_mask = pred_rgb_masks[video_id]
        im = Image.fromarray(one_mask)
        im.save(os.path.join(save_dir, f"{video_id + 1}_video_mask.png"), format='PNG')

def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    color_mask_PIL = resize_img(224, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label) # [H, W]
    color_label = color_label.unsqueeze(0)
    # binary_mask = (color_label != (cfg.NUM_CLASSES-1)).float()
    # return color_label, binary_mask # both [1, H, W]
    return color_label # both [1, H, W]

def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W], # now: [BF, H, W]
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    # predict = torch.argmax(output, 1) + 1
    predict = output.float() + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list

def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background # now [BF x H x W]
            target: size [BF x H x W]
    """  
    nclass = 71
    # pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T) 
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass, T) 
    return miou, fscore, cls_count, vid_miou_list
