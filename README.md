# AL-Ref-SAM 2: Unleashing the Temporal-Spatial Reasoning Capacity of GPT for Training-Free Audio and Language Referenced Video Object Segmentation

[[`Paper`](https://arxiv.org/pdf/2408.15876)] [[`Project`](https://github.com/appletea233/AL-Ref-SAM2)] 


## Release Notes
*  [2024/08/29]🔥We release our [Technical Report](https://arxiv.org/pdf/2408.15876) and our training free Referring Video Object Segmetation (RVOS) code.

## TODO

* [ ] Release our training free Audio Visual Segmentation (AVS) code.
* [ ] Release online demo.

## Overall Pipeline

![AL-Ref-SAM 2 architecture](assets/pipeline.png?raw=true)



In this project, we propose an **Audio-Language-Referenced SAM 2 (AL-Ref-SAM 2)** pipeline to explore the training-free paradigm for audio and language-referenced video object segmentation, namely AVS and RVOS tasks.  We propose
a novel GPT-assisted Pivot Selection (GPT-PS) module to
instruct GPT-4 to perform two-step temporal-spatial reasoning for sequentially selecting pivot frames and pivot boxes,
thereby providing SAM 2 with a high-quality initial object
prompt. Furthermore, we propose a Language-Binded Reference Unification (LBRU) module to convert audio signals
into language-formatted references, thereby unifying the formats of AVS and RVOS tasks in the same pipeline. Extensive
experiments on both tasks show that our training-free AL-Ref-SAM 2 pipeline achieves performances comparable to
or even better than fully-supervised fine-tuning methods.


## Installation 

1. Install [SAM 2](https://github.com/facebookresearch/segment-anything-2) and [Grouding DINO](https://github.com/IDEA-Research/GroundingDINO) refer to their origin code.

2. Install other requirements by:
```
pip install -r requirements.txt
```

## Data Preparation

### Referring Video Object Segmentation

Please refer to [ReferFormer](https://github.com/wjn922/ReferFormer) for Ref-Youtube-VOS and Ref-DAVIS17 data preparation and refer to [MeViS](https://github.com/henghuiding/MeViS) for MeViS data preparation.

## Get Started

### Referring Video Object Segmentation

For Ref-Youtube-VOS and MeViS dataset, you need to check the code in `rvos/ytvos`. For Ref-DAVIS17 dataset, you need to check the code in `rvos/davis`.

Please first check and change the config settings under the `opt.py` in the corresponding folder. 
Next, run the code in the order shown in the diagram.

![code pipeline](assets/code_pipeline.png?raw=true)

## Citation
```
@article{huang2024unleashing,
      title={Unleashing the Temporal-Spatial Reasoning Capacity of GPT for Training-Free Audio and Language Referenced Video Object Segmentation}, 
      author={Huang, Shaofei and Ling, Rui and Li, Hongyu and Hui, Tianrui and Tang, Zongheng and Wei, Xiaoming and Han, Jizhong and Liu, Si},
      journal={arXiv preprint arXiv:2408.15876},
      year={2024},
}
```