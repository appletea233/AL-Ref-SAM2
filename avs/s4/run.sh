# obtain audio labels
python beats.py
python png_to_jpg.py
# data preprocessing for LBRU and pivot frame selection
python concat_stage_1.py
# LBRU and pivot frame selection
python gpt_cot_inference_stage_1.py
# obtain candidate boxes with Grounding Dino
python gdino.py
# data preprocessing for pivot box selection
python concat_stage_2.py
# pivot box selection
python gpt_cot_inference_stage_2.py
# video segmentation with SAM 2
python sam2.py
# evaluation
python eval_miou_and_fscore.py