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
# obtain semantic labels for each frame
python languagebind_stage.py
# merge masks of different sounding objects
python merge.py
# evaluation
python eval_miou_and_fscore.py