import os
from easydict import EasyDict as edict

cfg = edict()

cfg.SETTING = "ms3"
cfg.SPLIT = "test"
cfg.PREFIX = "AL-Ref-SAM 2"
cfg.OUTPUT_DIR = os.path.join("../results", cfg.SETTING, cfg.PREFIX)
cfg.OBJECTS = ['accordion', 'airplane', 'axe', 'baby', 'bassoon', 'bell', 'bird', 'boat', 'boy', 'bus',
               'car', 'cat', 'cello', 'clarinet', 'clipper', 'clock', 'dog', 'donkey', 'drum', 'duck',
               'elephant', 'emergency-car', 'erhu', 'flute', 'frying-food', 'girl', 'goose', 'guitar', 'gun', 'guzheng',
               'hair-dryer', 'handpan', 'harmonica', 'harp', 'helicopter', 'hen', 'horse', 'keyboard', 'leopard', 'lion',
               'man', 'marimba', 'missile-rocket', 'motorcycle', 'mower', 'parrot', 'piano', 'pig', 'pipa', 'saw',
               'saxophone', 'sheep', 'sitar', 'sorna', 'squirrel', 'tabla', 'tank', 'tiger', 'tractor', 'train',
               'trombone', 'truck', 'trumpet', 'tuba', 'ukulele', 'utv', 'vacuum-cleaner', 'violin', 'wolf', 'woman']

# beats
cfg.BEATS = edict()
cfg.BEATS.CKPT_20K = "../checkpoints/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt2.pt"
cfg.BEATS.CKPT_2M = "../checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
cfg.BEATS.ONTOLOGY = "../beats_model/ontology.json"
cfg.BEATS.OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, "beats_outputs.json")

# grounding dino
cfg.GDINO = edict()
cfg.GDINO.CONFIG = "GDINO_CONFIG" # TODO
cfg.GDINO.CKPT = "GDINO_CKPT" # TODO
cfg.GDINO.BOX_TRESHOLD = 0.25
cfg.GDINO.TEXT_TRESHOLD = 0.25
cfg.GDINO.OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, "gdino_outputs.json")
cfg.GDINO.VIS_OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "gdino_vis", cfg.SPLIT)

# sam2
cfg.SAM2 = edict()
cfg.SAM2.CONFIG = "SAM2_CONFIG" # TODO
cfg.SAM2.CKPT = "SAM2_CKPT" # TODO
cfg.SAM2.VIS_OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "sam2_vis", cfg.SPLIT)

# languagebind
cfg.LANGUAGEBIND = edict()
cfg.LANGUAGEBIND.AUDIO_SEGMENT_FILE = "../audio_segment/ms3_audio_segment.json"
cfg.LANGUAGEBIND.AUDIO_SEGMENT_WAV_DIR = os.path.join("../audio_segment_wav", cfg.SETTING, cfg.SPLIT)
cfg.LANGUAGEBIND.LABEL4SECOND = os.path.join(cfg.OUTPUT_DIR, "label4second.json")

# merge
cfg.MERGE = edict()
cfg.MERGE.VIS_OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "merge_vis", cfg.SPLIT)

# gpt
cfg.GPT = edict()
cfg.GPT.GPT_API_KEY = "YOUR API KEY" # TODO
cfg.GPT.GPT_API_BASE = "YOUR API BASE" # TODO

# gpt cot inference stage 1
cfg.GPT_STAGE_1 = edict()
cfg.GPT_STAGE_1.NTHREAD = 1
cfg.GPT_STAGE_1.OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, "cot_stage_1_outputs.json")

# gpt cot inference stage 2
cfg.GPT_STAGE_2 = edict()
cfg.GPT_STAGE_2.NTHREAD = 1
cfg.GPT_STAGE_2.OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, "cot_stage_2_outputs.json")

# data
cfg.DATA = edict()
cfg.DATA.BASE_DIR = "YOUR AVSBench Multi-sources DIR" # TODO
cfg.DATA.ANNO_CSV = os.path.join(cfg.DATA.BASE_DIR, "ms3_meta_data.csv")
cfg.DATA.PNG_IMG_DIR = os.path.join(cfg.DATA.BASE_DIR, "ms3_data", "visual_frames")
cfg.DATA.AUDIO_WAV_DIR = os.path.join(cfg.DATA.BASE_DIR, "ms3_data", "audio_wav", cfg.SPLIT)
cfg.DATA.GT_MASK_DIR = os.path.join(cfg.DATA.BASE_DIR, "ms3_data", "gt_masks", cfg.SPLIT)
cfg.DATA.JPG_IMG_DIR = os.path.join("../jpg", cfg.SETTING, cfg.SPLIT)
cfg.DATA.CONCAT_1_JPG_IMG_DIR = os.path.join("../concat_1_jpg", cfg.SETTING, cfg.SPLIT)
cfg.DATA.CONCAT_2_JPG_IMG_DIR = os.path.join("../concat_2_jpg", cfg.SETTING, cfg.SPLIT)

# eval
cfg.EVAL = edict()
cfg.EVAL.OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, "miou_and_fscore.txt")
cfg.EVAL.DETAILED_OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, "detailed_miou_and_fscore.txt")


if __name__ == "__main__":
    print(cfg)
    breakpoint()
