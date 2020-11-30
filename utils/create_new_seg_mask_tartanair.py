import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import os
from os.path import join
import sys
sys.path.append('/home/frederik/detectron2')
from collections import Counter

# Some basic setup:
# Setup detectron2 logger
import argparse
import copy
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from tqdm import tqdm

def remap_seg(seg, uniq_seg_ids, id_to_color):
    
    H, W = seg.shape
    seg_new = np.zeros_like(seg)
    seg_color =  np.zeros((H,W,3), dtype=np.uint8)
    
    for c in np.unique(seg):
        
        mask = seg == c
        count = uniq_seg_ids[c]
        label = count.most_common()[0][0]
        seg_new[mask] = label
        seg_color[mask, :] = id_to_color[label]
        
        #print(c, label, uniq_seg_ids[c].most_common()[0][1], id_to_color[label])
        #plt.imshow(mask)
        #plt.show()
        
    return seg_new,seg_color

def aggregate_seg(seg, panoptic_seg, uniq_seg_ids, segments_info, id_to_stuff, id_to_thing):

    panoptic_seg = panoptic_seg.cpu().numpy()
    
    for info in segments_info:
        if info['isthing']:
            label = id_to_thing[info['category_id']]
        else:
            label = id_to_stuff[info['category_id']]
        
        panoptic_seg[panoptic_seg == info['id']] = label
    
    for c in np.unique(seg):
        mask = seg == c
        count = Counter(panoptic_seg[mask])
        
        if c in uniq_seg_ids:
            uniq_seg_ids[c] += count
        else:
            uniq_seg_ids[c] = count
    
    return uniq_seg_ids
    
if __name__ == "__main__":
    

    # init model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)

    cat = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    id_to_stuff = {}
    id_to_thing = {}
    id_to_color = {}
    for key in cat.stuff_dataset_id_to_contiguous_id:
        id_to_stuff[cat.stuff_dataset_id_to_contiguous_id[key]] = key
        id_to_color[key] = cat.stuff_colors[cat.stuff_dataset_id_to_contiguous_id[key]]

    for key in cat.thing_dataset_id_to_contiguous_id:
        id_to_thing[cat.thing_dataset_id_to_contiguous_id[key]] = key
        id_to_color[key] = cat.thing_colors[cat.thing_dataset_id_to_contiguous_id[key]]

    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type=str, required=True)
    args = parser.parse_args()

    for split in ['test','val','train']:
        for env in os.listdir(os.path.join(args.basepath, split)):
            for seq in os.listdir(os.path.join(args.basepath, split, env, 'Easy')):
                seqpath = os.path.join(args.basepath, split, env, 'Easy', seq)

                uniq_seg_ids = {}
                for i in tqdm(sorted(os.listdir(os.path.join(seqpath,"ground_truth/seg0/data/")))):
                    
                    i = i.replace(".npy", "")
                    impath = os.path.join(seqpath,"cam0/data/", i + ".png")
                    segpath = os.path.join(seqpath,"ground_truth/seg0/data/", i + ".npy")
                    
                    im = np.asarray(Image.open(impath))
                    seg = np.load(segpath)
                    
                    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]        
                    uniq_seg_ids = aggregate_seg(copy.deepcopy(seg), panoptic_seg.clone(), uniq_seg_ids,
                                            segments_info, id_to_stuff, id_to_thing)
                    
                tmp = os.path.join(seqpath,"ground_truth/new_seg0/data/")
                if not os.path.isdir(tmp): 
                    os.makedirs(tmp)
                
                for i in tqdm(sorted(os.listdir(os.path.join(seqpath,"ground_truth/seg0/data/")))):
                    
                    i = i.replace(".npy", "")
                    segpath = os.path.join(seqpath,"ground_truth/seg0/data/", i + ".npy")
                    new_segpath = os.path.join(seqpath,"ground_truth/new_seg0/data/", i + ".npy")
                    
                    seg = np.load(segpath)
                    new_seg, new_seg_color = remap_seg(copy.deepcopy(seg), uniq_seg_ids, id_to_color)
                    
                    np.save(new_segpath, new_seg)