import os
import os.path as osp
import numpy as np
import cv2 as cv
import json
from copy import deepcopy
from argparse import ArgumentParser

from mmdet.apis import init_detector, inference_detector
from mmcv import Config
from ssod.utils import patch_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', default="", help='Image file')
    parser.add_argument('config', default="", help='Config file')
    parser.add_argument('checkpoint', default="", help='Checkpoint file')
    parser.add_argument('--output', default="", help='Path to output file')
    parser.add_argument('--device', default="cuda:0", help='Device used for inference such as cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.2, help='bbox score threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    args = parser.parse_args()
    
    return args


def make_result(img_path, results, ids=None, score_thr=0.2):
    shapes = []
    img = cv.imread(img_path)
    det = deepcopy(img)
    if ids == None:
        ids = range(len(results))
    for id, bbox in enumerate(results):
        if len(ids) != 0 and id in ids:
            cotegory_id = id
            for point in bbox:
                if point[-1] > score_thr:
                    point = [int(i) for i in point]
                    cv.rectangle(det, (point[0], point[1]), (point[2], point[3]), (0, 0, 255), 2)
                    
                    point = [float(i) for i in point]
                    shape = {
                        "label": "person",
                        "is_verify": None,
                        "points": [[point[0], point[1]],[point[2], point[3]]],
                        "score": point[4],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    shapes.append(shape)

    return det, shapes


def save_image(img, save_path):
    cv.imwrite(save_path, img)


def save_json(json_file, save_path):
    with open(save_path, "w", encoding="utf8") as fp:
        json.dump(json_file, fp, indent=2)


def main(args):
    img_dir = args.img
    # configs = args.configs
    checkpoints = args.checkpoint
    out_dir = args.output
    device = args.device
    score_thr = args.score_thr
    classes = args.classes

    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)

    if not os.path.exists(out_dir):
        os.makedirs(osp.join(out_dir, "images"))
        os.makedirs(osp.join(out_dir, "labels_json"))

    img_list = sorted(os.listdir(img_dir))
    model = init_detector(cfg, checkpoints, device=device)
    count = 0
    for img_name in img_list:
        img_path = osp.join(img_dir, img_name)
        count += 1
        print(f'model is processing the {count}/{len(img_list)} images.')
        result = inference_detector(model, img_path)

        img = cv.imread(img_path)
        img_h, img_w = img.shape[0], img.shape[1]


        det_img, shapes = make_result(img_path, result, ids=classes, score_thr=score_thr)
        
        labelme_result = {
            "version": "",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path,
            "imageData": None,
            "imageWidth": img_w,
            "imageHeight": img_h
        }

        save_label_path = osp.join(out_dir, "labels_json", img_name)
        save_img_path = osp.join(out_dir, "images", img_name)
        # print(labelme_result)
        if len(shapes) != 0:
            save_json(labelme_result, save_label_path.replace("jpg", "json"))
            save_image(det_img, save_img_path)
        
    print(f"All result save to {out_dir}")


if __name__ == "__main__":
    opts = parse_args()
    
    main(opts)
