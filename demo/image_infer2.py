# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import asyncio
import glob
import os
import os.path as osp
from argparse import ArgumentParser
import cv2 as cv
from copy import deepcopy
import json

from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot

from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file, such as: /path/to/sample.jpg or '/path/to/*.jpg'")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="bbox score threshold"
    )
    parser.add_argument(
        "--async-test",
        action="store_true",
        help="whether to set async options for async inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="specify the directory to save visualization results.",
    )
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
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    imgs = glob.glob(args.img)


    if not os.path.exists(args.output):
        os.makedirs(osp.join(args.output, "images"))
        os.makedirs(osp.join(args.output, "labels_json"))

    for img in imgs:
        # test a single image
        result = inference_detector(model, img)

        # show the results
        if args.output is None:
            show_result_pyplot(model, img, result, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            # save_result(
            #     model, img, result, score_thr=args.score_thr, out_file=out_file_path
            # )

            imgs = cv.imread(img)
            img_h, img_w = imgs.shape[0], imgs.shape[1]


            det_img, shapes = make_result(img, result, ids=[0], score_thr=args.score_thr)
            
            labelme_result = {
                "version": "",
                "flags": {},
                "shapes": shapes,
                "imagePath": img,
                "imageData": None,
                "imageWidth": img_w,
                "imageHeight": img_h
            }

            save_label_path = osp.join(args.output, "labels_json", osp.basename(img))
            save_img_path = osp.join(args.output, "images", osp.basename(img))
            # print(labelme_result)
            if len(shapes) != 0:
                save_json(labelme_result, save_label_path.replace("jpg", "json"))
                save_image(det_img, save_img_path)
            
    print(f"All result save to {args.output}")


async def async_main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    # test a single image
    args.img = glob.glob(args.img)
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    for img, pred in zip(args.img, result):
        if args.output is None:
            show_result_pyplot(model, img, pred, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, pred, score_thr=args.score_thr, out_file=out_file_path
            )

            


if __name__ == "__main__":
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
