#!/bin/bash
python ./demo/image_demo.py \
./resources/000000000139.jpg \
./configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py \
./weights/iter_180000.pth \
--output runs/inference/exp