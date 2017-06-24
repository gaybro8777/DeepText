#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__','ambiguous_text','text')

demo_text_images_dir = 'demo_text_images/'
text_detection_results_dir = 'text_detection_results/'

def com_overlap(bbox_idx, bbox_jdx):
    x1i = int(round(bbox_idx[0]))
    y1i = int(round(bbox_idx[1]))
    x2i = int(round(bbox_idx[2]))
    y2i = int(round(bbox_idx[3]))
    areai = (x2i - x1i + 1) * (y2i - y1i + 1)

    x1j = int(round(bbox_jdx[0]))
    y1j = int(round(bbox_jdx[1]))
    x2j = int(round(bbox_jdx[2]))
    y2j = int(round(bbox_jdx[3]))

    xx1 = np.maximum(x1i, x1j)
    yy1 = np.maximum(y1i,y1j)
    xx2 = np.minimum(x2i,x2j)
    yy2 = np.minimum(y2i,y2j)
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    overlap = inter / areai;
    return overlap

def bbox_adjust(dets):
    keep = [n for n in range(len(dets))]
    for idx in range(len(dets)):
        si = dets[idx,4]
        for jdx in range(len(dets)):
            if jdx==idx:
                continue
            sj = dets[jdx,4]
            overlap = com_overlap(dets[idx, :4], dets[jdx, :4])
            th = np.minimum(sj+0.1, 1.0)
            if ((overlap>0.7) and (si<=th)):
                if idx in keep:
                    keep.remove(idx)
                break
            if ((overlap>0.7) and (si>th)):
                if jdx in keep:
                    keep.remove(jdx)
                break
    return keep

def demo_textdetection(net, image_name, cls):
    # Load the demo image
    im_file=os.path.join(demo_text_images_dir, image_name)
    im = cv2.imread(im_file)

    CONF_THRESH = 0.70
    NMS_THRESH = 0.3
    linewidth = 2

    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    cls_ind = CLASSES.index(cls)
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    nms_dets = dets[keep, :]
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} text proposals').format(timer.total_time, boxes.shape[0])

    combine_idx = bbox_adjust(nms_dets)
    nms_dets = nms_dets[combine_idx, :]

    keep = np.where(nms_dets[:, -1] >= CONF_THRESH)[0]
    nms_dets = nms_dets[keep, :]

    detection_result= text_detection_results_dir+image_name.split('.', -1)[0]+'.txt'

    if nms_dets.shape[0] == 0:
        with open(detection_result,"w") as f:
            f.close()
        print('{} has no text detection').format(image_name)

    with open(detection_result,"w") as f:
        for bbox in nms_dets:
            bbox[:4] = np.round(bbox[:4])
            f.write(str(int(bbox[0]))+',')
            f.write(str(int(bbox[1]))+',')
            f.write(str(int(bbox[2]))+',')
            f.write(str(int(bbox[3]))+'\n')
            cv2.rectangle(im, tuple(bbox[:2]), tuple(bbox[2:4]), (0,0,255), linewidth)
            cv2.putText(im, '{:.3f}'.format(bbox[4]), (bbox[0], int(np.maximum(bbox[1]-5, 0))), 0, 0.5, (0,255,255), linewidth)
    f.close()
    cv2.imwrite(text_detection_results_dir+image_name.split('.', -1)[0]+'.jpg', im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = (600,)
    cfg.TEST.MAX_SIZE = 1000
    cfg.TEST.RPN_NMS_THRESH = 0.7
    cfg.TEST.RPN_MIN_SIZE = 8
    args = parse_args()

    prototxt = 'models/text_detection/test_deep_text.pt'

    caffemodel = 'models/text_detection/vgg16_deep_text_trained_model.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    included_extenstions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']
    image_names = [fn for fn in os.listdir(demo_text_images_dir)
              if any(fn.endswith(ext) for ext in included_extenstions)]

    nimgs = len(image_names)
    print "totally "+str(nimgs)+" images"
    for i, image_name in enumerate(image_names):
        demo_textdetection(net, image_name, 'text')
        print i, ":", nimgs, " ", image_name, "has converted! "

