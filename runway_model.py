from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import json
import base64
import io
import numpy as np
import runway
from runway.data_types import image
from PIL import Image

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
from vis_utils import vis_one_image

c2_utils.import_detectron_ops()

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Densepose args
cfg_file = 'configs/DensePose_ResNet50_FPN_s1x-e2e.yaml'
weights_file = 'weights/DensePose_ResNet50_FPN_s1x-e2e.pkl'
output_dir = 'DensePoseData/infer_out/'
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)
logger = logging.getLogger(__name__)
merge_cfg_from_file(cfg_file)
cfg.NUM_GPUS = 1
cfg.TEST.BBOX_AUG.ENABLED = False
cfg.MODEL.MASK_ON = False
cfg.MODEL.KEYPOINTS_ON = False

assert_and_infer_cfg(cache_urls=False)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()

@runway.setup
def setup():
  return infer_engine.initialize_model_from_cfg(weights_file)

  
@runway.command('visualize', inputs={'input': image}, outputs={'output': image})
def visualize(model, inputs):
  img = np.array(inputs['input'])
  with c2_utils.NamedCudaScope(0):
    cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
      model, img, None
    )
  output = vis_one_image(img, 'testImage', output_dir, cls_boxes, cls_segms, cls_keyps, cls_bodys, dataset=dummy_coco_dataset, box_alpha=0.3, show_class=True, thresh=0.7, kp_thresh=2)
  b, g, r = output.split()
  return Image.merge("RGB", (r, g, b))


if __name__ == '__main__':
  runway.run()