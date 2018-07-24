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
import common
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

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
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Densepose args
cfg_file = 'configs/DensePose_ResNet101_FPN_s1x-e2e.yaml'
weights_file = 'weights/DensePose_ResNet101_FPN_s1x-e2e.pkl'
output_dir = 'DensePoseData/infer_out/'
image_ext = 'jpg'
im_or_folder = ''
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)
logger = logging.getLogger(__name__)
merge_cfg_from_file(cfg_file)
cfg.NUM_GPUS = 1
# weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
# assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weights_file)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()

# Server configs
PORT = 22100
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Take in base64 string and return PIL image
def stringToImage(base64_string):
  imgdata = base64.b64decode(base64_string)
  return PILImage.open(io.BytesIO(imgdata))

# Convert PIL Image to an RGB image(technically a numpy array) that's compatible with opencv
def toRGB(image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def main(input_img):
  image = stringToImage(input_img[input_img.find(",")+1:])
  image = toRGB(image)
  img = Image(image) 
  logger.info('Processing {} -> {}'.format('New Image', 'Output...'))
  timers = defaultdict(Timer)
  t = time.time()
  with c2_utils.NamedCudaScope(0):
    cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
      model, img, None, timers=timers
    )
  logger.info('Inference time: {:.3f}s'.format(time.time() - t))
  for k, v in timers.items():
    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
  vis_utils.vis_one_image(
    img[:, :, ::-1],  # BGR -> RGB for visualization
    'testImage',
    output_dir,
    cls_boxes,
    cls_segms,
    cls_keyps,
    cls_bodys,
    dataset=dummy_coco_dataset,
    box_alpha=0.3,
    show_class=True,
    thresh=0.7,
    kp_thresh=2
  )
  return 'true'

# --- 
# Server Routes
# --- 

# Base route, functions a simple testing 
@app.route('/')
def index():
  return jsonify(status="200", message='Densepose is running', query_route='/query', test_route='/test')

# Test the model with a fix to see if it's working
@app.route('/test')
def query():
  results = main(None)
  return jsonify(status="200", model='Densepose', response=results)

# When a client socket connects
@socketio.on('connect', namespace='/query')
def new_connection():
  emit('successful_connection', {"data": "connection established"})

# When a client socket disconnects
@socketio.on('disconnect', namespace='/query')
def disconnect():
  print('Client Disconnect')

# When a client sends data. This should call the main() function
@socketio.on('update_request', namespace='/query')
def new_request(request):
  results = main(request["data"])
  emit('update_response', {"results": results})

if __name__ == '__main__':
  socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
