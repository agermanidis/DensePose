nvidia-docker run --rm -it -v /home/paperspace/DensePose/DensePoseData:/denseposedata \
  -v /home/paperspace/Datasets/coco:/coco \
  -v /storage:/storage \
  -v /home/paperspace/DensePose/tools:/densepose/tools \
  densepose:c2-cuda9-cudnn7-wdata bash
