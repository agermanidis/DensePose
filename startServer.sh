nvidia-docker run --rm -it -v /home/paperspace/DensePose/DensePoseData:/denseposedata \
  -v /home/paperspace/Datasets/coco:/coco \
  densepose:c2-cuda9-cudnn7-wdata bash