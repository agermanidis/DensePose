nvidia-docker run --rm -it -v /home/paperspace/DensePose/DensePoseData:/denseposedata \
  -v /home/paperspace/Datasets/coco:/coco \
  -p 22100:22100 \
  denseposeserver:latest