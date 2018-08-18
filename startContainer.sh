nvidia-docker run --rm -it -v /home/paperspace/DensePose/DensePoseData:/denseposedata \
  -v /home/paperspace/Datasets/coco:/coco \
  -v /home/paperspace/DensePose/server/:/densepose/server \
  -v /storage:/storage \
  -v /home/paperspace/DensePose/weights:/densepose/weights \
  -v /home/paperspace/DensePose/tools:/densepose/tools \
  --entrypoint "bash" denseposeserver:latest
