nvidia-docker run --rm -it -v /home/paperspace/DensePose/DensePoseData:/denseposedata \
  -v /home/paperspace/Datasets/coco:/coco \ 
  -v /home/paperspace/DensePose/server/:/densepose/server \
  -p 22100:22100 \
  --entrypoint "bash" denseposeserver:latest