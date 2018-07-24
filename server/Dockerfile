FROM c2-cuda9-cudnn7-wdata

# Copy the weights
RUN mkdir /densepose/weights
COPY ../weights /densepose/weights
# Install Server Dependencies
RUN pip install cython common flask flask_cors flask_socketio pillow gevent