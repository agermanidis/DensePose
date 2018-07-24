# Custom docker file for the server
FROM 0970c8793bfc

# Copy the weights
RUN mkdir /densepose/weights
COPY weights /densepose/weights
# Install Server Dependencies
RUN pip install cython common flask flask_cors flask_socketio pillow gevent