# Custom docker file for the server
FROM 0970c8793bfc

# Copy the weights
RUN mkdir /densepose/weights && mkdir /densepose/server
COPY weights /densepose/weights
COPY server /densepose/server
# Install Server Dependencies
RUN pip install cython common flask flask_cors flask_socketio pillow gevent

ENTRYPOINT [ "python2", "server/server.py" ]