# Custom docker file for the server
FROM 0970c8793bfc

RUN apt-get update && apt-get install -y nano && apt-get install -y iputils-ping

# Copy the weights
RUN mkdir /densepose/weights && mkdir /densepose/server
COPY weights /densepose/weights
COPY server /densepose/server
# Install Server Dependencies
RUN pip install cython common flask flask_cors flask_socketio pillow gevent requests socketIO-client

ENTRYPOINT [ "python2", "server/server.py" ]
