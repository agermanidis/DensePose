# Simple socket client to test with pix2pix server
from socketIO_client import SocketIO, BaseNamespace, LoggingNamespace

class QueryNamespace(BaseNamespace):
  def query_response(self, *args):
    print('query response', args)

def handle_pix2pix_response(*args):
  print('handle_pix2pix', args)

socketIO = SocketIO('127.0.0.1', 23100)
socketIO.on('update_response', handle_pix2pix_response)

query_namespace = socketIO.define(QueryNamespace, '/query')
query_namespace.emit('update_request', {'data': 'yyy'})

#with SocketIO('127.0.0.1', 23100, LoggingNamespace) as socketIO:
 # socketIO.emit('update_request', {'data': 'yyy'}, handle_pix2pix)
 # socketIO.wait_for_callbacks(seconds=4)
