import websocket, httplib, sys, asyncore

server = '65.19.181.36'
port = 23100

def _onopen():
  print("opened!")

def _onmessage(msg):
  print("msg: " + str(msg))

def _onclose():
  print("closed!")

def connect(server, port):
  print("connecting to: %s:%d" %(server, port))
  conn  = httplib.HTTPConnection(server + ":" + str(port))
  conn.request('POST','/socket.io/1/')
  resp  = conn.getresponse()
  hskey = resp.read().split(':')[0]
  ws = websocket.WebSocket(
                  'ws://'+server+':'+str(port)+'/socket.io/1/websocket/'+hskey,
                  onopen   = _onopen,
                  onmessage = _onmessage,
                  onclose = _onclose)
  return ws
